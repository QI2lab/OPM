import glob
import os
import datetime
import time
import numpy as np
import pickle
import localize
import tifffile
import pygpufit.gpufit as gf
import napari
import matplotlib.pyplot as plt

tbegin = time.perf_counter()

# paths to image files
root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210409a", "second_round_r9_r10")
data_fname = os.path.join(root_dir, "second_round_MMStack_Default.ome.tif")

plot_results = True
plot_centers = True
plot_centers_guess = True

# ###############################
# load/set scan parameters
# ###############################
nvols = 1
nchannels = 3
channels_to_use = [True, False, False]
nimgs_per_vol = 55
nyp = 2048
nxp = 2048
dc = 0.065
dz = 0.230
theta = np.pi / 2

volume_um3 = (dz * nimgs_per_vol) * (nyp * dc) * (nxp * dc)

frame_time_ms = None
na = 1.35
ni = 1.4
excitation_wavelength = 0.561
emission_wavelength = 0.605
sigma_xy = 0.22 * emission_wavelength / na
sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

# ###############################
# build save dir
# ###############################
now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
# time_stamp = "test"

save_dir = os.path.join(root_dir, "..", "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ###############################
# identify candidate points in opm data
# ###############################
absolute_threshold = 50
# difference of gaussian filer
filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
filter_sigma_large = (5 * sigma_z, 20 * sigma_xy, 20 * sigma_xy)
# fit roi size
roi_size = (3 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
roi_size_pix = localize.get_roi_size(roi_size, dc, dz)
# assume points closer together than this come from a single bead
min_dists = (3 * sigma_z, 2 * sigma_xy, 2 * sigma_xy)
# exclude points with sigmas outside these ranges
sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
sigmas_max = (2 * sigma_z, 2 * sigma_xy, 2 * sigma_xy)

# don't consider any points outside of this polygon
# cx, cy
# allowed_camera_region = np.array([[357, 0], [464, 181], [1265, 231], [1387, 0]])
allowed_camera_region = None

# ###############################
# loop over volumes
# ###############################

volume_process_times = np.zeros(nvols)
chunk_size = 555
chunk_overlap = 3
more_chunks = True
for vv in range(nvols):
    print("################################\nstarting volume %d/%d" % (vv + 1, nvols))
    tstart_vol = time.perf_counter()

    centers_vol = []
    fit_params_vol = []
    rois_vol = []
    centers_guess_vol = []

    ichunk = 0
    chunk_counter_x = 0
    chunk_counter_y = 0
    while more_chunks:
        print("processing chunk %d" % ichunk)
        tstart_chunk = time.perf_counter()

        tstart_load = time.perf_counter()
        ix_start = int(np.max([chunk_counter_x * chunk_size - chunk_overlap, 0]))
        ix_end = int(np.min([ix_start + chunk_size, nxp]))

        iy_start = int(np.max([chunk_counter_y * chunk_size - chunk_overlap, 0]))
        iy_end = int(np.min([iy_start + chunk_size, nyp]))

        imgs = np.squeeze(tifffile.imread(data_fname)[:, channels_to_use, iy_start:iy_end, ix_start:ix_end])

        tend_load = time.perf_counter()
        print("loaded chunk in %0.2fs" % (tend_load - tstart_load))

        # get image coordinates
        x, y, z, = localize.get_coords(imgs.shape, dc, dz)
        x += dc * ix_start
        y += dc * iy_start

        # filter
        ks = localize.get_filter_kernel(filter_sigma_small, dc, dz, sigma_cutoff=2)
        kl = localize.get_filter_kernel(filter_sigma_large, dc, dz, sigma_cutoff=2)
        imgs_filtered = localize.filter_convolve(imgs, ks) - localize.filter_convolve(imgs, kl)

        # find candidate peaks
        footprint = localize.get_footprint(min_dists, dc, dz)
        centers_guess_inds = localize.find_peak_candidates(imgs_filtered, footprint, absolute_threshold)

        # convert to real coordinates
        xc = x[0, 0, centers_guess_inds[:, 2]]
        yc = y[0, centers_guess_inds[:, 1], 0]
        zc = z[centers_guess_inds[:, 0], 0, 0]
        centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

        # get rois and coordinates for each center
        rois, xrois, yrois, zrois = zip(*[localize.get_roi(c, x, y, z, roi_size_pix, imgs.shape) for c in centers_guess])
        rois = np.asarray(rois)

        # fit
        roi_sizes = np.array([(r[1] - r[0]) * (r[3] - r[2]) * (r[5] - r[4]) for r in rois])
        nmax = roi_sizes.max()
        centers_temp = centers_guess

        print("using parallelization on GPU")
        imgs_roi = [np.expand_dims(localize.cut_roi(r, imgs).ravel(), axis=0) for r in rois]
        # pad to make sure all rois same size
        imgs_roi = [np.pad(ir, ((0, 0), (0, nmax - ir.size)), mode="constant") for ir in imgs_roi]

        data = np.concatenate(imgs_roi, axis=0)
        data = data.astype(np.float32)
        nfits, n_pts_per_fit = data.shape

        coords = [np.broadcast_arrays(x, y, z) for x, y, z in zip(xrois, yrois, zrois)]
        coords = [(np.pad(c[0].ravel(), (0, nmax - c[0].size)),
                   np.pad(c[1].ravel(), (0, nmax - c[1].size)),
                   np.pad(c[2].ravel(), (0, nmax - c[2].size))) for c in coords]
        user_info = np.concatenate([np.concatenate((c[0], c[1], c[2])) for c in coords])
        # user_info = np.concatenate([np.concatenate((c[0].ravel(), c[1].ravel(), c[2].ravel())) for c in coords])
        user_info = user_info.astype(np.float32)
        user_info = np.concatenate((user_info, roi_sizes.astype(np.float32)))

        init_params = np.concatenate((100 * np.ones((nfits, 1)), centers_temp[:, 2][:, None],
                                      centers_temp[:, 1][:, None], centers_temp[:, 0][:, None],
                                      0.14 * np.ones((nfits, 1)), 0.4 * np.ones((nfits, 1)), 0 * np.ones((nfits, 1))),
                                     axis=1)
        init_params = init_params.astype(np.float32)

        if data.ndim != 2:
            raise ValueError
        if init_params.ndim != 2 or init_params.shape != (nfits, 7):
            raise ValueError
        if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
            raise ValueError


        fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data, None, gf.ModelID.GAUSS_3D_ARB, init_params,
                                                                 max_number_iterations=100,
                                                                 estimator_id=gf.EstimatorID.LSE,
                                                                 user_info=user_info)

        tstart = time.perf_counter()
        centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]),
                                     axis=1)

        # only keep points if fit parameters were reasonable
        sigmas_min_keep = sigmas_min
        sigmas_max_keep = sigmas_max
        amp_min = 0.1 * absolute_threshold
        xmin = x.min() - sigmas_max[2]
        xmax = x.max() + sigmas_max[2]
        ymin = y.min() - sigmas_max[1]
        ymax = y.max() + sigmas_max[1]
        zmin = z.min() - sigmas_max[0]
        zmax = z.max() + sigmas_max[0]

        to_keep = np.logical_and.reduce((fit_params[:, 0] >= amp_min,
                                         fit_params[:, 1] >= xmin,
                                         fit_params[:, 1] <= xmax,
                                         fit_params[:, 2] >= ymin,
                                         fit_params[:, 2] <= ymax,
                                         fit_params[:, 3] >= zmin,
                                         fit_params[:, 3] <= zmax,
                                         fit_params[:, 4] <= sigmas_max_keep[2],
                                         fit_params[:, 4] >= sigmas_min_keep[2],
                                         fit_params[:, 4] <= sigmas_max_keep[1],
                                         fit_params[:, 4] >= sigmas_min_keep[1],
                                         fit_params[:, 5] <= sigmas_max_keep[0],
                                         fit_params[:, 5] >= sigmas_min_keep[0]))

        tend = time.perf_counter()
        print("identified %d valid localizations with:\n"
              "amp >= %0.5g\n"
              "%0.5g <= cx <= %.5g, %0.5g <= cy <= %.5g, %0.5g <= cz <= %.5g\n"
              "%0.5g <= sx <= %0.5g, %0.5g <= sy <= %0.5g and %0.5g <= sz <= %0.5g in %0.3f" %
              (np.sum(to_keep), 0.5 * absolute_threshold, xmin, xmax, ymin, ymax, zmin, zmax,
               sigmas_min_keep[2], sigmas_max_keep[2], sigmas_min_keep[1], sigmas_max_keep[1],
               sigmas_min_keep[0], sigmas_max_keep[0], tend - tstart))

        tstart = time.perf_counter()
        # only keep unique center if close enough
        centers_unique, unique_inds = localize.combine_nearby_peaks(centers_fit[to_keep], min_dists[1], min_dists[0],
                                                           mode="keep-one")
        fit_params_unique = fit_params[to_keep][unique_inds]
        rois_unique = rois[to_keep][unique_inds]
        tend = time.perf_counter()
        print("identified %d unique points, dxy > %0.5g, and dz > %0.5g in %0.3f" %
              (len(centers_unique), min_dists[1], min_dists[0], tend - tstart))

        # store results
        centers_vol.append(centers_unique)
        fit_params_vol.append(fit_params_unique)
        rois_vol.append(rois_unique)
        centers_guess_vol.append(centers_guess)

        tend_chunk = time.perf_counter()
        print("Processed chunk = %0.2fs" % (tend_chunk - tstart_chunk))

        # update chunk counters
        if ix_end < nxp:
            chunk_counter_x += 1
            ichunk += 1
        elif iy_end < nyp:
            chunk_counter_x = 0
            chunk_counter_y += 1
            ichunk += 1
        else:
            more_chunks = False

    # assemble results
    centers_vol = np.concatenate(centers_vol, axis=0)
    fit_params_vol = np.concatenate(fit_params_vol, axis=0)
    rois_vol = np.concatenate(rois_vol, axis=0)
    centers_guess_vol = np.concatenate(centers_guess_vol, axis=0)

    if ichunk > 0:
        # since left some overlap at the edges, have to again combine results
        centers_unique, unique_inds = localize.combine_nearby_peaks(centers_vol, min_dists[1], min_dists[0],
                                                                    mode="keep-one")
        fit_params_unique = fit_params_vol[unique_inds]
        rois_unique = rois_vol[unique_inds]
    else:
        centers_unique = centers_vol
        fit_params_unique = fit_params_vol
        rois_unique = rois_vol

    volume_process_times[vv] = time.perf_counter() - tstart_vol

    # save results
    full_results = {"centers": centers_unique, "centers_guess": centers_guess_vol,
                    "fit_params": fit_params_unique, "rois": rois_unique,
                    "volume_um3": volume_um3, "frame_time_ms": frame_time_ms, "elapsed_t": volume_process_times[vv]}
    fname = os.path.join(save_dir, "localization_results_vol_%d.pkl" % vv)
    with open(fname, "wb") as f:
        pickle.dump(full_results, f)

    # ###############################
    # print timing information
    # ###############################
    elapsed_t = volume_process_times[vv]
    hrs = (elapsed_t) // (60 * 60)
    mins = (elapsed_t - hrs * 60 * 60) // 60
    secs = (elapsed_t - hrs * 60 * 60 - mins * 60)
    print("Found %d centers in: %dhrs %dmins and %0.2fs" % (len(centers_unique), hrs, mins, secs))

    elapsed_t_total = tend - tbegin
    days = elapsed_t_total // (24 * 60 * 60)
    hrs = (elapsed_t_total - days * 24 * 60 * 60) // (60 * 60)
    mins = (elapsed_t_total - days * 24 * 60 * 60 - hrs * 60 * 60) // 60
    secs = (elapsed_t_total - days * 24 * 60 * 60 - hrs * 60 * 60 - mins * 60)
    print("Total elapsed time: %ddays %dhrs %dmins and %0.2fs" % (days, hrs, mins, secs))

    time_remaining = np.mean(volume_process_times[:vv + 1]) * (nvols - vv - 1)
    days = time_remaining // (24 * 60 * 60)
    hrs = (time_remaining - days * 24 * 60 * 60) // (60 * 60)
    mins = (time_remaining - days * 24 * 60 * 60 - hrs * 60 * 60) // 60
    secs = (time_remaining - days * 24 * 60 * 60 - hrs * 60 * 60 - mins * 60)
    print("Estimated time remaining: %ddays %dhrs %dmins and %0.2fs" % (days, hrs, mins, secs))

# get histgram info
    nh = 1000
    ah, bin_edges_ah = np.histogram(fit_params_vol[:, 0], nh)
    bin_centers_ah = 0.5 * (bin_edges_ah[:-1] + bin_edges_ah[1:])

    bh, bin_edges_bh = np.histogram(fit_params_vol[:, 6], nh)
    bin_centers_bh = 0.5 * (bin_edges_bh[:-1] + bin_edges_bh[1:])

    sxyh, bin_edges_sxyh = np.histogram(fit_params_vol[:, 4], nh)
    bin_centers_sxyh = 0.5 * (bin_edges_sxyh[:-1] + bin_edges_sxyh[1:])

    szh, bin_edges_szh = np.histogram(fit_params_vol[:, 5], nh)
    bin_centers_szh = 0.5 * (bin_edges_szh[:-1] + bin_edges_szh[1:])

if plot_results:

    # plot histograms
    figh = plt.figure(figsize=(12, 8))
    plt.suptitle("Histograms of fit quantities for %d localizations" % len(fit_params_vol))
    grid = plt.GridSpec(2, 2, hspace=0.5, wspace=0.5)

    ax = plt.subplot(grid[0, 0])
    ax.plot(bin_centers_ah, ah)
    ax.set_title("Amplitudes")
    ax.set_xlabel("Amplitude (ADU)")
    ax.set_ylabel("Counts")

    ax = plt.subplot(grid[0, 1])
    ax.plot(bin_centers_bh, bh)
    ax.set_title("Backgrounds")
    ax.set_xlabel("Background (ADU)")
    ax.set_ylabel("Counts")

    ax = plt.subplot(grid[1, 0])
    ax.plot(bin_centers_sxyh, sxyh)
    ax.set_title("Sigma xy")
    ax.set_xlabel("Sigma xy (um)")
    ax.set_ylabel("Counts")

    ax = plt.subplot(grid[1, 1])
    ax.plot(bin_centers_szh, szh)
    ax.set_title("Sigma z")
    ax.set_xlabel("Sigma z (um)")
    ax.set_ylabel("Counts")


    # show results with napari
    imgs = np.squeeze(tifffile.imread(data_fname)[:, channels_to_use])
    # plot with napari
    with napari.gui_qt():
        # specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
        viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=save_dir)

        centers_pix = centers_vol / np.expand_dims(np.array([dz, dc, dc]), axis=0)
        centers_guess_pix = centers_guess_vol / np.expand_dims(np.array([dz, dc, dc]), axis=0)
        if plot_centers:
            viewer.add_points(centers_pix, size=2, face_color="red", opacity=0.75, n_dimensional=True)
        if plot_centers_guess:
            viewer.add_points(centers_guess_pix, size=2, face_color="green", opacity=0.5, n_dimensional=True)