"""
Localize RNA FISH data

/mnt/opm2/20210610/output/tiffs/fused_tiff/

The file naming is: TL{r}_Ch{c}_Tile{t}.tiff

r = round. There are RNA in rounds r =[0,1,2,3,4,5,6,7]
c = channel. There are RNA in c = [1,2]
t = tile. There are RNA in t = [0,1,2,3,4,5,6,7,8]

c=1 is atto565 (emission ~ 590 nm)
c=2 is alexa647 (emission ~ 670 nm)

In-plane pixel = .065 um
Axial step = .250 um
"""

import matplotlib.pyplot as plt
import glob
import os
import datetime
import time
import numpy as np
import pickle
import localize_skewed
import localize
import tifffile
import napari
import rois

debug = False
root_dir = r"\\10.206.26.21\opm2\20210610\output\fused_tiff"
# paths to image files

# ###############################
# load/set parameters for all datasets
# ###############################
chunk_size = 256
chunk_overlap = 5
nvols = 1
channel_to_use = [False, True, True]
excitation_wavelengths = [0.488, 0.561, 0.638]
emission_wavelengths = [0.515, 0.590, 0.670]
thresholds = [100, 5, 3.5]
dc = 0.065
dz = 0.25
frame_time_ms = None
na = 1.35
ni = 1.4

# ###############################
# build save dir
# ###############################
now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
save_dir = os.path.join(root_dir, "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ###############################
# do localizations
# ###############################
tstart_all = time.perf_counter()
for round in [0, 1, 2, 3, 4, 5, 6, 7]:

    for ch in range(len(excitation_wavelengths)):
        # if ch == 1:
        #     continue

        if not channel_to_use[ch]:
            continue

        sigma_xy = 0.22 * emission_wavelengths[ch] / na
        sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelengths[ch] / na ** 2

        for tl in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            tstart_folder = time.perf_counter()

            # ###############################
            # load data
            # ###############################
            data_fname = os.path.join(root_dir, "img_TL%d_Ch%d_Tile%d.tif" % (round, ch, tl))

            tstart_load = time.perf_counter()
            imgs = tifffile.imread(data_fname)
            print("Took %0.2fs to load images" % (time.perf_counter() - tstart_load))

            nimgs_per_vol, nyp, nxp = imgs.shape
            volume_um3 = (dz * nimgs_per_vol) * (nyp * dc) * (nxp * dc)

            # ###############################
            # identify candidate points in opm data
            # ###############################
            # difference of gaussian filer
            filter_sigma_small = (0.5 * sigma_z, 0.5 * sigma_xy, 0.5 * sigma_xy)
            filter_sigma_large = (5 * sigma_z, 5 * sigma_xy, 5 * sigma_xy)
            # fit roi size
            roi_size = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
            roi_size_pix = rois.get_roi_size(roi_size, dc, dz)
            # assume points closer together than this come from a single bead
            min_dists = (3 * sigma_z, 2 * sigma_xy)
            # exclude points with sigmas outside these ranges
            sigmas_min = (0.5 * sigma_z, dc)
            sigmas_max = (2 * sigma_z, 2 * sigma_xy)

            # don't consider any points outside of this polygon
            # cx, cy
            allowed_camera_region = None

            # ###############################
            # loop over volumes
            # ###############################

            volume_process_times = np.zeros(nvols)
            nchunks = int(np.ceil(nyp / (chunk_size - chunk_overlap)) * np.ceil(nxp / (chunk_size - chunk_overlap)))
            for vv in range(nvols):
                print("################################\nstarting volume %d/%d" % (vv + 1, nvols))
                tstart_vol = time.perf_counter()

                fit_params_vol = []
                init_params_vol = []
                rois_vol = []
                fit_results_vol = []
                to_keep_vol = []
                conditions_vol = []

                ichunk = 0
                chunk_counter_x = 0
                chunk_counter_y = 0
                more_chunks = True
                while more_chunks:
                    tstart_chunk = time.perf_counter()
                    print("################################\nprocessing chunk %d/%d" % (ichunk + 1, nchunks))

                    ix_start = int(np.max([chunk_counter_x * chunk_size - chunk_overlap, 0]))
                    ix_end = int(np.min([ix_start + chunk_size, nxp]))

                    iy_start = int(np.max([chunk_counter_y * chunk_size - chunk_overlap, 0]))
                    iy_end = int(np.min([iy_start + chunk_size, nyp]))

                    imgs_chunk = imgs[:, iy_start:iy_end, ix_start:ix_end]

                    # get image coordinates
                    x, y, z, = localize.get_coords(imgs_chunk.shape, dc, dz)
                    x += dc * ix_start
                    y += dc * iy_start

                    # filter
                    ks = localize.get_filter_kernel(filter_sigma_small, dc, dz, sigma_cutoff=2)
                    kl = localize.get_filter_kernel(filter_sigma_large, dc, dz, sigma_cutoff=2)
                    imgs_filtered = localize.filter_convolve(imgs_chunk, ks) - localize.filter_convolve(imgs_chunk, kl)

                    # find candidate peaks
                    footprint = localize.get_max_filter_footprint((min_dists[0], min_dists[1], min_dists[1]), dc, dz)
                    centers_guess_inds, centers_guess_amps = localize.find_peak_candidates(imgs_filtered, footprint, thresholds[ch])

                    # convert to distance coordinates
                    xc = x[0, 0, centers_guess_inds[:, 2]]
                    yc = y[0, centers_guess_inds[:, 1], 0]
                    zc = z[centers_guess_inds[:, 0], 0, 0]
                    centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

                    if debug:

                        # maximum projection
                        figh = plt.figure()
                        plt.title("%s\nThreshold = %.2f" % (data_fname, thresholds[ch]))
                        imgs_max_proj = np.max(imgs_filtered, axis=0)
                        vmin = np.percentile(imgs_max_proj, 0.1)
                        vmax = np.percentile(imgs_filtered, 99.95)
                        plt.imshow(imgs_max_proj, vmin=vmin, vmax=vmax, cmap="bone")
                        plt.plot(centers_guess_inds[:, 2], centers_guess_inds[:, 1], 'rx')
                        plt.colorbar()

                        # 3D volume
                        with napari.gui_qt():
                            vmin = np.percentile(imgs_filtered, 0.1)
                            vmax = np.percentile(imgs_filtered, 99.95)

                            # specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
                            viewer = napari.view_image(imgs_filtered, colormap="bone",
                                                       contrast_limits=[vmin, vmax],
                                                       multiscale=False, title="chunk = %d" % ichunk)

                            viewer.add_points(centers_guess_inds, size=2, face_color="red", opacity=0.5, name="centers guess", n_dimensional=True)

                    if len(centers_guess) > 0:
                        # ###############################
                        # do fitting
                        # ###############################
                        tstart_ip = time.perf_counter()

                        # get rois and coordinates for each center
                        # todo: better to fit filtered images or not?
                        rois, img_rois, xrois, yrois, zrois = zip(*[localize.get_roi(c, imgs_chunk, x, y, z, roi_size_pix) for c in centers_guess])
                        # rois, img_rois, xrois, yrois, zrois = zip(*[localize.get_roi(c, imgs_filtered, x, y, z, roi_size_pix) for c in centers_guess])
                        rois = np.asarray(rois)
                        nsizes = (rois[:, 1] - rois[:, 0]) * (rois[:, 3] - rois[:, 2]) * (rois[:, 5] - rois[:, 4])
                        nfits = len(rois)

                        # extract guess values
                        bgs = np.array([np.mean(r) for r in img_rois])

                        # todo: what is best way to guess amps? Think that using the maximum value can bias the fits
                        # option 1: peak value from max filter
                        # amps = np.expand_dims(centers_guess_amps, axis=1)

                        # option 2: peak value minus background
                        # amps = centers_guess_amps - bgs

                        # option 3: use 90th percentile minus bg. But set to zero if smaller than sd
                        sds = np.array([np.std(r) for r in img_rois])

                        amps = np.array([np.percentile(r, 90) for r in img_rois]) - bgs
                        amps[amps < sds] = 0

                        sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in zip(img_rois, xrois, centers_guess)])
                        sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in zip(img_rois, yrois, centers_guess)])
                        sxys = np.expand_dims(0.5 * sxs + sys, axis=1)
                        szs = np.expand_dims(np.array([np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in zip(img_rois, zrois, centers_guess)]), axis=1)


                        init_params = np.concatenate((np.expand_dims(amps, axis=1),
                                                      centers_guess[:, 2][:, None],
                                                      centers_guess[:, 1][:, None],
                                                      centers_guess[:, 0][:, None],
                                                      sxys, szs,
                                                      np.expand_dims(bgs, axis=1)),
                                                      axis=1)

                        tend_ip = time.perf_counter()
                        print("Estimated initial parameters in %0.2fs" % (tend_ip - tstart_ip))

                        print("Fitting %d rois on GPU" % (len(rois)))
                        # use MLE if background subtracted images
                        fit_params, fit_states, chi_sqrs, niters, fit_t = localize.fit_gauss_rois(img_rois, (zrois, yrois, xrois), init_params,
                                                                                                   estimator="LSE", sf=1, dc=dc, angles=(0, 0, 0))

                        # ###############################
                        # filter some peaks
                        # ###############################
                        tstart = time.perf_counter()

                        to_keep, conditions, condition_names, filter_settings = localize_skewed.filter_localizations(fit_params, init_params,
                                                                                                                     (z, y, x), (sigma_z, sigma_xy),
                                                                                                                     min_dists, (sigmas_min, sigmas_max),
                                                                                                                     0.5 * thresholds[ch], mode="straight")

                        tend = time.perf_counter()
                        print("identified %d/%d localizations in %0.3f" % (np.sum(to_keep), to_keep.size,  time.perf_counter() - tstart))

                        # ###############################
                        # store results
                        # ###############################
                        fit_results_vol.append(np.concatenate((np.expand_dims(fit_states, axis=1),
                                                                   np.expand_dims(chi_sqrs, axis=1),
                                                                   np.expand_dims(niters, axis=1)), axis=1))

                        fit_params_vol.append(fit_params)
                        init_params_vol.append(init_params)
                        conditions_vol.append(conditions)
                        to_keep_vol.append(to_keep)

                        rois[:, 2:4] += iy_start
                        rois[:, 4:6] += ix_start
                        rois_vol.append(rois)

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
                fit_params_vol = np.concatenate(fit_params_vol, axis=0)
                init_params_vol = np.concatenate(init_params_vol, axis=0)
                rois_vol = np.concatenate(rois_vol, axis=0)
                fit_results_vol = np.concatenate(fit_results_vol, axis=0)
                to_keep_vol = np.concatenate(to_keep_vol, axis=0)
                conditions_vol = np.concatenate(conditions_vol, axis=0)

                volume_process_times[vv] = time.perf_counter() - tstart_vol

                # save results
                localization_settings = {"filter_sigma_small_um": filter_sigma_small, "filter_sigma_large_um": filter_sigma_large,
                                         "roi_size_um": roi_size, "roi_size_pix": roi_size_pix, "min_dists_um": min_dists,
                                         "sigmas_min_um": sigmas_min, "sigmas_max_um": sigmas_max, "threshold": thresholds[ch],
                                         "chunk_size": chunk_size, "chunk_overlap": chunk_overlap
                                         }

                physical_data = {"dc": dc, "dz": dz, "frame_time_ms": frame_time_ms, "na": na, "ni": ni,
                                 "excitation_wavelength": excitation_wavelengths[ch],
                                 "emission_wavelength": emission_wavelengths[ch],
                                 "sigma_xy_ideal": sigma_xy, "sigma_z_ideal": sigma_z}

                full_results = {"fit_params": fit_params_vol, "init_params": init_params_vol,
                                "rois": rois_vol, "fit_results": fit_results_vol,
                                "to_keep": to_keep_vol, "conditions": conditions_vol, "conditions_names": condition_names,
                                "volume_um3": volume_um3, "elapsed_t": volume_process_times[vv],
                                "localization_settings": localization_settings, "filter_settings": filter_settings,
                                "physical_data": physical_data
                                }

                _, fn = os.path.split(data_fname)
                fn_no_ext, _ = os.path.splitext(fn)
                fname = os.path.join(save_dir, "%s_round=%d_ch=%d_tile=%d_vol=%d.pkl" % (fn_no_ext, round, ch, tl, vv))
                with open(fname, "wb") as f:
                    pickle.dump(full_results, f)

            # ###############################
            # print timing information
            # ###############################
            print("Found %d centers in              : %s" % (np.sum(to_keep_vol), str(datetime.timedelta(seconds=volume_process_times[vv]))))
            print("Total elapsed time (folder)      : %s" % datetime.timedelta(seconds=tend - tstart_folder))

            time_remaining_vol = datetime.timedelta(seconds=np.mean(volume_process_times[:vv + 1]) * (nvols - vv - 1))
            print("Estimated time remaining (folder): %s" % str(time_remaining_vol))

            print("Total elapsed time (all)         : %s" % datetime.timedelta(seconds=tend - tstart_all))