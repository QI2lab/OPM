import glob
import os
import datetime
import time
import numpy as np
import pickle
import localize
import tifffile
import napari

debug = False
# paths to image files
top_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210503a")
root_dirs = glob.glob(os.path.join(top_dir, "r*_atto565_r*_alexa647_1"))

# ###############################
# load/set parameters for all datasets
# ###############################
chunk_size = 301
chunk_overlap = 5
nvols = 1
nchannels = 3
channels_to_use = [False, True, True]
excitation_wavelengths = [0.488, 0.565, 0.647]
emission_wavelengths = [0.515, 0.600, 0.680]
thresholds = [300, 40, 40]
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
save_dir = os.path.join(top_dir, "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ###############################
# do runs
# ###############################
tbegin = time.perf_counter()
for root_dir in root_dirs:
    data_fname = glob.glob(os.path.join(root_dir, "*.ome.tif"))[0]

    # get tif file dimensions
    tif = tifffile.TiffFile(data_fname)
    nimgs_per_vol, nch, nyp, nxp = tif.series[0].shape
    # can also read dz from tif tags, but lots of trouble
    # tag = tif.pages[0].tags["IJMetadata"].value["Info"]
    tif.close()

    volume_um3 = (dz * nimgs_per_vol) * (nyp * dc) * (nxp * dc)

    if nch != nchannels:
        raise ValueError("nchannels=%d, but found %d channels in image %s" % (nchannels, nch, data_fname))

    for jj in range(nchannels):
        print("################################\nstarting channel %d/%d" % (jj + 1, nchannels))
        if not channels_to_use[jj]:
            continue

        tstart_load = time.perf_counter()
        imgs = np.squeeze(tifffile.imread(data_fname)[:, jj])
        print("Took %0.2fs to load images" % (time.perf_counter() - tstart_load))

        sigma_xy = 0.22 * emission_wavelengths[jj] / na
        sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelengths[jj] / na ** 2

        # ###############################
        # identify candidate points in opm data
        # ###############################
        # difference of gaussian filer
        filter_sigma_small = (0.5 * sigma_z, 0.5 * sigma_xy, 0.5 * sigma_xy)
        filter_sigma_large = (5 * sigma_z, 5 * sigma_xy, 5 * sigma_xy)
        # fit roi size
        # roi_size = (3 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
        roi_size = (5 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
        roi_size_pix = localize.get_roi_size(roi_size, dc, dz)
        # assume points closer together than this come from a single bead
        min_dists = (3 * sigma_z, 2 * sigma_xy, 2 * sigma_xy)
        # exclude points with sigmas outside these ranges
        # sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
        sigmas_min = (0.25 * sigma_z, dc, dc)
        sigmas_max = (np.inf * sigma_z, 3 * sigma_xy, 3 * sigma_xy)

        # don't consider any points outside of this polygon
        # cx, cy
        # allowed_camera_region = np.array([[357, 0], [464, 181], [1265, 231], [1387, 0]])
        allowed_camera_region = None

        # ###############################
        # loop over volumes
        # ###############################

        volume_process_times = np.zeros(nvols)
        nchunks = int(np.ceil(nyp / chunk_size) * np.ceil(nxp / chunk_size))
        for vv in range(nvols):
            print("################################\nstarting volume %d/%d" % (vv + 1, nvols))
            tstart_vol = time.perf_counter()

            centers_vol = []
            fit_params_vol = []
            rois_vol = []
            # with full dim
            rois_all_vol = []
            centers_guess_vol = []
            init_params_vol = []
            fit_params_all_vol = []
            fit_results_all_vol = []
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
                footprint = localize.get_footprint(min_dists, dc, dz)
                centers_guess_inds, amps = localize.find_peak_candidates(imgs_filtered, footprint, thresholds[jj])

                # convert to distance coordinates
                xc = x[0, 0, centers_guess_inds[:, 2]]
                yc = y[0, centers_guess_inds[:, 1], 0]
                zc = z[centers_guess_inds[:, 0], 0, 0]
                centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

                if debug:
                    with napari.gui_qt():
                        # specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
                        viewer = napari.view_image(imgs_filtered, colormap="bone",
                                                   contrast_limits=[np.percentile(imgs_filtered, 0.1), np.percentile(imgs_filtered, 99.9)],
                                                   multiscale=False, title="chunk = %d" % ichunk)

                        # viewer = napari.view_image(imgs_chunk, colormap="bone",
                        #                            contrast_limits=[np.percentile(imgs_chunk, 0.1),
                        #                                             np.percentile(imgs_chunk, 99.9)],
                        #                            multiscale=False, title="chunk = %d" % ichunk)

                        viewer.add_points(centers_guess_inds, size=2, face_color="green", opacity=0.5, name="centers guess", n_dimensional=True)



                if len(centers_guess) > 0:
                    # ###############################
                    # do fitting
                    # ###############################
                    tstart_ip = time.perf_counter()

                    # get rois and coordinates for each center
                    # rois, img_rois, xrois, yrois, zrois = zip(*[localize.get_roi(c, imgs_chunk, x, y, z, roi_size_pix) for c in centers_guess])
                    rois, img_rois, xrois, yrois, zrois = zip(*[localize.get_roi(c, imgs_filtered, x, y, z, roi_size_pix) for c in centers_guess])
                    rois = np.asarray(rois)
                    nsizes = (rois[:, 1]- rois[:, 0]) * (rois[:, 3] - rois[:, 2]) * (rois[:, 5] - rois[:, 4])
                    nfits = len(rois)

                    # extract guess values
                    bgs = np.array([np.mean(r) for r in img_rois])
                    sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in zip(img_rois, xrois, centers_guess)])
                    sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in zip(img_rois, yrois, centers_guess)])
                    sxys = np.expand_dims(0.5 * sxs + sys, axis=1)
                    szs = np.expand_dims(np.array([np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in zip(img_rois, zrois, centers_guess)]), axis=1)

                    init_params = np.concatenate((np.expand_dims(amps, axis=1), centers_guess[:, 2][:, None],
                                                  centers_guess[:, 1][:, None], centers_guess[:, 0][:, None],
                                                  sxys, szs, np.expand_dims(bgs, axis=1)),
                                                  axis=1)

                    tend_ip = time.perf_counter()
                    print("Estimated initial parameters in %0.2fs" % (tend_ip - tstart_ip))

                    print("Fitting %d rois on GPU" % (len(rois)))
                    # use MLE if background subtracted images
                    fit_params, fit_states, chi_sqrs, niters, fit_t = localize.fit_rois(img_rois, (zrois, yrois, xrois), init_params,
                                                                                        estimator="LSE")

                    # ###############################
                    # store unfiltered results
                    # ###############################
                    fit_results_all_vol.append(np.concatenate((np.expand_dims(fit_states, axis=1),
                                                               np.expand_dims(chi_sqrs, axis=1),
                                                               np.expand_dims(niters, axis=1)), axis=1))

                    fit_params_all_vol.append(np.array(fit_params, copy=True))

                    rois_temp = np.array(rois, copy=True)
                    rois_temp[:, 2:4] += iy_start
                    rois_temp[:, 4:6] += ix_start
                    rois_all_vol.append(rois_temp)

                    # ###############################
                    # filter some peaks
                    # ###############################
                    tstart = time.perf_counter()
                    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]),
                                                 axis=1)

                    # only keep points if fit parameters were reasonable
                    sigmas_min_keep = sigmas_min
                    sigmas_max_keep = sigmas_max
                    amp_min = 0.5 * thresholds[jj]
                    xmin = x.min() + sigma_xy
                    xmax = x.max() - sigma_xy
                    ymin = y.min() + sigma_xy
                    ymax = y.max() - sigma_xy
                    zmin = z.min()
                    zmax = z.max()

                    in_bounds = np.logical_and.reduce((fit_params[:, 1] >= xmin,
                                                       fit_params[:, 1] <= xmax,
                                                       fit_params[:, 2] >= ymin,
                                                       fit_params[:, 2] <= ymax,
                                                       fit_params[:, 3] >= zmin,
                                                       fit_params[:, 3] <= zmax))

                    center_close_to_guess_xy = np.sqrt( (centers_guess[:, 2] - fit_params[:, 1])**2 + (centers_guess[:, 1] - fit_params[:, 2])**2) <= sigma_xy
                    center_close_to_guess_z = np.abs(centers_guess[:, 0]- fit_params[:, 3]) <= sigma_z

                    condition_names = ["in_bounds", "center_close_to_guess_xy", "center_close_to_guess_z",
                                       "xy_size_small_enough", "xy_size_big_enough", "z_size_small_enough",
                                       "z_size_big_enough", "amp_ok"]
                    conditions = np.concatenate((np.expand_dims(in_bounds, axis=1),
                                                 np.expand_dims(center_close_to_guess_xy, axis=1),
                                                 np.expand_dims(center_close_to_guess_z, axis=1),
                                                 np.expand_dims(fit_params[:, 4] <= sigmas_max_keep[1], axis=1),
                                                 np.expand_dims(fit_params[:, 4] >= sigmas_min_keep[1], axis=1),
                                                 np.expand_dims(fit_params[:, 5] <= sigmas_max_keep[0], axis=1),
                                                 np.expand_dims(fit_params[:, 5] >= sigmas_min_keep[0], axis=1),
                                                 np.expand_dims(fit_params[:, 0] >= amp_min, axis=1)), axis=1)

                    to_keep = np.logical_and.reduce(conditions, axis=1)

                    tend = time.perf_counter()
                    print("identified %d/%d localizations in %0.3f with:\n"
                          "amp >= %0.5g\n"
                          "xy distance from guess <= %0.5g\n"
                          "z distance from guess <= %0.5g\n"
                          "%0.5g <= cx <= %.5g\n"
                          "%0.5g <= cy <= %.5g\n"
                          "%0.5g <= cz <= %.5g\n"
                          "%0.5g <= sxy <= %0.5g\n"
                          "%0.5g <= sz <= %0.5g " %
                          (np.sum(to_keep), to_keep.size, tend - tstart,
                           amp_min, sigma_xy, sigma_z,
                           xmin, xmax, ymin, ymax, zmin, zmax,
                           sigmas_min_keep[1], sigmas_max_keep[1],
                           sigmas_min_keep[0], sigmas_max_keep[0]))

                    tstart = time.perf_counter()
                    # only keep unique center if close enough

                    if np.sum(to_keep) > 0:
                        centers_unique, unique_inds = localize.combine_nearby_peaks(centers_fit[to_keep], min_dists[1],
                                                                                    min_dists[0], mode="keep-one")
                        fit_params_unique = fit_params[to_keep][unique_inds]
                        rois_unique = rois[to_keep][unique_inds]
                        tend = time.perf_counter()
                        print("identified %d/%d unique points as unique: separations dxy > %0.5g, and dz > %0.5g in %0.3f" %
                              (len(centers_unique), np.sum(to_keep), min_dists[1], min_dists[0], tend - tstart))

                        # correct for full image coordinates
                        rois_unique[:, 2:4] += iy_start
                        rois_unique[:, 4:6] += ix_start
                    else:
                        centers_unique = np.zeros((0, 3))
                        fit_params_unique = np.zeros((0, 7))
                        rois_unique = np.zeros((0, 6))

                    # store results
                    centers_vol.append(centers_unique)
                    fit_params_vol.append(fit_params_unique)
                    rois_vol.append(rois_unique)
                    #
                    centers_guess_vol.append(centers_guess)
                    conditions_vol.append(conditions)
                    init_params_vol.append(init_params)

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
            fit_params_all_vol = np.concatenate(fit_params_all_vol, axis=0)
            rois_vol = np.concatenate(rois_vol, axis=0)
            rois_all_vol = np.concatenate(rois_all_vol, axis=0)
            centers_guess_vol = np.concatenate(centers_guess_vol, axis=0)
            conditions_vol = np.concatenate(conditions_vol, axis=0)
            init_params_vol = np.concatenate(init_params_vol, axis=0)
            fit_results_all_vol = np.concatenate(fit_results_all_vol, axis=0)

            volume_process_times[vv] = time.perf_counter() - tstart_vol

            # save results
            localization_settings = {"filter_sigma_small_um": filter_sigma_small, "filter_sigma_large_um": filter_sigma_large,
                                     "roi_size_um": roi_size, "roi_size_pix": roi_size_pix, "min_dists_um": min_dists,
                                     "sigmas_min_um": sigmas_min, "sigmas_max_um": sigmas_max, "threshold": thresholds[jj],
                                     "chunk_size": chunk_size, "chunk_overlap": chunk_overlap
                                     }

            full_results = {"centers": centers_vol,
                            "centers_guess": centers_guess_vol,
                            "conditions": conditions_vol, "conditions_names": condition_names,
                            "fit_params": fit_params_vol, "rois": rois_vol,
                            "fit_params_all": fit_params_all_vol, "init_params": init_params_vol,
                            "rois_all": rois_all_vol, "fit_results_all": fit_results_all_vol,
                            "volume_um3": volume_um3, "frame_time_ms": frame_time_ms,
                            "elapsed_t": volume_process_times[vv],
                            "excitation_wavelength_um": excitation_wavelengths[jj],
                            "localization_settings": localization_settings
                            }

            fname = os.path.join(save_dir, "%s_ch=%d_vol=%d.pkl" % (os.path.split(root_dir)[1], jj, vv))
            with open(fname, "wb") as f:
                pickle.dump(full_results, f)

            # ###############################
            # print timing information
            # ###############################
            elapsed_t = volume_process_times[vv]
            hrs = (elapsed_t) // (60 * 60)
            mins = (elapsed_t - hrs * 60 * 60) // 60
            secs = (elapsed_t - hrs * 60 * 60 - mins * 60)
            print("################################\nFound %d centers in: %dhrs %dmins and %0.2fs" % (len(centers_vol), hrs, mins, secs))

            elapsed_t_total = time.perf_counter() - tbegin
            days = elapsed_t_total // (24 * 60 * 60)
            hrs = (elapsed_t_total - days * 24 * 60 * 60) // (60 * 60)
            mins = (elapsed_t_total - days * 24 * 60 * 60 - hrs * 60 * 60) // 60
            secs = (elapsed_t_total - days * 24 * 60 * 60 - hrs * 60 * 60 - mins * 60)
            print("Total elapsed time: %ddays %dhrs %dmins and %0.2fs" % (days, hrs, mins, secs))
