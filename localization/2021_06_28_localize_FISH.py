"""
Localize OPM RNA-FISH data
"""
import matplotlib.pyplot as plt
import glob
import os
import datetime
import time
import numpy as np
import pickle
import pycromanager
import tifffile
import napari

import image_post_processing as pp
import data_io
import localize

debug = False
root_dir = r"\\10.206.26.21\opm2\20210628"
dir_format = "bDNA_stiff_gel_human_lung_r%04d_y%04d_z%04d_ch%04d_1"
scan_data_path = os.path.join(root_dir, "scan_metadata.csv")

# ###############################
# load/set parameters for all datasets
# ###############################
chunk_size_planes = 200
chunk_size_x = 325
chunk_overlap = 5
channel_to_use = [False, True, True]
excitation_wavelengths = [0.488, 0.561, 0.638]
emission_wavelengths = [0.515, 0.590, 0.670]
thresholds = [None, 100, 100]
fit_thresholds = [None, 50, 50]
frame_time_ms = None
na = 1.35
ni = 1.4

# ###############################
# load/set scan parameters
# ###############################
scan_data = data_io.read_metadata(scan_data_path)

nt = scan_data["num_t"]
nimgs_per_vol = scan_data["scan_axis_positions"]
nyp = scan_data["y_pixels"]
nxp = scan_data["x_pixels"]
dc = scan_data["pixel_size"] / 1000
dstage = scan_data["scan_step"] / 1000
theta = scan_data["theta"] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)])  # normal of camera pixel

# trapezoid volume
volume_um3 = (dstage * nimgs_per_vol) * (dc * np.sin(theta) * nyp) * (dc * nxp)

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
for round in [0]:
    for ch in range(3):
        if not channel_to_use[ch]:
            continue

        sigma_xy = 0.22 * emission_wavelengths[ch] / na
        sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelengths[ch] / na ** 2

        for tl in [7]:
            for iz in [1]:
                tstart_folder = time.perf_counter()

                # ###############################
                # load dataset
                # ###############################
                data_fdir = os.path.join(root_dir, dir_format % (round, tl, iz, ch + 1))
                dset = pycromanager.Dataset(data_fdir)

                md = dset.read_metadata(z=0, channel=0)
                frame_time_ms = float(md["OrcaFusionBT-Exposure"])

                # ###############################
                # identify candidate points in opm data
                # ###############################
                # difference of gaussian filer
                filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
                filter_sigma_large = (5 * sigma_z, 5 * sigma_xy, 5 * sigma_xy)
                # fit roi size
                roi_size = (3 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
                # assume points closer together than this come from a single bead
                min_spot_sep = (5 * sigma_z, 4 * sigma_xy)
                # exclude points with sigmas outside these ranges
                sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy)
                sigmas_max = (2 * sigma_z, 3 * sigma_xy)

                # ###############################
                # loop over volumes
                # ###############################

                volume_process_times = np.zeros(nt)
                nchunks = int(np.ceil(nimgs_per_vol / (chunk_size_planes - chunk_overlap)) * np.ceil(nxp / (chunk_size_x - chunk_overlap)))
                for vv in range(nt):
                    print("################################\nstarting volume %d/%d" % (vv + 1, nt))
                    tstart_vol = time.perf_counter()

                    fit_params_vol = []
                    init_params_vol = []
                    rois_vol = []
                    fit_results_vol = []
                    to_keep_vol = []
                    conditions_vol = []

                    # loop over chunks of images
                    more_chunks = True
                    ichunk = 0
                    chunk_counter_p = 0
                    chunk_counter_x = 0
                    while more_chunks:
                        print("Chunk %d/%d" % (ichunk + 1, nchunks))
                        # load images
                        tstart_load = time.perf_counter()
                        ix_start = int(np.max([chunk_counter_x * chunk_size_x - chunk_overlap, 0]))
                        ix_end = int(np.min([ix_start + chunk_size_x, nxp]))

                        ip_start = int(np.max([chunk_counter_p * chunk_size_planes - chunk_overlap, 0]))
                        ip_end = int(np.min([ip_start + chunk_size_planes, nimgs_per_vol]))

                        imgs = []
                        for kk in range(ip_start, ip_end):
                            imgs.append(dset.read_image(z=kk, channel=0)[:, ix_start:ix_end])
                        imgs = np.asarray(imgs)

                        # if no images returned, we are done
                        if imgs.shape[0] == 0:
                            break

                        imgs = np.flip(imgs, axis=0)  # to match deskew convention...

                        tend_load = time.perf_counter()
                        print("loaded images in %0.2fs" % (tend_load - tstart_load))

                        # get image coordinates
                        npos, ny, nx = imgs.shape
                        y_offset = ip_start * dstage
                        x_offset = ix_start * dc

                        x, y, z = localize.get_skewed_coords((npos, ny, nx), dc, dstage, theta)
                        x += x_offset
                        y += y_offset

                        # do localization
                        # ###################################################
                        # smooth image and remove background with difference of gaussians filter
                        # ###################################################
                        tstart = time.perf_counter()

                        ks = localize.get_filter_kernel_skewed(filter_sigma_small, dc, theta, dstage, sigma_cutoff=2)
                        kl = localize.get_filter_kernel_skewed(filter_sigma_large, dc, theta, dstage, sigma_cutoff=2)
                        imgs_hp = localize.filter_convolve(imgs, ks)
                        imgs_lp = localize.filter_convolve(imgs, kl, use_gpu=True)
                        imgs_filtered = imgs_hp - imgs_lp

                        print("Filtered images in %0.2fs" % (time.perf_counter() - tstart))

                        # ###################################################
                        # identify candidate beads
                        # ###################################################
                        tstart = time.perf_counter()

                        dz_min, dxy_min = min_spot_sep

                        footprint = localize.get_skewed_footprint((dz_min, dxy_min, dxy_min), dc, dstage, theta)
                        centers_guess_inds, amps = localize.find_peak_candidates(imgs_filtered, footprint, thresholds[ch])

                        # convert to xyz coordinates
                        xc = x[0, 0, centers_guess_inds[:, 2]]
                        yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
                        zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
                        centers_guess = np.stack((zc, yc, xc), axis=1)

                        print("Found %d points above threshold in %0.2fs" % (
                        len(centers_guess), time.perf_counter() - tstart))

                        # ###################################################
                        # check fits
                        # ###################################################
                        if debug:
                            imgs_filtered_deskewed = pp.deskew(imgs_filtered, [theta * 180/np.pi, dstage, dc])
                            # imgs_filtered_deskewed = pp.deskew(imgs, [theta * 180 / np.pi, dstage, dc])

                            figh = plt.figure()
                            extent = [-0.5 * dc + x_offset, dc * imgs_filtered_deskewed.shape[2] + 0.5 * dc + x_offset,
                                      -0.5 * dc + y_offset, dc * imgs_filtered_deskewed.shape[1] + + 0.5 * dc + y_offset]
                            max_proj = np.max(imgs_filtered_deskewed, axis=0)
                            plt.imshow(max_proj, extent=extent, origin="lower",
                                       vmin=np.percentile(max_proj, 1), vmax=np.percentile(max_proj, 99.9))
                            plt.plot(centers_guess[:, 2], centers_guess[:, 1], 'rx')

                            with napari.gui_qt():
                                viewer = napari.view_image(imgs_filtered_deskewed, colormap="bone",
                                                           contrast_limits=[np.percentile(imgs_filtered_deskewed, 1),
                                                                            np.percentile(imgs_filtered_deskewed, 99.99)],
                                                           multiscale=False)

                                centers_napari = (centers_guess - np.expand_dims(np.array([0, y_offset, x_offset]), axis=0)) / dc
                                viewer.add_points(centers_napari, size=2, face_color="red", name="fits", opacity=0.75,
                                                  n_dimensional=True)

                        if len(centers_guess) != 0:
                            # ###################################################
                            # average multiple points too close together. Necessary bc if naive threshold, may identify several points
                            # from same spot. Particularly important if spots have very different brightness levels.
                            # ###################################################
                            tstart = time.perf_counter()

                            inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
                            weights = imgs_filtered.ravel()[inds]
                            centers_guess, inds_comb = localize.combine_nearby_peaks(centers_guess, dxy_min, dz_min, weights=weights,
                                                                            mode="average")

                            amps = amps[inds_comb]
                            print("Found %d points separated by dxy > %0.5g and dz > %0.5g in %0.1fs" %
                                  (len(centers_guess), dxy_min, dz_min, time.perf_counter() - tstart))

                            # ###################################################
                            # prepare ROIs
                            # ###################################################
                            tstart = time.perf_counter()

                            # cut rois out
                            roi_size_skew = localize.get_skewed_roi_size(roi_size, theta, dc, dstage, ensure_odd=True)
                            rois, img_rois, xrois, yrois, zrois = zip(*[localize.get_skewed_roi(c, imgs, x, y, z, roi_size_skew) for c in centers_guess])
                            rois = np.asarray(rois)

                            # exclude some regions of roi
                            roi_masks = [localize.get_roi_mask(c, (np.inf, 0.5 * roi_size[1]), (zrois[ii], yrois[ii], xrois[ii])) for
                                         ii, c in enumerate(centers_guess)]

                            # mask regions
                            xrois, yrois, zrois, img_rois = zip(
                                *[(xr[rm][None, :], yr[rm][None, :], zr[rm][None, :], ir[rm][None, :])
                                  for xr, yr, zr, ir, rm in zip(xrois, yrois, zrois, img_rois, roi_masks)])

                            # extract guess values
                            bgs = np.array([np.mean(r) for r in img_rois])
                            sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in
                                            zip(img_rois, xrois, centers_guess)])
                            sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in
                                            zip(img_rois, yrois, centers_guess)])
                            sxys = np.expand_dims(0.5 * (sxs + sys), axis=1)
                            szs = np.expand_dims(np.array(
                                [np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in
                                 zip(img_rois, zrois, centers_guess)]), axis=1)

                            # get initial parameter guesses
                            init_params = np.concatenate((np.expand_dims(amps, axis=1),
                                                          centers_guess[:, 2][:, None],
                                                          centers_guess[:, 1][:, None],
                                                          centers_guess[:, 0][:, None],
                                                          sxys, szs,
                                                          np.expand_dims(bgs, axis=1)),
                                                         axis=1)

                            print("Prepared %d rois and estimated initial parameters in %0.2fs" % (len(rois), time.perf_counter() - tstart))

                            # ###################################################
                            # localization
                            # ###################################################
                            print("starting fitting for %d rois" % centers_guess.shape[0])
                            tstart = time.perf_counter()

                            fit_params, fit_states, chi_sqrs, niters, fit_t = localize.fit_rois(img_rois, (zrois, yrois, xrois),
                                                                                       init_params, estimator="LSE",
                                                                                       sf=1, dc=dc, angles=(0., theta, 0.))

                            tend = time.perf_counter()
                            print("Localization took %0.2fs" % (tend - tstart))

                            # fitting
                            print("Fitting %d rois on GPU" % (len(rois)))
                            fit_results = np.concatenate((np.expand_dims(fit_states, axis=1),
                                                          np.expand_dims(chi_sqrs, axis=1),
                                                          np.expand_dims(niters, axis=1)), axis=1)

                            # ###################################################
                            # preliminary fitting of results
                            # ###################################################
                            tstart = time.perf_counter()

                            to_keep, conditions, condition_names, filter_settings = localize.filter_localizations(
                                fit_params, init_params, (z, y, x),
                                (sigma_z, sigma_xy), min_spot_sep,
                                (sigmas_min, sigmas_max),
                                fit_thresholds[ch],
                                dist_boundary_min=(0.5 * sigma_z, sigma_xy))

                            print("identified %d/%d localizations in %0.3f" % (np.sum(to_keep), to_keep.size, time.perf_counter() - tstart))

                            if debug:
                                ind = len(to_keep) // 2
                                x, y, z = localize.get_skewed_coords(imgs.shape, dc, dstage, theta)
                                x += x_offset
                                y += y_offset
                                localize.plot_skewed_roi(fit_params[ind], rois[ind], imgs, theta, x, y, z, init_params[ind])

                            # store results
                            fit_params_vol.append(fit_params)
                            init_params_vol.append(init_params)
                            rois_vol.append(rois)
                            fit_results_vol.append(fit_results)
                            conditions_vol.append(conditions)
                            to_keep_vol.append(to_keep)

                        chunk_counter_p += 1

                        # update chunk counters
                        if ix_end < nxp:
                            chunk_counter_x += 1
                            ichunk += 1
                        elif ip_end < nimgs_per_vol:
                            chunk_counter_x = 0
                            chunk_counter_p += 1
                            ichunk += 1
                        else:
                            more_chunks = False

                    # assemble results
                    fit_params_vol = np.concatenate(fit_params_vol, axis=0)
                    init_params_vol = np.concatenate(init_params_vol, axis=0)
                    rois_vol = np.concatenate(rois_vol, axis=0)
                    fit_results_vol = np.concatenate(fit_results_vol, axis=0)
                    conditions_vol = np.concatenate(conditions_vol, axis=0)
                    to_keep_vol = np.concatenate(to_keep_vol, axis=0)

                    tend = time.perf_counter()
                    volume_process_times[vv] = tend - tstart_vol

                    # save results
                    localization_settings = {"filter_sigma_small_um": filter_sigma_small,
                                             "filter_sigma_large_um": filter_sigma_large,
                                             "roi_size_um": roi_size, "min_dists_um": min_spot_sep,
                                             "sigmas_min_um": sigmas_min, "sigmas_max_um": sigmas_max,
                                             "threshold": thresholds[ch],
                                             "chunk_size": chunk_size_planes, "chunk_overlap": chunk_overlap
                                             }
                    physical_data = {"dc": dc, "dstep": dstage, "frame_time_ms": frame_time_ms, "na": na, "ni": ni,
                                     "excitation_wavelength": excitation_wavelengths[ch],
                                     "emission_wavelength": emission_wavelengths[ch],
                                     "sigma_xy_ideal": sigma_xy, "sigma_z_ideal": sigma_z}

                    full_results = {"fit_params": fit_params_vol, "init_params": init_params_vol,
                                    "rois": rois_vol, "fit_results": fit_results_vol,
                                    "to_keep": to_keep_vol, "conditions": conditions_vol,
                                    "condition_names": condition_names,
                                    "volume_size_pix": (len(dset.axes["z"]), dset.image_height, dset.image_width),
                                    "volume_um3": volume_um3, "elapsed_t": volume_process_times[vv],
                                    "localization_settings": localization_settings,
                                    "filter_settings": filter_settings, "physical_data": physical_data}


                    fname = os.path.join(save_dir, "localization_round=%d_ch=%d_tile=%d_z=%d_t=%d.pkl" % (round, ch + 1, tl, iz, vv))
                    with open(fname, "wb") as f:
                        pickle.dump(full_results, f)

                # ###############################
                # print timing information
                # ###############################
                print("Found %d centers in              : %s" % (np.sum(to_keep_vol), str(datetime.timedelta(seconds=volume_process_times[vv]))))
                print("Total elapsed time (folder)      : %s" % datetime.timedelta(seconds=tend - tstart_folder))

                time_remaining_vol = datetime.timedelta(seconds=np.mean(volume_process_times[:vv + 1]) * (nt - vv - 1))
                print("Estimated time remaining (folder): %s" % str(time_remaining_vol))

                print("Total elapsed time (all)         : %s" % datetime.timedelta(seconds=tend - tstart_all))