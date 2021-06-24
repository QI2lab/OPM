import glob
import re
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import localize
import pycromanager
import sys

# add path for data_io.py
fdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(fdir)

import data_io

# paths to image files
root_dirs = [r"\\10.206.26.21\opm2\20210624a\glycerol80_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624b\glycerol80_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624c\glycerol80_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624d\glycerol80_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624e\glycerol60_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624f\glycerol60_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624g\glycerol60_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624h\glycerol60_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624i\glycerol50_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624j\glycerol50_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624k\glycerol50_1\Full resolution",
             r"\\10.206.26.21\opm2\20210624l\glycerol50_1\Full resolution"
            ]


for rd in root_dirs:
    if not os.path.exists(rd):
        raise ValueError("path %s does not exist" % rd)

tstart_all = time.perf_counter()
for ii, root_dir in enumerate(root_dirs):
    tstart_folder = time.perf_counter()

    dset_dir, _ = os.path.split(root_dir)

    # paths to metadata
    scan_data_dir = os.path.join(root_dir, "..", "..", "scan_metadata.csv")

    # ###############################
    # load/set scan parameters
    # ###############################
    scan_data = data_io.read_metadata(scan_data_dir)

    nvols = scan_data["num_t"]
    nimgs_per_vol = scan_data["scan_axis_positions"]
    nyp = scan_data["y_pixels"]
    nxp = scan_data["x_pixels"]
    dc = scan_data["pixel_size"] / 1000
    dstage = scan_data["scan_step"] / 1000
    theta = scan_data["theta"] * np.pi / 180
    normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel

    # trapezoid volume
    volume_um3 = (dstage * nimgs_per_vol) * (dc * np.sin(theta) * nyp) * (dc * nxp)

    na = 1.
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

    save_dir = os.path.join(root_dir, "..", "%s_localization" % time_stamp)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # ###############################
    # identify candidate points in opm data
    # ###############################
    absolute_threshold = 50
    # difference of gaussian filter parameters
    filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
    filter_sigma_large = (5 * sigma_z, 20 * sigma_xy, 20 * sigma_xy)
    # fit roi size
    roi_size = (3 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
    # assume points closer together than this come from a single bead
    min_spot_sep = (3 * sigma_z, 2 * sigma_xy)
    # exclude points with sigmas outside these ranges
    sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy)
    sigmas_max = (2 * sigma_z, 3 * sigma_xy)

    # don't consider any points outside of this polygon
    # cx, cy
    # allowed_camera_region = np.array([[357, 0], [464, 181], [1265, 231], [1387, 0]])
    allowed_camera_region = None

    # ###############################
    # load dataset
    # ###############################
    dset = pycromanager.Dataset(dset_dir)

    md = dset.read_metadata(z=0, t=0, c=0)
    frame_time_ms = float(md["OrcaFusionBT-Exposure"])

    if False:
        # get real time stamps
        time_stamps = []
        for it in range(nvols):
            for iz in range(nimgs_per_vol):
                md = dset.read_metadata(z=iz, t=it, c=0)
                time_stamps.append(datetime.datetime.strptime(md["TimeReceivedByCore"], "%Y-%m-%d %H:%M:%S.%f"))

        dts = []
        for ii in range(len(time_stamps) - 1):
            dts.append((time_stamps[ii + 1] - time_stamps[ii]).microseconds / 1000)

    # ###############################
    # loop over volumes
    # ###############################

    imgs_per_chunk = 100
    n_chunk_overlap = 3
    volume_process_times = np.zeros(nvols)
    for vv in range(nvols):
        print("################################\nstarting volume %d/%d in %s" % (vv + 1, nvols, dset_dir))
        tstart_vol = time.perf_counter()

        # variables to store results. List of results for each chunk
        fit_params_vol = []
        init_params_vol = []
        rois_vol = []
        fit_results_vol = []
        to_keep_vol = []
        conditions_vol = []

        # loop over chunks of images
        more_chunks = True
        chunk_counter = 0
        while more_chunks:

            # load images
            tstart_load = time.perf_counter()
            img_start = int(np.max([chunk_counter * imgs_per_chunk - n_chunk_overlap, 0]))
            img_end = int(np.min([img_start + imgs_per_chunk, nimgs_per_vol]))

            imgs = []
            for kk in range(img_start, img_end):
                imgs.append(dset.read_image(z=kk, t=vv, c=0))
            imgs = np.asarray(imgs)

            # if no images returned, we are done
            if imgs.shape[0] == 0:
                break

            imgs = np.flip(imgs, axis=0) # to match deskew convention...

            tend_load = time.perf_counter()
            print("loaded images in %0.2fs" % (tend_load - tstart_load))

            # get image coordinates
            npos, ny, nx = imgs.shape
            y_offset = img_start * dstage

            # do localization
            rois, fit_params, init_params, fit_results, imgs_filtered, coords = localize.localize_skewed(
                imgs, {"dc": dc, "dstep": dstage, "theta": theta}, absolute_threshold, roi_size,
                filter_sigma_small, filter_sigma_large, min_spot_sep,
                offsets=(0, y_offset, 0), allowed_polygon=allowed_camera_region, mode="fit")



            # filter results
            to_keep, conditions, condition_names, filter_settings = localize.filter_localizations(fit_params, init_params, coords,
                                                                                 (sigma_z, sigma_xy), min_spot_sep,
                                                                                 (sigmas_min, sigmas_max),
                                                                                 0.5 * absolute_threshold,
                                                                                 dist_boundary_min=(0.5*sigma_z, sigma_xy))

            # ind = 333
            # x, y, z = localize.get_skewed_coords(imgs.shape, dc, dstage, theta)
            # localize.plot_skewed_roi(fit_params[ind], rois[ind], imgs, theta, x, y, z, init_params[ind])

            # store results
            fit_params_vol.append(fit_params)
            init_params_vol.append(init_params)
            rois_vol.append(rois)
            fit_results_vol.append(fit_results)
            conditions_vol.append(conditions)
            to_keep_vol.append(to_keep)

            chunk_counter += 1

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
                                 "threshold": absolute_threshold,
                                 "chunk_size": imgs_per_chunk, "chunk_overlap": n_chunk_overlap
                                 }
        physical_data = {"dc": dc, "dstep": dstage, "frame_time_ms": frame_time_ms, "na": na, "ni": ni,
                         "excitation_wavelength": excitation_wavelength, "emission_wavelength": emission_wavelength,
                         "sigma_xy_ideal": sigma_xy, "sigma_z_ideal": sigma_z}

        full_results = {"fit_params": fit_params_vol, "init_params": init_params_vol,
                        "rois": rois_vol, "fit_results": fit_results_vol,
                        "to_keep": to_keep_vol, "conditions": conditions_vol, "condition_names": condition_names,
                        "volume_size_pix": (len(dset.axes["z"]), dset.image_height, dset.image_width),
                        "volume_um3": volume_um3, "elapsed_t": volume_process_times[vv],
                        "localization_settings": localization_settings,
                        "filter_settings": filter_settings, "physical_data": physical_data}

        fname = os.path.join(save_dir, "localization_results_vol_%d.pkl" % vv)
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
        time_remaining_all = datetime.timedelta(seconds=np.mean(volume_process_times[:vv + 1]) *
                                                       (nvols - vv - 1 + nvols * (len(root_dirs) - ii - 1)))
        print("Estimated time remaining (all)   : %s" % str(time_remaining_all))
