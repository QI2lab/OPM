"""
Fit single spots to model with variable lightsheet angle/stage step. Useful as a tool for checking calibration of system
"""
import glob
import re
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import localize
import load_dataset
import pycromanager
import data_io
import scipy

tbegin = time.perf_counter()

# basic parameters
plot_results = False

# paths to image files
root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210430a", "beads_1", "Full resolution")
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
normal = np.array([0, -np.sin(theta), np.cos(theta)])  # normal of camera pixel

volume_um3 = (dstage * nimgs_per_vol) * (dc * nxp) * (dc * np.cos(theta) * nyp)

frame_time_ms = 2
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
# time_stamp = "test"

save_dir = os.path.join(root_dir, "..", "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ###############################
# identify candidate points in opm data
# ###############################
absolute_threshold = 500
# difference of gaussian filer
filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
filter_sigma_large = (5 * sigma_z, 20 * sigma_xy, 20 * sigma_xy)
# fit roi size
roi_size = (3 * sigma_z, 8 * sigma_xy, 8 * sigma_xy)
# assume points closer together than this come from a single bead
min_dists = (3 * sigma_z, 2 * sigma_xy, 2 * sigma_xy)
# exclude points with sigmas outside these ranges
sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
sigmas_max = (2 * sigma_z, 2 * sigma_xy, 2 * sigma_xy)

# ###############################
# load dataset
# ###############################
dset = pycromanager.Dataset(dset_dir)

# ###############################
# loop over volumes
# ###############################

imgs_per_chunk = 100
n_chunk_overlap = 3
volume_process_times = np.zeros(nvols)
for vv in range(nvols):
    print("################################\nstarting volume %d/%d" % (vv + 1, nvols))
    tstart = time.perf_counter()

    # variables to store results. List of results for each chunk
    fit_params_vol = []
    rois_vol = []
    centers_guess_vol = []

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

        imgs = np.flip(imgs, axis=0)  # to match deskew convention...

        tend_load = time.perf_counter()
        print("loaded images in %0.2fs" % (tend_load - tstart_load))

        # get image coordinates
        npos, ny, nx = imgs.shape
        gn = np.arange(npos) * dstage
        y_offset = img_start * dstage

        # find points to fit
        # try to do filtering in innocuous way for whatever the real coordinates are
        ksmall = np.ones((7, 7, 7))
        ksmall = ksmall / np.sum(ksmall)
        imgs_filtered = localize.filter_convolve(imgs, ksmall)

        footprint_form = np.ones((15, 30, 30), dtype=np.bool)
        img_max_filtered = scipy.ndimage.maximum_filter(imgs_filtered, footprint=footprint_form)
        centers_guess_inds = np.argwhere(np.logical_and(imgs_filtered == img_max_filtered, imgs > absolute_threshold))

        roi_shape = (13, 30, 30)
        rois = np.array([localize.get_centered_roi(c, roi_shape, min_vals=(0, 0, 0), max_vals=imgs.shape) for c in centers_guess_inds])

        plotted_counter = 0
        max_plot = np.inf
        fit_ps = np.zeros((len(rois), 9))
        for ind in range(len(rois)):
            roi = rois[ind]
            img_roi = localize.cut_roi(roi, imgs)
            current_roi_shape = (roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4])

            def fit_fn(p): return localize.gaussian3d_angle(current_roi_shape, dc, p)
            def jac_fn(p): return localize.gaussian3d_angle_jacobian(current_roi_shape, dc, p)

            # guess params
            xg, yg, zg = localize.get_skewed_coords(current_roi_shape, dc, dstage, theta)
            ind_max = np.unravel_index(np.argmax(img_roi), current_roi_shape)
            init_params = [np.nanmax(img_roi), xg[0, 0, ind_max[2]], yg[ind_max[0], ind_max[1], 0], zg[0, ind_max[1], 0], sigma_xy, sigma_z,
                           np.nanmean(img_roi), theta, dstage]
            fixed_params = [False, False, False, False, False, False, False, True, False]
            results = localize.fit_model(img_roi, fit_fn, init_params, fixed_params=fixed_params, model_jacobian=jac_fn)

            xf, yf, zf = localize.get_skewed_coords(current_roi_shape, dc, results["fit_params"][-1], results["fit_params"][-2])
            cfit = np.array([results["fit_params"][3], results["fit_params"][2], results["fit_params"][1]])

            fit_ps[ind] = results["fit_params"]

            if results["fit_params"][0] < 100 or results["fit_params"][4] < 0.5 * sigma_xy: # or not localize.point_in_trapezoid(cfit, xf, yf, zf):
                continue

            if plotted_counter > max_plot:
                break

            img_fit = fit_fn(results["fit_params"])
            img_guess = fit_fn(init_params)

            if plot_results:
                plotted_counter += 1

                plt.figure()
                plt.suptitle("%d, angle=%0.2fdeg, ds=%0.3f" % (ind, results["fit_params"][-2] * 180/np.pi, results["fit_params"][-1]))
                grid = plt.GridSpec(3, 3)

                for ii in range(3):
                    plt.subplot(grid[0, ii])
                    plt.imshow(np.nanmax(img_roi, axis=ii))
                    plt.title("data")

                    plt.subplot(grid[1, ii])
                    plt.imshow(np.nanmax(img_fit, axis=ii))
                    plt.title("fit")

                    plt.subplot(grid[2, ii])
                    plt.imshow(np.nanmax(img_guess, axis=ii))
                    plt.title("guess")


        fit_params_vol.append(fit_ps)
        centers_guess_vol.append(centers_guess_inds)
        rois_vol.append(rois)

    fit_params_vol = np.concatenate(fit_params_vol)
    centers_guess_vol = np.concatenate(centers_guess_vol)
    rois_vol = np.concatenate(rois_vol)

    tend = time.perf_counter()
    volume_process_times[vv] = tend - tstart

    figh = plt.figure()

    steps, bin_edges = np.histogram(fit_params_vol[:, -1], 15)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.plot(bin_centers, steps)
    plt.xlabel("Step size (um)")
    plt.ylabel('counts')
    plt.title("histogram of fit step size, mean=%0.3f" % np.mean(fit_params_vol[:, -1]))
    plt.xlim([0, 0.4])

    # save results
    # full_results = {"centers": centers_unique, "centers_guess": centers_guess_vol,
    #                 "fit_params": fit_params_unique, "rois": rois_unique,
    #                 "volume_um3": volume_um3, "frame_time_ms": frame_time_ms, "elapsed_t": volume_process_times[vv]}
    # fname = os.path.join(save_dir, "localization_results_vol_%d.pkl" % vv)
    # with open(fname, "wb") as f:
    #     pickle.dump(full_results, f)

