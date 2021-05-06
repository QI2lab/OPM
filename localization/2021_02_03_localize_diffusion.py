"""
Localize diffusing beads in a series of volumes
"""
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import pycromanager

import localize

# basic parameters
plot_extra = False
plot_results = False
figsize = (16, 8)
now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

# root_dir = r"\\10.206.26.21\opm2\20210203\beads_50glyc_1000dilution_1"
root_dir = r"\\10.206.26.21\opm2\20210203\beads_0glyc_1000dilution_1"
save_dir = os.path.join(root_dir, "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# paths to relevant data
data_dir = root_dir
scan_data_dir = os.path.join(root_dir, "galvo_scan_params.pkl")

# load data
with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

ds = pycromanager.Dataset(data_dir, full_res_only=True)
# img, img_metadata = ds.read_image(read_metadata=True)
summary = ds.summary_metadata
# md = ds.read_metadata(channel=0, z=3)
nvols = len(ds.axes["t"])
nimgs = len(ds.axes["x"])
nyp = ds.image_height
nxp = ds.image_width


# load parameters
na = 1.
ni = 1.4
excitation_wavelength = 0.561
emission_wavelength = 0.605
sigma_xy = 0.22 * emission_wavelength / na
sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

theta = scan_data["theta"][0] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel

dc = scan_data["pixel size"][0] / 1000
dstage = scan_data["scan step"][0] / 1000


# ###############################
# identify candidate points in opm data
# ###############################
nmax_try = 1
nsingle = np.min([100, nimgs])
n_overlap = 3
thresh = 100
# difference of gaussian filer
xy_filter_small = 0.25 * sigma_xy
xy_filter_big = 20 * sigma_xy
# fit roi size
xy_roi_size = 12 * sigma_xy
z_roi_size = 3 * sigma_z
# assume points closer together than this come from a single bead
min_z_dist = 3 * sigma_z
min_xy_dist = 2 * sigma_xy
# exclude points with sigmas outside these ranges
sigma_xy_max = 2 * sigma_xy
sigma_xy_min = 0.25 * sigma_xy
sigma_z_max = 2 * sigma_z
sigma_z_min = 0.25 * sigma_z

for vv in range(nvols):
    save_dir_vol = os.path.join(save_dir, "vol_%d" % vv)
    if not os.path.exists(save_dir_vol):
        os.mkdir(save_dir_vol)

    tstart = time.perf_counter()
    centers_unique_all = []
    fit_params_unique_all = []
    rois_unique_all = []
    centers_fit_sequence_all = []
    centers_guess_all = []
    for aa in range(0, nimgs, nsingle):
        save_dir_sub = os.path.join(save_dir_vol, "region_%d" % aa)
        if not os.path.exists(save_dir_sub):
            os.mkdir(save_dir_sub)

        img_start = int(np.max([aa - n_overlap, 0]))
        img_end = int(np.min([img_start + nsingle, ]))
        nimgs_temp = img_end - img_start

        imgs = np.zeros((nimgs_temp, nyp, nxp))
        for kk in range(img_start, img_end):
            imgs[kk] = ds.read_image(t=vv, x=kk)
        # imgs = np.flip(imgs, axis=1) # to mach the conventions I have been using
        imgs = np.flip(imgs, axis=0)  # to match deskew convention...

        npos, ny, nx = imgs.shape
        gn = np.arange(npos) * dstage

        y_offset = (aa - n_overlap) * dstage

        imgs_filtered, centers_unique, fit_params_unique, rois_unique, centers_guess = localize.localize(
            imgs, {"dc": dc, "dstep": dstage, "theta": theta}, thresh, xy_roi_size, z_roi_size, 0, 0, min_z_dist,
            min_xy_dist, sigma_xy_max, sigma_xy_min, sigma_z_max, sigma_z_min, offsets=(0, y_offset, 0))

        centers_unique_all.append(centers_unique)
        fit_params_unique_all.append(fit_params_unique)
        rois_unique_all.append(rois_unique)
        # centers_fit_sequence_all.append(centers_fit_sequence)
        centers_guess_all.append(centers_guess)

        # plot localization fit diagnostic on good points
        # picture coordinates in coverslip frame
        # x, y, z = localize.get_lab_coords(nx, ny, dc, theta, gn)
        x, y, z = localize.get_skewed_coords((npos, ny, nx), dc, dstage, theta)
        y += y_offset

        if plot_results:
            plt.ioff()
            plt.switch_backend("agg")
            print("plotting %d ROI's" % len(fit_params_unique))
            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(localize.plot_roi)(fit_params_unique[ii], rois_unique[ii], imgs_filtered, theta, x, y, z,
                                          figsize=figsize, prefix=("%04d" % ii), save_dir=save_dir_sub)
                for ii in range(len(fit_params_unique)))

            # for debugging
            # for ii in range(len(fit_params_unique)):
            #     localize.plot_roi(fit_params_unique[ii], rois_unique[ii], imgs_filtered, theta, x, y, z,
            #     figsize=figsize, prefix=("%04d" % ii), save_dir=save_dir_sub)

            plt.ion()
            # plt.switch_backend("TkAgg")
            plt.switch_backend("Qt5Agg")

    # assemble results
    centers_unique_all = np.concatenate(centers_unique_all, axis=0)
    fit_params_unique_all = np.concatenate(fit_params_unique_all, axis=0)
    rois_unique_all = np.concatenate(rois_unique_all, axis=0)
    # centers_fit_sequence_all = np.concatenate(centers_fit_sequence_all, axis=0)
    centers_guess_all = np.concatenate(centers_guess_all, axis=0)

    # since left some overlap at the edges, have to again combine results
    centers_unique, unique_inds =localize.combine_nearby_peaks(centers_unique_all, min_xy_dist, min_z_dist, mode="keep-one")
    fit_params_unique = fit_params_unique_all[unique_inds]
    rois_unique = rois_unique_all[unique_inds]

    tend = time.perf_counter()
    elapsed_t = tend - tstart
    hrs = (elapsed_t) // (60 * 60)
    mins = (elapsed_t - hrs * 60 * 60) // 60
    secs = (elapsed_t - hrs * 60 * 60 - mins * 60)
    print("Found %d centers in %dhrs %dmins and %0.2fs" % (len(centers_unique), hrs, mins, secs))

    full_results = {"centers": centers_unique, "fit_params": fit_params_unique, "rois": rois_unique,
                    "elapsed_t": elapsed_t}
    fname = os.path.join(save_dir_vol, "localization_results.pkl")
    with open(fname, "wb") as f:
        pickle.dump(full_results, f)

"""
# ###############################
# interpolate images so are on grid s in coverslip coordinate system and plot all results
# ###############################
imgs_all = np.flip(ds_array.compute(), axis=1)
nstep, ny, nx = imgs_all.shape
gn = np.arange(nstep) * dstage
x, y, z = localize.get_lab_coords(nx, ny, dc, theta, gn)

xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs_all, dc, dstage, theta, mode="ortho-interp")
# xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs_all, dc, dstage, theta, mode="row-interp")
dxi = xi[1] - xi[0]
dyi = yi[1] - yi[0]
dzi = zi[1] - zi[0]

vmin = np.percentile(imgs_all, 0.1)
vmax = np.percentile(imgs_all, 99.99)

plt.figure(figsize=figsize)
plt.suptitle("Maximum intensity projection, XY\n"
             "wavelength=%0.0fnm, NA=%0.3f, n=%0.2f\n"
             "dc=%0.3fum, stage step=%0.3fum, dx interp=%0.3fum, dy interp=%0.3fum, dz interp =%0.3fum, theta=%0.2fdeg"
             % (emission_wavelength * 1e3, na, ni, dc, dstage, dxi, dyi, dzi, theta * 180 / np.pi))

plt.imshow(np.nanmax(imgs_unskew, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
if plot_extra:
    # plt.plot(centers_guess[:, 1], centers_guess[:, 2], 'gx')
    plt.plot(centers_fit_sequence_all[:, :, 1].ravel(), centers_fit_sequence_all[:, :, 2].ravel(), 'gx')
    # plt.plot(centers_fit[:, 1], centers_fit[:, 2], 'mx')
plt.plot(centers_unique_all[:, 1], centers_unique_all[:, 2], 'rx')
plt.xlabel("Y (um)")
plt.ylabel("X (um)")
plt.title("XY")

plt.figure(figsize=figsize)
plt.suptitle("Maximum intensity projection, XZ\n"
             "wavelength=%0.0fnm, NA=%0.3f, n=%0.2f\n"
             "dc=%0.3fum, stage step=%0.3fum, dx interp=%0.3fum, dy interp=%0.3fum, dz interp =%0.3fum, theta=%0.2fdeg"
             % (emission_wavelength * 1e3, na, ni, dc, dstage, dxi, dyi, dzi, theta * 180 / np.pi))

plt.imshow(np.nanmax(imgs_unskew, axis=1), vmin=vmin, vmax=vmax, origin="lower",
           extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
if plot_extra:
    # plt.plot(centers_guess[:, 2], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit_sequence_all[:, :, 2].ravel(), centers_fit_sequence_all[:, :, 0].ravel(), 'gx')
    # plt.plot(centers_fit[:, 2], centers_fit[:, 0], 'mx')
plt.plot(centers_unique_all[:, 2], centers_unique_all[:, 0], 'rx')
plt.xlabel("X (um)")
plt.ylabel("Z (um)")
plt.title("XZ")

plt.figure(figsize=figsize)
plt.suptitle("Maximum intensity projection, YZ\n"
             "wavelength=%0.0fnm, NA=%0.3f, n=%0.2f\n"
             "dc=%0.3fum, stage step=%0.3fum, dx interp=%0.3fum, dy interp=%0.3fum, dz interp =%0.3fum, theta=%0.2fdeg"
             % (emission_wavelength * 1e3, na, ni, dc, dstage, dxi, dyi, dzi, theta * 180 / np.pi))
plt.imshow(np.nanmax(imgs_unskew, axis=2), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
if plot_extra:
    # plt.plot(centers_guess[:, 1], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit_sequence_all[:, :, 1].ravel(), centers_fit_sequence_all[:, :, 0].ravel(), 'gx')
    # plt.plot(centers_fit[:, 1], centers_fit[:, 0], 'mx')
plt.plot(centers_unique_all[:, 1], centers_unique_all[:, 0], 'rx')
plt.xlabel("Y (um)")
plt.ylabel("Z (um)")
plt.title("YZ")

plt.show()
"""