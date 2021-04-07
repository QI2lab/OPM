import glob
import re
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import tifffile
import pycromanager
import localize

tbegin = time.perf_counter()

# basic parameters
plot_extra = False
plot_results = False
figsize = (16, 8)

frame_time_ms = 2
na = 1.
ni = 1.4
excitation_wavelength = 0.561
emission_wavelength = 0.605
sigma_xy = 0.22 * emission_wavelength / na
sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

# load data
root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210309", "crowders-10x-50glyc")
fnames = glob.glob(os.path.join(root_dir, "*.tif"))
img_inds = np.array([int(re.match(".*Image\d+_(\d+).tif", f).group(1)) for f in fnames])
fnames = [f for _, f in sorted(zip(img_inds, fnames))]

# paths to relevant data
data_dir = root_dir
scan_data_dir = os.path.join(root_dir, "galvo_scan_params.pkl")

# load data
with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

# load/set scan parameters
nvols = 10000
nimgs = 25
nyp = 256
nxp = 1600
volume_process_times = np.zeros(nvols)

theta = scan_data["theta"][0] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel

dc = scan_data["pixel size"][0] / 1000
dstage = scan_data["scan step"][0] / 1000

volume_um3 = (dstage * nimgs) * (dc * nxp) * (dc * np.cos(theta) * nyp)

# build save dir
now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

save_dir = os.path.join(root_dir, "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# ###############################
# identify candidate points in opm data
# ###############################
imgs_per_chunk = np.min([100, nimgs])
n_chunk_overlap = 3
absolute_threshold = 50
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

# don't consider any points outside of this polygon
# cx, cy
allowed_camera_region = np.array([[357, 0], [464, 181], [1265, 231], [1387, 0]])

# loop over volumes
for vv in range(nvols):
    print("################################\nstarting volume %d/%d" % (vv + 1, nvols))
    tstart = time.perf_counter()

    centers_unique_all = []
    fit_params_unique_all = []
    rois_unique_all = []
    centers_fit_sequence_all = []
    centers_guess_all = []
    for aa in range(0, nimgs, imgs_per_chunk):
        # handle analyzing images in chunks
        img_start = int(np.max([aa - n_chunk_overlap, 0]))
        img_end = int(np.min([img_start + imgs_per_chunk, ]))
        nimgs_temp = img_end - img_start

        # load images
        tstart_load = time.process_time()
        imgs = np.zeros((nimgs_temp, nyp, nxp))
        for kk in range(img_start, img_end):
            imgs[kk] = tifffile.imread(fnames[vv * nimgs + kk])
        imgs = np.flip(imgs, axis=1) # to mach the conventions I have been using

        tend_load = time.process_time()
        print("loaded images in %0.2fs" % (tend_load - tstart_load))

        # get image coordinates
        npos, ny, nx = imgs.shape
        gn = np.arange(npos) * dstage

        # y_offset = (aa - n_chunk_overlap) * dstage
        y_offset = img_start * dstage

        # do localization
        imgs_filtered, centers_unique, fit_params_unique, rois_unique, centers_guess = localize.localize(
            imgs, {"dc": dc, "dstep": dstage, "theta": theta}, absolute_threshold, roi_size,
            filter_sigma_small, filter_sigma_large, min_dists, (sigmas_min, sigmas_max),
            y_offset=y_offset, allowed_polygon=allowed_camera_region, mode="fit")

        centers_unique_all.append(centers_unique)
        fit_params_unique_all.append(fit_params_unique)
        rois_unique_all.append(rois_unique)
        # centers_fit_sequence_all.append(centers_fit_sequence)
        centers_guess_all.append(centers_guess)

        # plot localization fit diagnostic on good points
        # picture coordinates in coverslip frame
        x, y, z = localize.get_lab_coords(nx, ny, dc, theta, gn)
        y += y_offset

        # max projections
        # tstart = time.process_time()
        # xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs, dc, dstage, theta, mode="ortho-interp")
        ## tifffile.imsave(os.path.join(save_dir, "vol=%d_chunk=%d.tiff") % (vv, aa), imgs_unskew)
        # tifffile.imsave(os.path.join(save_dir, "max_proj_xy_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=0))
        # tifffile.imsave(os.path.join(save_dir, "max_proj_xz_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=1))
        # tifffile.imsave(os.path.join(save_dir, "max_proj_yz_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=2))
        #
        # tend = time.process_time()
        # print("deskewing and saving images took %0.2fs" % (tend - tstart))

    # assemble results
    centers_unique_all = np.concatenate(centers_unique_all, axis=0)
    fit_params_unique_all = np.concatenate(fit_params_unique_all, axis=0)
    rois_unique_all = np.concatenate(rois_unique_all, axis=0)
    # centers_fit_sequence_all = np.concatenate(centers_fit_sequence_all, axis=0)
    centers_guess_all = np.concatenate(centers_guess_all, axis=0)

    if nimgs > imgs_per_chunk:
        # since left some overlap at the edges, have to again combine results
        centers_unique, unique_inds = localize.combine_nearby_peaks(centers_unique_all, min_dists[1], min_dists[0], mode="keep-one")
        fit_params_unique = fit_params_unique_all[unique_inds]
        rois_unique = rois_unique_all[unique_inds]
    else:
        centers_unique = centers_unique_all
        fit_params_unique = fit_params_unique_all
        rois_unique = rois_unique_all

    tend = time.perf_counter()
    volume_process_times[vv] = tend - tstart

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

    if vv > 0:
        time_remaining = np.mean(volume_process_times[:vv]) * (nvols - vv - 1)
        days = time_remaining // (24 * 60 * 60)
        hrs = (time_remaining - days * 24 * 60 * 60) // (60 * 60)
        mins = (time_remaining - days * 24 * 60 * 60 - hrs * 60 * 60) // 60
        secs = (time_remaining - days * 24 * 60 * 60 - hrs * 60 * 60 - mins * 60)
        print("Estimated time remaining: %ddays %dhrs %dmins and %0.2fs" % (days, hrs, mins, secs))

    # save results
    full_results = {"centers": centers_unique, "centers_guess": centers_guess_all,
                    "fit_params": fit_params_unique, "rois": rois_unique,
                    "volume_um3": volume_um3, "frame_time_ms": frame_time_ms, "elapsed_t": elapsed_t}
    fname = os.path.join(save_dir, "localization_results_vol_%d.pkl" % vv)
    with open(fname, "wb") as f:
        pickle.dump(full_results, f)
