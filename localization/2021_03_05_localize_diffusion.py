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

# basic parameters
plot_extra = False
plot_results = False
figsize = (16, 8)

na = 1.
ni = 1.4
excitation_wavelength = 0.561
emission_wavelength = 0.605
sigma_xy = 0.22 * emission_wavelength / na
sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)

# load data
root_dir = r"\\10.206.26.21\opm2\20210305a\crowders_densest_50glycerol"
fnames = glob.glob(os.path.join(root_dir, "*.tif"))
img_inds = np.array([int(re.match(".*Image1_(\d+).tif", f).group(1)) for f in fnames])
fnames = [f for _, f in sorted(zip(img_inds, fnames))]

# paths to relevant data
data_dir = root_dir
scan_data_dir = os.path.join(root_dir, "..", "galvo_scan_params.pkl")

# load data
with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

# load/set scan parameters
nvols = 10000
nimgs = 25
nyp = 256
nxp = 1600

theta = scan_data["theta"][0] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel

dc = scan_data["pixel size"][0] / 1000
dstage = scan_data["scan step"][0] / 1000

# build save dir
save_dir = os.path.join(root_dir, "%s_localization" % time_stamp)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# ###############################
# identify candidate points in opm data
# ###############################
nmax_fit_tries = 1
imgs_per_chunk = np.min([100, nimgs])
n_chunk_overlap = 3
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

# loop over volumes
for vv in range(nvols):
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
        imgs = np.zeros((nimgs_temp, nyp, nxp))
        for kk in range(img_start, img_end):
            imgs[kk] = tifffile.imread(fnames[vv * nimgs + kk])
        imgs = np.flip(imgs, axis=1) # to mach the conventions I have been using

        # get image coordinates
        npos, ny, nx = imgs.shape
        gn = np.arange(npos) * dstage

        # y_offset = (aa - n_chunk_overlap) * dstage
        y_offset = img_start * dstage

        # do localization
        imgs_filtered, centers_unique, fit_params_unique, rois_unique, centers_guess = localize.localize(
            imgs, {"dc": dc, "dstep": dstage, "theta": theta}, thresh, xy_roi_size, z_roi_size, 0, 0, min_z_dist,
            min_xy_dist, sigma_xy_max, sigma_xy_min, sigma_z_max, sigma_z_min, nmax_try=nmax_fit_tries,
            y_offset=y_offset)

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
        xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs, dc, dstage, theta, mode="ortho-interp")
        tifffile.imsave(os.path.join(save_dir, "max_proj_xy_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=0))
        tifffile.imsave(os.path.join(save_dir, "max_proj_xz_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=1))
        tifffile.imsave(os.path.join(save_dir, "max_proj_yz_vol=%d_chunk=%d.tiff" % (vv, aa)), np.nanmax(imgs_unskew, axis=2))

    # assemble results
    centers_unique_all = np.concatenate(centers_unique_all, axis=0)
    fit_params_unique_all = np.concatenate(fit_params_unique_all, axis=0)
    rois_unique_all = np.concatenate(rois_unique_all, axis=0)
    # centers_fit_sequence_all = np.concatenate(centers_fit_sequence_all, axis=0)
    centers_guess_all = np.concatenate(centers_guess_all, axis=0)

    # since left some overlap at the edges, have to again combine results
    centers_unique, unique_inds = localize.combine_nearby_peaks(centers_unique_all, min_xy_dist, min_z_dist, mode="keep-one")
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
    fname = os.path.join(save_dir, "localization_results_vol_%d.pkl" % vv)
    with open(fname, "wb") as f:
        pickle.dump(full_results, f)
