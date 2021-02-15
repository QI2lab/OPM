"""
2/5/2021, Peter T. Brown
"""
import os
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import skimage.filters
import scipy.signal
import dask
import joblib
import pycromanager

import localize

# basic parameters
plot_extra = False
figsize = (16, 8)
now = datetime.datetime.now()
time_stamp = '%04d_%02d_%02d_%02d;%02d;%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
save_dir = "./%s_localization_fits" % time_stamp
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# paths to relevant data
data_dir = r"\\10.206.26.21\opm2\20210202\beads_561nm_200nm_step\beads561_200nm_r0000_y0000_z0000_1"
stage_data_dir = r"\\10.206.26.21\opm2\20210202\beads_561nm_200nm_step\stage_positions.pkl"
scan_data_dir = r"\\10.206.26.21\opm2\20210202\beads_561nm_200nm_step\stage_scan_params.pkl"

# load data
with open(stage_data_dir, "rb") as f:
    stage_data = pickle.load(f)

with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

ds = pycromanager.Dataset(data_dir)
# img, img_metadata = ds.read_image(read_metadata=True)
summary = ds.summary_metadata
# md = ds.read_metadata(channel=0, z=3)

# dataset as dask array
# each image is 256 x 1600, with 256 direction along the lightsheet, i.e. the y'-direction
ds_array = ds.as_array()
# imgs = ds_array[50:100, :, :400].compute()
# imgs = ds_array[50:100].compute()
imgs = ds_array.compute()
imgs = np.flip(imgs, axis=1) # to mach the conventions I have been using

# load parameters
na = 1.
ni = 1.4
excitation_wavelength = 0.561
emission_wavelength = 0.605
sigma_xy = 0.22 * emission_wavelength / na
sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

theta = scan_data["theta"][0] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel

npos, ny, nx = imgs.shape
dc = scan_data["pixel size"][0] / 1000
dstage = scan_data["scan step"][0] / 1000
gn = np.arange(npos) * dstage

# ###############################
# get coordinates
# ###############################
# picture coordinates in coverslip frame
x, y, z = localize.get_lab_coords(nx, ny, dc, theta, gn)

# picture coordinates
xp = dc * np.arange(nx)
yp = dc * np.arange(ny)

# ###############################
# identify candidate points in opm data
# ###############################

# difference of gaussian's filter
# sig_xy_big = 5 * sigma_xy
# sig_z_big = 5 * sigma_z
# sig_xy_small = 0.25 * sigma_xy
# sig_z_small = 0.25 * sigma_z

# another approach to differnece of guassians filter... using real coordinates
# problem: with the convolve functions I get strange behavior near the boundaries.
# dxy_min = 4 * sig_xy_big
# dz_min = 3 * sig_xy_big
#
# # same as x-di
# n3k = 2 * int(np.ceil(dxy_min / dc / 2)) + 1
# # set so insure maximum desired z-point is contained
# n2k = 2 * int(np.ceil(dz_min / (dc * np.sin(theta)))) + 1
# # set so that @ top and bottom z-points catch desired y-points
# n1k = 2 * int(np.ceil((0.5 * (n2k + 1)) * dc * np.cos(theta) + dxy_min) / dstage) + 1
# if np.mod(n1k, 2) == 0:
#     n1k += 1
# xc = (n3k - 1) / 2 * dc
# yc = (n2k - 1) / 2 * dc * np.cos(theta) + (n1k - 1) / 2 * dstage
# zc = (n2k - 1) / 2 * dc * np.sin(theta)
# xk, yk, zk = get_lab_coords(n3k, n2k, dc, theta, np.arange(n1k) * dstage)
# xk -= xc
# yk -= yc
# zk -= zc
# kernel1 = np.exp(-xk**2/(2*sig_xy_small**2) - yk**2/(2*sig_xy_small**2) - zk**2/(2*sig_z_small))
# kernel1 = kernel1 / np.sum(kernel1)
# kernel2 = np.exp(-xk**2/(2*sig_xy_big**2) - yk**2/(2*sig_xy_big**2) - zk**2/(2*sig_z_big))
# kernel2 = kernel2 / np.sum(kernel2)
# kernel = kernel1 - kernel2
# imgs_filtered2 = scipy.signal.oaconvolve(imgs, kernel2, mode="same")

# todo: may want to make own kernel and do convolution, rather than this
imgs_filtered = skimage.filters.difference_of_gaussians(np.array(imgs, dtype=np.float),
                                                        0.25 * (sigma_xy / dc),
                                                        high_sigma=(10 * sigma_xy / dc, 5 * sigma_z / dstage, 10 * sigma_xy / dc))

thresh = 50
centers_guess_inds = localize.find_candidate_beads(imgs_filtered, filter_xy_pix=0, filter_z_pix=0, max_thresh=thresh, mode="threshold")
xc = x[0, 0, centers_guess_inds[:, 2]]
yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)
print("Found %d points above threshold" % len(centers_guess))

# average multiple points too close together
min_z_dist = 3 * sigma_z
min_xy_dist = 2 * sigma_xy
centers_guess = localize.combine_nearby_peaks(centers_guess, min_xy_dist, min_z_dist, mode="average")
print("Found %d points well separated" % len(centers_guess))

# ###############################
# do localization
# ###############################
# roi sizes
xy_roi_size = 12 * sigma_xy
z_roi_size = 3 * sigma_z

print("fitting %d rois" % centers_guess.shape[0])
results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
    joblib.delayed(localize.fit_roi)(centers_guess[ii], imgs_filtered, dc, theta, x, y, z, xy_roi_size, z_roi_size, 0.2, 3) for ii in range(len(centers_guess)))
# for ii in range(len(centers_guess)):
#     localize.fit_roi(centers_guess[ii], imgs_filtered, dc, theta, x, y, z, xy_roi_size, z_roi_size, 0.2, 3)

# # unpack results
fit_params, ntries, rois, centers_fit_sequence = list(zip(*results))
fit_params = np.asarray(fit_params)
ntries = np.asarray(ntries)
rois = np.asarray(rois)
centers_fit_sequence = np.asarray(centers_fit_sequence)
centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]), axis=1)

# only keep points if think the fit was good
# first check center is within our region
to_keep_x = np.logical_and(centers_fit[:, 2] >= x.min(), centers_fit[:, 2] <= x.max())
to_keep_yz = np.logical_and(centers_fit[:, 0] <= centers_fit[:, 1] * np.tan(theta),
                            centers_fit[:, 0] >= (centers_fit[:, 1] - dstage * npos) * np.tan(theta))
to_keep_z = np.logical_and(centers_fit[:, 0] >= 0, centers_fit[:, 0] <= ny * dc * np.sin(theta))

to_keep_position = np.logical_and(np.logical_and(to_keep_x, to_keep_yz), to_keep_z)

# check that size was reasonable
to_keep_sigma_xy = np.logical_and(fit_params[:, 4] <= sigma_xy*3, fit_params[:, 4] >= 0.25 * sigma_xy)
to_keep_sigma_z = np.logical_and(fit_params[:, 5] <= sigma_z*2, fit_params[:, 5] >= 0.25 * sigma_z)

to_keep = np.logical_and(np.logical_and(to_keep_sigma_xy, to_keep_sigma_z), to_keep_position)

# plot localization fit diagnostic on good points
# plt.ioff()
# plt.switch_backend("agg")
# print("plotting %d ROI's" % np.sum(to_keep))
# inds = np.arange(len(to_keep))[to_keep]
# results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
#     joblib.delayed(plot_roi)(fit_params[ii], rois[ii], imgs_filtered, theta, x, y, z,
#                               center_guess=centers_fit_sequence[ii], figsize=figsize,
#                               prefix=("%04d" % ii), save_dir=save_dir)
#     for ii in inds)

# plt.ion()
# # plt.switch_backend("TkAgg")
# plt.switch_backend("Qt5Agg")


# only keep unique center if close enough
centers_unique = localize.combine_nearby_peaks(centers_fit[to_keep], min_xy_dist, min_z_dist, mode="keep-one")
print("identified %d unique points" % len(centers_unique))

# ###############################
# interpolate images so are on grids in coverslip coordinate system and plot all results
# ###############################
xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs_filtered, dc, dstage, theta, mode="ortho-interp")
dxi = xi[1] - xi[0]
dyi = yi[1] - yi[0]
dzi = zi[1] - zi[0]

vmin = np.percentile(imgs_filtered, 0.1)
vmax = np.percentile(imgs_filtered, 99.99)

plt.figure(figsize=figsize)
grid = plt.GridSpec(2, 2)
plt.suptitle("Maximum intensity projection comparison\n"
             "wavelength=%0.0fnm, NA=%0.3f, n=%0.2f\n"
             "dc=%0.3fum, stage step=%0.3fum, dx interp=%0.3fum, dy interp=%0.3fum, dz interp =%0.3fum, theta=%0.2fdeg"
             % (emission_wavelength * 1e3, na, ni, dc, dstage, dxi, dyi, dzi, theta * 180 / np.pi))


ax = plt.subplot(grid[:, 0])
plt.imshow(np.nanmax(imgs_unskew, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
if plot_extra:
    # plt.plot(centers_guess[:, 1], centers_guess[:, 2], 'gx')
    plt.plot(centers_fit_sequence[:, :, 1].ravel(), centers_fit_sequence[:, :, 2].ravel(), 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 2], 'mx')
plt.plot(centers_unique[:, 1], centers_unique[:, 2], 'rx')
plt.xlabel("Y (um)")
plt.ylabel("X (um)")
plt.title("XY")

ax = plt.subplot(grid[0, 1])
plt.imshow(np.nanmax(imgs_unskew, axis=1), vmin=vmin, vmax=vmax, origin="lower",
           extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
if plot_extra:
    # plt.plot(centers_guess[:, 2], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit_sequence[:, :, 2].ravel(), centers_fit_sequence[:, :, 0].ravel(), 'gx')
    plt.plot(centers_fit[:, 2], centers_fit[:, 0], 'mx')
plt.plot(centers_unique[:, 2], centers_unique[:, 0], 'rx')
plt.xlabel("X (um)")
plt.ylabel("Z (um)")
plt.title("XZ")

ax = plt.subplot(grid[1, 1])
plt.imshow(np.nanmax(imgs_unskew, axis=2), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
if plot_extra:
    # plt.plot(centers_guess[:, 1], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit_sequence[:, :, 1].ravel(), centers_fit_sequence[:, :, 0].ravel(), 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 0], 'mx')
plt.plot(centers_unique[:, 1], centers_unique[:, 0], 'rx')
plt.xlabel("Y (um)")
plt.ylabel("Z (um)")
plt.title("YZ")
