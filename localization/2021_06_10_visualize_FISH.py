"""
Plot data and localizations
"""
import os
import time

import numpy as np
import itertools
import scipy.ndimage as ndi
import skimage.segmentation
import skimage.feature
import skimage.filters
import pickle
import tifffile

# visualization
import napari
from napari_animation import AnimationWidget
import matplotlib.pyplot as plt
# custom library
import localize

plot_centers = True
plot_centers_guess = True
plot_fit_filters = False

figsize = (16, 8)
root_dir = r"\\10.206.26.21\opm2\20210610\output\fused_tiff"
data_dir = os.path.join(root_dir, "2021_06_27_11;08;01_localization")

channels = [1, 2]
nchannels = len(channels)
rounds = list(range(8))
nrounds = len(rounds)
img_fname = os.path.join(root_dir, "img_TL0_Ch0_Tile0.tif") # cell outlines


data_fnames = ["img_TL%d_Ch%d_Tile0_round=%d_ch=%d_tile=0_vol=0.pkl" % (r, c, r, c) for r, c in itertools.product(rounds, channels)]
data_fnames = [os.path.join(data_dir, d) for d in data_fnames]

data_fnames = [d for d in data_fnames if os.path.exists(d)]

# load image data and coordinates
imgs = tifffile.imread(img_fname)

dc = 0.065
dz = 0.25
x, y, z = localize.get_coords(imgs.shape, dc, dz)
dx = x[0, 0, 1] - x[0, 0, 0]
dy = y[0, 1, 0] - y[0, 0, 0]
dz = z[1, 0, 0] - z[0, 0, 0]

# filter image
tstart = time.perf_counter()
kernel = localize.get_filter_kernel((10*dz, 20*dc, 20*dc), dc, dz)
imgs_filtered = localize.filter_convolve(imgs, kernel, use_gpu=False)
tend = time.perf_counter()
print("filtering image took %0.2fs" % (tend - tstart))

# create cell mask
thresh = 120
mask = imgs_filtered > thresh

# a = imgs_filtered[100]
# t = skimage.filters.sobel(a, mask)
#
# # get local maxima
# amax = skimage.feature.peak_local_max(a * mask, footprint=np.ones((25, 25)))
# # convert maxima to mask
# max_mask = np.zeros(a.shape, dtype=bool)
# max_mask[tuple(amax.transpose())] = True
# markers, _ = ndi.label(max_mask)
# ls = skimage.segmentation.watershed(-a, markers, mask=mask)

# tstart = time.perf_counter()
# labels = skimage.segmentation.watershed(-imgs_filtered)
# tend = time.perf_counter()
# print("segmentation in %0.2fs" % (tend - tstart))


# load localization data
fps = [[]] * (nrounds * nchannels)
ips = [[]] * (nrounds * nchannels)
rois = [[]] * (nrounds * nchannels)
to_keep = [[]] * (nrounds * nchannels)
centers = [[]] * (nrounds * nchannels)
centers_pixel = [[]] * (nrounds * nchannels)
centers_napari = [[]] * (nrounds * nchannels)
centers_guess_napari = [[]] * (nrounds * nchannels)

for ii, d in enumerate(data_fnames):
    with open(d, "rb") as f:
        data = pickle.load(f)

    fps[ii] = data["fit_params"]
    ips[ii] = data["init_params"]
    rois[ii] = data["rois"]
    to_keep[ii] = data["to_keep"]

    # massage data into plottable format
    centers[ii] = np.stack((fps[ii][:, 3][to_keep[ii]], fps[ii][:, 2][to_keep[ii]], fps[ii][:, 1][to_keep[ii]]), axis=1)
    centers_pixel[ii] = np.round(centers[ii] / np.expand_dims(np.array([dz, dc, dc]), axis=0)).astype(np.int)

    centers_napari[ii] = centers[ii] / np.expand_dims(np.array([dc, dc, dc]), axis=0)

    centers_guess = np.stack((ips[ii][:, 3], ips[ii][:, 2], ips[ii][:, 1]), axis=1)
    centers_guess_napari[ii] = centers_guess / np.expand_dims(np.array([dc, dc, dc]), axis=0)

# plot max projections
cmap = plt.cm.get_cmap('gist_rainbow')
colors = cmap(np.arange(nrounds * nchannels) / nrounds / nchannels)
colors[:, -1] = 0.25

figh = plt.figure(figsize=(16, 8))
ax = plt.gca()
extent = [x.min() - 0.5 * dx, x.max() + 0.5 * dx, y.min() - 0.5 * dy, y.max() + 0.5 * dy]
ax.imshow(np.max(imgs, axis=0), vmin=110, vmax=180, extent=extent, origin="lower", cmap="bone")
for ii in range(len(data_fnames)):
    to_plot = mask[tuple(centers_pixel[ii].transpose())]
    ax.plot(centers[ii][to_plot, 2], centers[ii][to_plot, 1], '.', color=colors[ii])
# ax.set_xlabel("x-position ($\mu m$)")
# ax.set_ylabel("y-position ($\mu m$)")
ax.set_xticks([])
ax.set_yticks([])

fname = os.path.join(data_dir, "FISH_maxproj_z.png")
figh.savefig(fname)

figh = plt.figure(figsize=(16, 8))
ax = plt.gca()
extent = [x.min() - 0.5 * dx, x.max() + 0.5 * dx, z.min() - 0.5 * dz, z.max() + 0.5 * dz]
ax.imshow(np.max(imgs, axis=1), vmin=110, vmax=180, extent=extent, origin="lower", cmap="bone")
for ii in range(len(data_fnames)):
    to_plot = mask[tuple(centers_pixel[ii].transpose())]
    ax.plot(centers[ii][to_plot, 2], centers[ii][to_plot, 0], '.', color=colors[ii])
# ax.set_xlabel("x-position ($\mu m$)")
# ax.set_ylabel("z-position ($\mu m$)")
ax.set_xticks([])
ax.set_yticks([])

fname = os.path.join(data_dir, "FISH_maxproj_y.png")
figh.savefig(fname)

figh = plt.figure(figsize=(16, 8))
ax = plt.gca()
extent = [y.min() - 0.5 * dy, y.max() + 0.5 * dy, z.min() - 0.5 * dz, z.max() + 0.5 * dz]
ax.imshow(np.max(imgs, axis=2), vmin=110, vmax=180, extent=extent, origin="lower", cmap="bone")
for ii in range(len(data_fnames)):
    to_plot = mask[tuple(centers_pixel[ii].transpose())]
    ax.plot(centers[ii][to_plot, 1], centers[ii][to_plot, 0], '.', color=colors[ii])
# ax.set_xlabel("y-position ($\mu m$)")
# ax.set_ylabel("z-position ($\mu m$)")
ax.set_xticks([])
ax.set_yticks([])

fname = os.path.join(data_dir, "FISH_maxproj_x.png")
figh.savefig(fname)

# plot mask max projection
# figh = plt.figure()
# plt.imshow(np.max(mask, axis=0), extent=extent, origin="lower")


# plot with napari
# specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
viewer = napari.Viewer(title=img_fname)
animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area='right')

# viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=img_fname)
viewer.add_image(imgs, scale=(dz/dc, 1, 1), colormap="gray_r", contrast_limits=[120, 200], multiscale=False)

if plot_centers:
    for ii in range(len(centers_napari)):
        to_plot = mask[tuple(centers_pixel[ii].transpose())]
        viewer.add_points(centers_napari[ii][to_plot], size=3, face_color=colors[ii], opacity=0.9, name="round %d" % ii, n_dimensional=True)
