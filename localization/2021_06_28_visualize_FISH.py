import os
import time
import numpy as np
import pickle
import pycromanager
import tifffile
import napari
from napari_animation import AnimationWidget
import matplotlib.pyplot as plt
import itertools
import localize
import image_post_processing as pp

plot_centers = True
plot_centers_guess = True
plot_fit_filters = False

figsize = (16, 8)

root_dir = r"\\10.206.26.21\opm2\20210628\new area"
img_fname = os.path.join(root_dir, "bDNA_stiff_gel_human_lung_r0001_y0014_z0000_ch0001_1")
dc = 0.115

# load data
data_dir = os.path.join(root_dir, "2021_07_01_17;27;16_localization")
channels = [2, 3]
nchannels = len(channels)
rounds = list(range(1, 8))
nrounds = len(rounds)

data_fnames = ["localization_round=%d_ch=%d_tile=14_z=0_t=0.pkl" % (r, c) for r, c in itertools.product(rounds, channels)]
data_fnames = [os.path.join(data_dir, d) for d in data_fnames]

for ii in range(len(data_fnames)):
    if not os.path.exists(data_fnames[ii]):
        raise FileExistsError(data_fnames[ii])

# load image
dset = pycromanager.Dataset(img_fname)

tstart = time.perf_counter()
imgs_raw = []
for ii in range(len(dset.axes["z"])):
    imgs_raw.append(dset.read_image(channel=0, z=ii))
print("loaded images in %0.2fs" % (time.perf_counter() - tstart))
imgs_raw = np.flip(np.asarray(imgs_raw), axis=0)

# deskew image
tstart = time.perf_counter()
imgs = pp.deskew(imgs_raw, 30., 0.4, 0.115)
print("deskewed in %0.2fs" % (time.perf_counter() - tstart))

# get image coordinates
x, y, z = localize.get_coords(imgs.shape, dc, dc)
dx = x[0, 0, 1] - x[0, 0, 0]
dy = y[0, 1, 0] - y[0, 0, 0]
dz = z[1, 0, 0] - z[0, 0, 0]

# get mask
thresh = 300
mask = imgs > thresh

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

    centers_napari[ii] = centers[ii] / dc

    centers_guess = np.stack((ips[ii][:, 3], ips[ii][:, 2], ips[ii][:, 1]), axis=1)
    centers_guess_napari[ii] = centers_guess / dc

# plot max projections
cmap = plt.cm.get_cmap('gist_rainbow')
colors = cmap(np.arange(nrounds * nchannels) / nrounds / nchannels)
colors[:, -1] = 0.25

figh = plt.figure(figsize=(16, 8))
ymax_ind = imgs.shape[1] - 1

ax = plt.gca()
extent = [y.min() - 0.5 * dy, y[0, ymax_ind, 0] + 0.5 * dy, x.min() - 0.5 * dx, x.max() + 0.5 * dx]
ax.imshow(np.max(imgs[:, :ymax_ind, :], axis=0).transpose(), vmin=-100, vmax=2000, extent=extent, origin="lower", cmap="bone")
for ii in range(len(data_fnames)):
    to_plot = mask[tuple(centers_pixel[ii].transpose())]
    ax.plot(centers[ii][to_plot, 1], centers[ii][to_plot, 2], '.', color=colors[ii])
# ax.set_xlabel("x-position ($\mu m$)")
# ax.set_ylabel("y-position ($\mu m$)")
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([extent[0], extent[1]])

fname = os.path.join(data_dir, "FISH_maxproj_z.png")
figh.savefig(fname)


viewer = napari.Viewer(title=img_fname)
animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area='right')

# viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=img_fname)
viewer.add_image(imgs, scale=(1, 1, 1), colormap="gray_r", contrast_limits=[120, 2000], multiscale=False)

if plot_centers:
    for ii in range(len(centers_napari)):
        to_plot = mask[tuple(centers_pixel[ii].transpose())]
        viewer.add_points(centers_napari[ii][to_plot], size=3, face_color=colors[ii], opacity=0.9, name="round %d" % ii, n_dimensional=True)
