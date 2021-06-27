"""
Plot data and localizations
"""
import os
import numpy as np
import pickle
import tifffile
import napari
from napari_animation import AnimationWidget
import localize
import matplotlib.pyplot as plt

plot_centers = True
plot_centers_guess = True
plot_fit_filters = False

figsize = (16, 8)

rounds = list(range(8))
nrounds = len(rounds)
img_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\img_TL0_Ch0_Tile0.tif" # cell outlines

data_fnames = [r"\\10.206.26.21\opm2\20210610\output\fused_tiff\2021_06_27_11;08;01_localization\img_TL%d_Ch1_Tile0_round=%d_ch=1_tile=0_vol=0.pkl" % (r, r)
               for r in rounds]

data_fnames = [d for d in data_fnames if os.path.exists(d)]

# load image data
imgs = tifffile.imread(img_fname)

# get coordinates
dc = 0.065
dz = 0.25
x, y, z = localize.get_coords(imgs.shape, dc, dz)

# load localization data
fps = [[]] * nrounds
ips = [[]] * nrounds
rois = [[]] * nrounds
to_keep = [[]] * nrounds
centers_napari = [[]] * nrounds
centers_guess_napari = [[]] * nrounds

for ii, d in enumerate(data_fnames):
    with open(d, "rb") as f:
        data = pickle.load(f)

    fps[ii] = data["fit_params"]
    ips[ii] = data["init_params"]
    rois[ii] = data["rois"]
    to_keep[ii] = data["to_keep"]

    # massage data into plottable format
    centers = np.stack((fps[ii][:, 3][to_keep[ii]], fps[ii][:, 2][to_keep[ii]], fps[ii][:, 1][to_keep[ii]]), axis=1)
    centers_napari[ii] = centers / np.expand_dims(np.array([dc, dc, dc]), axis=0)

    centers_guess = np.stack((ips[ii][:, 3], ips[ii][:, 2], ips[ii][:, 1]), axis=1)
    centers_guess_napari[ii] = centers_guess / np.expand_dims(np.array([dc, dc, dc]), axis=0)


# plot with napari
# specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
viewer = napari.Viewer(title=img_fname)
animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area='right')

# viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=img_fname)
viewer.add_image(imgs, scale=(dz/dc, 1, 1), colormap="bone", contrast_limits=[0, 750], multiscale=False)

# plot focus angle shift
cmap = plt.cm.get_cmap('gist_rainbow')
colors = cmap(np.array(rounds) / nrounds)

if plot_centers:
    for ii in range(len(centers_napari)):
        viewer.add_points(centers_napari[ii], size=2, face_color=colors[ii], opacity=0.75, name="round %d" % ii, n_dimensional=True)

