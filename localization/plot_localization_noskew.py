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

# round = 2
#
# img_fnames = [r"\\10.206.26.21\opm2\20210503a\r1_atto565_r2_alexa647_1\r1_atto565_r2_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r1_atto565_r2_alexa647_1\r1_atto565_r2_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r3_atto565_r4_alexa647_1\r3_atto565_r4_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r3_atto565_r4_alexa647_1\r3_atto565_r4_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r5_atto565_r6_alexa647_1\r5_atto565_r6_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r5_atto565_r6_alexa647_1\r5_atto565_r6_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r7_atto565_r8_alexa647_1\r7_atto565_r8_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r7_atto565_r8_alexa647_1\r7_atto565_r8_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r9_atto565_r10_alexa647_1\r9_atto565_r10_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r9_atto565_r10_alexa647_1\r9_atto565_r10_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r11_atto565_r12_alexa647_1\r11_atto565_r12_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r11_atto565_r12_alexa647_1\r11_atto565_r12_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r13_atto565_r14_alexa647_1\r13_atto565_r14_alexa647_1_MMStack_Default.ome.tif",
#               r"\\10.206.26.21\opm2\20210503a\r13_atto565_r14_alexa647_1\r13_atto565_r14_alexa647_1_MMStack_Default.ome.tif"
#               ]
# data_root = os.path.join(r"\\10.206.26.21\opm2\20210503a", "2021_05_10_13;53;02_localization")
# data_fnames = ["r1_atto565_r2_alexa647_1_ch=1_vol=0.pkl",
#                "r1_atto565_r2_alexa647_1_ch=2_vol=0.pkl",
#                "r3_atto565_r4_alexa647_1_ch=1_vol=0.pkl",
#                "r3_atto565_r4_alexa647_1_ch=2_vol=0.pkl",
#                "r5_atto565_r6_alexa647_1_ch=1_vol=0.pkl",
#                "r5_atto565_r6_alexa647_1_ch=2_vol=0.pkl",
#                "r7_atto565_r8_alexa647_1_ch=1_vol=0.pkl",
#                "r7_atto565_r8_alexa647_1_ch=2_vol=0.pkl",
#                "r9_atto565_r10_alexa647_1_ch=1_vol=0.pkl",
#                "r9_atto565_r10_alexa647_1_ch=2_vol=0.pkl",
#                "r11_atto565_r12_alexa647_1_ch=1_vol=0.pkl",
#                "r11_atto565_r12_alexa647_1_ch=2_vol=0.pkl",
#                "r13_atto565_r14_alexa647_1_ch=1_vol=0.pkl",
#                "r13_atto565_r14_alexa647_1_ch=2_vol=0.pkl"]
# data_fnames = [os.path.join(data_root, fn) for fn in data_fnames]
# channels = [1, 2] * 7
#
#
# img_fname = img_fnames[round - 1]
# data_fname = data_fnames[round - 1]
# channel = channels[round - 1]

# img_fname = r"\\10.206.26.21\opm2\20210507cells\tiffs\xy004\crop_decon_split\registered\round001_alexa647_FLNB.tif"
# data_fname = r"C:\Users\ptbrown2\Desktop\2021_06_03_13;54;13_localization\localization.pkl"
# channel = 0
# imgs = np.squeeze(tifffile.imread(img_fname)[:, channel])

round = 0
channel = 1
tile = 0
# img_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\img_TL0_Ch0_Tile0.tif" # cell outlines
img_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\img_TL%d_Ch%d_Tile%d.tif" % (round, channel, tile)
# data_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\2021_06_27_11;08;01_localization\img_TL0_Ch1_Tile0_round=0_ch=1_tile=0_vol=0.pkl"
data_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\2021_06_27_14;02;39_localization\img_TL%d_Ch%d_Tile%d_round=%d_ch=%d_tile=%d_vol=0.pkl" % \
             (round, channel, tile, round, channel, tile)
imgs = tifffile.imread(img_fname)

# get coordinates
dc = 0.065
dz = 0.25
x, y, z = localize.get_coords(imgs.shape, dc, dz)

# load localization data
with open(data_fname, "rb") as f:
    data = pickle.load(f)

fps = data["fit_params"]
ips = data["init_params"]
rois = data["rois"]
to_keep = data["to_keep"]

# massage data into plottable format
centers = np.stack((fps[:, 3][to_keep], fps[:, 2][to_keep], fps[:, 1][to_keep]), axis=1)
centers_napari = centers / np.expand_dims(np.array([dc, dc, dc]), axis=0)

centers_guess = np.stack((ips[:, 3], ips[:, 2], ips[:, 1]), axis=1)
centers_guess_napari = centers_guess / np.expand_dims(np.array([dc, dc, dc]), axis=0)

# plot some fits
if False:
    num_plotted = 0
    ind = 0
    while num_plotted < 40:
        if to_keep[ind]:
            figa = localize.plot_roi(fps[ind], rois[ind], imgs, x, y, z,
                                                  init_params=ips[ind],
                                                  figsize=(16, 8), same_color_scale=True)
            num_plotted += 1
        ind += 1


# maximum intensity projection
figh2 = plt.figure(figsize=figsize)
plt.suptitle("Maximum intensity projection")
grid = plt.GridSpec(1, 2)

ax = plt.subplot(grid[0, 0])
maxproj = np.nanmax(imgs, axis=0)
vmin = np.percentile(maxproj, 0.1)
vmax = np.percentile(maxproj, 99.5)
plt.imshow(maxproj, vmin=vmin, vmax=vmax, origin="lower",
           extent=[x[0, 0, 0] - 0.5 * dc, x[0, 0, -1] + 0.5 * dc, y[0, 0, 0] - 0.5 * dc, y[0, -1, 0] + 0.5 * dc])
plt.plot(centers[:, 2], centers[:, 1], 'rx')

ax = plt.subplot(grid[0, 1])
plt.imshow(maxproj, vmin=vmin, vmax=vmax, origin="lower",
           extent=[x[0, 0, 0] - 0.5 * dc, x[0, 0, -1] + 0.5 * dc, y[0, 0, 0] - 0.5 * dc, y[0, -1, 0] + 0.5 * dc])
plt.plot(centers_guess[:, 2], centers_guess[:, 1], 'gx')

# fit statistics
try:
    figh3 = plt.figure(figsize=figsize)
    plt.suptitle("Fit statistics\nmedians: amp = %0.3f, $\sigma_{xy}$ = %0.3f, $\sigma_z$ = %0.3f, bg = %0.3f" %
                 (np.median(fps[:, 0][to_keep]), np.median(fps[:, 4][to_keep]), np.median(fps[:, 5][to_keep]),
                  np.median(fps[:, 6][to_keep])))
    grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)

    ax = plt.subplot(grid[0, 0])
    ax.set_title("amp vs. $\sigma_{xy}$")
    plt.plot(fps[:, 0][to_keep], fps[:, 4][to_keep], '.')
    plt.xlabel("Amp")
    plt.ylabel("$\sigma_{xy}$ (um)")
    plt.xlim([0, np.percentile(fps[:, 0][to_keep], 99) * 1.2])
    plt.ylim([-0.01, np.percentile(fps[:, 4][to_keep], 99) * 1.2])

    ax = plt.subplot(grid[0, 1])
    ax.set_title("amp vs. $\sigma_z$")
    plt.plot(fps[:, 0][to_keep], fps[:, 5][to_keep], '.')
    plt.xlabel("Amp")
    plt.ylabel("$\sigma_z$ (um)")
    plt.xlim([0, np.percentile(fps[:, 0][to_keep], 99) * 1.2])
    plt.ylim([0, np.percentile(fps[:, 5][to_keep], 99) * 1.2])

    ax = plt.subplot(grid[1, 0])
    ax.set_title("$\sigma_{xy}$ vs. $\sigma_z$")
    plt.plot(fps[:, 4][to_keep], fps[:, 5][to_keep], '.')
    plt.xlabel("$\sigma_{xy} (um)$")
    plt.ylabel("$\sigma_z$ (um)")
    plt.xlim([0, np.percentile(fps[:, 4][to_keep], 99) * 1.2])
    plt.ylim([0, np.percentile(fps[:, 5][to_keep], 99) * 1.2])

    ax = plt.subplot(grid[1, 1])
    ax.set_title("amp vs. bg")
    plt.plot(fps[:, 0][to_keep], fps[:, 6][to_keep], '.')
    plt.xlabel("Amp")
    plt.ylabel("background")
    plt.xlim([0, np.percentile(fps[:, 0][to_keep], 99) * 1.2])
    plt.ylim([np.percentile(fps[:, 6][to_keep], 1) - 5, np.percentile(fps[:, 6][to_keep], 99) + 5])
except:
    pass

# plot with napari
# with napari.gui_qt():
# specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
viewer = napari.Viewer(title=img_fname)
animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area='right')

# viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=img_fname)
viewer.add_image(imgs, scale=(dz/dc, 1, 1), colormap="bone", contrast_limits=[0, 750], multiscale=False)

if plot_centers:
    viewer.add_points(centers_napari, size=2, face_color="red", opacity=0.75, name="fits", n_dimensional=True)
if plot_centers_guess:
    viewer.add_points(centers_guess_napari, size=2, face_color="green", opacity=0.5, name="guesses", n_dimensional=True)

if plot_fit_filters:
    conditions = data["conditions"].transpose()
    condition_names = data["conditions_names"]
    colors = ["purple", "blue", "green", "yellow", "orange"] * int(np.ceil(len(conditions) / 5))
    for c, cn, col in zip(conditions, condition_names, colors):
        ct = centers_guess[np.logical_not(c)] / np.expand_dims(np.array([dz, dc, dc]), axis=0)
        viewer.add_points(ct, size=2, face_color=col, opacity=0.5, name="not %s" % cn.replace("_", " "), n_dimensional=True)
