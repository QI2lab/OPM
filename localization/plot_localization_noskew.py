import os
import numpy as np
import pickle
import tifffile
import napari
import localize
import matplotlib.pyplot as plt

plot_centers = True
plot_centers_guess = True

round = 2

img_fnames = [r"\\10.206.26.21\opm2\20210503a\r1_atto565_r2_alexa647_1\r1_atto565_r2_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r1_atto565_r2_alexa647_1\r1_atto565_r2_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r3_atto565_r4_alexa647_1\r3_atto565_r4_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r3_atto565_r4_alexa647_1\r3_atto565_r4_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r5_atto565_r6_alexa647_1\r5_atto565_r6_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r5_atto565_r6_alexa647_1\r5_atto565_r6_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r7_atto565_r8_alexa647_1\r7_atto565_r8_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r7_atto565_r8_alexa647_1\r7_atto565_r8_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r9_atto565_r10_alexa647_1\r9_atto565_r10_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r9_atto565_r10_alexa647_1\r9_atto565_r10_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r11_atto565_r12_alexa647_1\r11_atto565_r12_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r11_atto565_r12_alexa647_1\r11_atto565_r12_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r13_atto565_r14_alexa647_1\r13_atto565_r14_alexa647_1_MMStack_Default.ome.tif",
              r"\\10.206.26.21\opm2\20210503a\r13_atto565_r14_alexa647_1\r13_atto565_r14_alexa647_1_MMStack_Default.ome.tif"
              ]
data_root = os.path.join(r"\\10.206.26.21\opm2\20210503a", "2021_05_10_13;53;02_localization")
data_fnames = ["r1_atto565_r2_alexa647_1_ch=1_vol=0.pkl",
               "r1_atto565_r2_alexa647_1_ch=2_vol=0.pkl",
               "r3_atto565_r4_alexa647_1_ch=1_vol=0.pkl",
               "r3_atto565_r4_alexa647_1_ch=2_vol=0.pkl",
               "r5_atto565_r6_alexa647_1_ch=1_vol=0.pkl",
               "r5_atto565_r6_alexa647_1_ch=2_vol=0.pkl",
               "r7_atto565_r8_alexa647_1_ch=1_vol=0.pkl",
               "r7_atto565_r8_alexa647_1_ch=2_vol=0.pkl",
               "r9_atto565_r10_alexa647_1_ch=1_vol=0.pkl",
               "r9_atto565_r10_alexa647_1_ch=2_vol=0.pkl",
               "r11_atto565_r12_alexa647_1_ch=1_vol=0.pkl",
               "r11_atto565_r12_alexa647_1_ch=2_vol=0.pkl",
               "r13_atto565_r14_alexa647_1_ch=1_vol=0.pkl",
               "r13_atto565_r14_alexa647_1_ch=2_vol=0.pkl"]
data_fnames = [os.path.join(data_root, fn) for fn in data_fnames]
channels = [1, 2] * 7


img_fname = img_fnames[round - 1]
data_fname = data_fnames[round - 1]
channel = channels[round - 1]

dc = 0.065
dz = 0.25

imgs = np.squeeze(tifffile.imread(img_fname)[:, channel])

with open(data_fname, "rb") as f:
    data = pickle.load(f)

centers = data["centers"]
centers_pix = centers / np.expand_dims(np.array([dz, dc, dc]), axis=0)

centers_guess = data["centers_guess"]
centers_guess_pix = centers_guess / np.expand_dims(np.array([dz, dc, dc]), axis=0)

x, y, z = localize.get_coords(imgs.shape, dc, dz)
rois = data["rois"].astype(np.int)
fit_params = data["fit_params"]
fit_params_all = data["fit_params_all"]
to_use = np.logical_and.reduce(data["conditions"], axis=1)

# inds = [10, 1022, 1088]
# inds = [1, 999, 1239]
inds = [30, 1206, 1310]
dists = ((centers_guess_pix[:, 0] - inds[0])**2 + (centers_guess_pix[:, 1] - inds[1])**2 + (centers_guess_pix[:, 2] - inds[2])**2)
# dists[np.logical_not(to_use)] = np.nan
ind = np.nanargmin(dists)
print(data["init_params"][ind])
figh1 = localize.plot_roi(data["fit_params_all"][ind], data["rois_all"][ind], imgs, x, y, z, center_guess=centers_guess[ind], same_color_scale=False)


figh2 = plt.figure(figsize=(16, 8))
grid = plt.GridSpec(1, 2)

ax = plt.subplot(grid[0, 0])
maxproj = np.nanmax(imgs, axis=0)
vmin = np.percentile(maxproj, 0.1)
vmax = np.percentile(maxproj, 99.9)
plt.imshow(maxproj, vmin=vmin, vmax=vmax, origin="lower",
           extent=[x[0, 0, 0] - 0.5 * dc, x[0, 0, -1] + 0.5 * dc, y[0, 0, 0] - 0.5 * dc, y[0, -1, 0] + 0.5 * dc])
plt.plot(centers[:, 2], centers[:, 1], 'rx')

ax = plt.subplot(grid[0, 1])
plt.imshow(maxproj, vmin=vmin, vmax=vmax, origin="lower",
           extent=[x[0, 0, 0] - 0.5 * dc, x[0, 0, -1] + 0.5 * dc, y[0, 0, 0] - 0.5 * dc, y[0, -1, 0] + 0.5 * dc])
plt.plot(centers_guess[:, 2], centers_guess[:, 1], 'gx')

try:
    figh3 = plt.figure()
    plt.suptitle("Fit statistics")
    grid = plt.GridSpec(2, 2)

    ax = plt.subplot(grid[0, 0])
    ax.set_title("amp vs. $\sigma_{xy}$")
    plt.plot(fit_params[:, 0], fit_params[:, 4], '.')
    plt.xlabel("Amp")
    plt.ylabel("$\sigma_{xy}$ (um)")
    plt.xlim([0, np.percentile(fit_params[:, 0], 99) * 1.2])

    ax = plt.subplot(grid[0, 1])
    ax.set_title("amp vs. $\sigma_z$")
    plt.plot(fit_params[:, 0], fit_params[:, 5], '.')
    plt.xlabel("Amp")
    plt.ylabel("$\sigma_z$ (um)")
    plt.xlim([0, np.percentile(fit_params[:, 0], 99) * 1.2])
    plt.ylim([0, np.percentile(fit_params[:, 5], 99) * 1.2])

    ax = plt.subplot(grid[1, 0])
    ax.set_title("$\sigma_{xy}$ vs. $\sigma_z$")
    plt.plot(fit_params[:, 4], fit_params[:, 5], '.')
    plt.xlabel("$\sigma_{xy} (um)$")
    plt.ylabel("$\sigma_z$ (um)")
    plt.xlim([0, np.percentile(fit_params[:, 4], 99) * 1.2])
    plt.ylim([0, np.percentile(fit_params[:, 5], 99) * 1.2])

    ax = plt.subplot(grid[1, 1])
    ax.set_title("amp vs. bg")
    plt.plot(fit_params[:, 0], fit_params[:, 6], '.')
    plt.xlabel("Amp")
    plt.ylabel("background")
    plt.xlim([0, np.percentile(fit_params[:, 0], 99) * 1.2])
    plt.ylim([np.min([np.percentile(fit_params[:, 6], 1), 0]), np.percentile(fit_params[:, 6], 99) * 1.2])
except:
    pass

# plot with napari
# with napari.gui_qt():
# specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
viewer = napari.view_image(imgs, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=img_fname)

if plot_centers:
    viewer.add_points(centers_pix, size=2, face_color="red", opacity=0.75, name="fits", n_dimensional=True)
if plot_centers_guess:
    viewer.add_points(centers_guess_pix, size=2, face_color="green", opacity=0.5, name="guesses", n_dimensional=True)

    conditions = data["conditions"].transpose()
    condition_names = data["conditions_names"]
    colors = ["purple", "blue", "green", "yellow", "orange"] * int(np.ceil(len(conditions) / 5))
    for c, cn, col in zip(conditions, condition_names, colors):
        ct = centers_guess[np.logical_not(c)] / np.expand_dims(np.array([dz, dc, dc]), axis=0)
        viewer.add_points(ct, size=2, face_color=col, opacity=0.5, name="not %s" % cn.replace("_", " "), n_dimensional=True)
