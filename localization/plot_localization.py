"""
Display results of OPM localization fit in matplotlib window with tools to display different ROI's
"""
import glob
import numpy as np
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import pickle
import os
import tifffile
import localize

plot_guesses = True
plot_tracks = True
index = 1
root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210309", "crowders-10x-50glyc")
# analysis_dir = os.path.join(root_dir, "2021_03_10_17;57;53_localization")
analysis_dir = glob.glob(os.path.join(root_dir, "2021_04*"))[-1]

# scan data
scan_data_dir = os.path.join(root_dir, "galvo_scan_params.pkl")
with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

# todo: all of these parameters should be read from metadata
dc = scan_data["pixel size"][0] / 1000
theta = scan_data["theta"][0] * np.pi/180
nstage = 25
ni1 = 256
ni2 = 1600
dstage = scan_data["scan step"][0] / 1000
nvols = 10000
x, y, z = localize.get_lab_coords(ni2, ni1, dc, theta, dstage * np.arange(nstage))

# initial roi values to plot
# z, y, x # these are not literal sizes, rather the computed ROI contains this cube
sizes_roi = (3., 3., 3.)
# sizes_roi = (1e10, 1e10, 1e10)
# (y, x)
centers_roi = (18., 100.)
vol_index = 0

# track data
track_data_file = os.path.join(analysis_dir, "tracks.pkl")
if os.path.exists(track_data_file):
    with open(track_data_file, "rb") as f:
        track_data_pd = pickle.load(f)

    pns_unq, inv = np.unique(track_data_pd["particle"], return_inverse=True)
    track_data_pd["particles unique"] = inv
    frames = np.unique(track_data_pd["frame"])

    track_data = np.zeros((len(pns_unq), len(frames), 3)) * np.nan
    f = track_data_pd["frame"].values
    p = track_data_pd["particles unique"].values
    track_data[p, f, 2] = track_data_pd["xum"].values
    track_data[p, f, 1] = track_data_pd["yum"].values
    track_data[p, f, 0] = track_data_pd["zum"].values

else:
    plot_tracks = False
    track_data = None

# create plot
figh = plt.figure(figsize=(14, 8), facecolor="gray")

xy_vstart = 0.3
xy_vsize = 0.6
xy_hstart = 0.3
plot_sep = 0.02

# [left, bottom, width, height]
ax_xy = plt.axes([0, 0, 0, 0])
ax_xz = plt.axes([0, 0.1, 0, 0])
ax_yz = plt.axes([0, 0.2, 0, 0])

# sliders and buttons
slider_axes = plt.axes([0.3, 0, 0.4, 0.05], facecolor='lightgoldenrodyellow')
sliders = matplotlib.widgets.Slider(slider_axes, 'index', 0, 100, valinit=0, valstep=1, closedmin=True, closedmax=False)

ax_b1 = plt.axes([0.05, 0, 0.1, 0.05])
ax_b2 = plt.axes([0.85, 0, 0.1, 0.05])
button1 = Button(ax_b1, '<', color='w', hovercolor='b')
button2 = Button(ax_b2, '>', color='w', hovercolor='b')

ax_b3 = plt.axes([0.7, 0.65, 0.05, 0.05])
button3 = Button(ax_b3, "update", color="w", hovercolor="b")


# textboxes
cx_box = plt.axes([0.7, 0.85, 0.05, 0.05])
cx_slider = matplotlib.widgets.Slider(cx_box, "cx", x.min(), x.max(), valinit=centers_roi[1], valstep=1, closedmin=True, closedmax=True)

cy_box = plt.axes([0.8, 0.85, 0.05, 0.05])
cy_slider = matplotlib.widgets.Slider(cy_box, "cy", y.min(), y.max(), valinit=centers_roi[0], valstep=1, closedmin=True, closedmax=True)


wx_box = plt.axes([0.7, 0.75, 0.05, 0.05])
wx_slider = matplotlib.widgets.Slider(wx_box, "wx", 0, x.max() - x.min(), valinit=sizes_roi[2], valstep=1, closedmin=True, closedmax=True)

wy_box = plt.axes([0.8, 0.75, 0.05, 0.05])
wy_slider = matplotlib.widgets.Slider(wy_box, "wy", 0, y.max() - y.min(), valinit=sizes_roi[1], valstep=1, closedmin=True, closedmax=True)

wz_box = plt.axes([0.9, 0.75, 0.05, 0.05])
wz_slider = matplotlib.widgets.Slider(wz_box, "wz", 0, z.max() - z.min(), valinit=sizes_roi[0], valstep=1, closedmin=True, closedmax=True)
# wx_text_box = matplotlib.widgets.TextBox(wx_box, 'wx', initial=str(sizes_roi[2]))
# wy_text_box = matplotlib.widgets.TextBox(wy_box, 'wy', initial=str(sizes_roi[1]))
# wz_text_box = matplotlib.widgets.TextBox(wz_box, 'wz', initial=str(sizes_roi[0]))


# colorbar axes
ax_cb_guess = plt.axes([0.8, 0.2, 0.05, 0.6])
ax_cb_guess.set_frame_on(False)
ax_cb_guess.set_xticks([])
ax_cb_guess.set_xticklabels([])
ax_cb_guess.set_yticks([])
ax_cb_guess.set_yticklabels([])

ax_cb_fit = plt.axes([0.9, 0.2, 0.05, 0.6])
ax_cb_fit.set_frame_on(False)
ax_cb_fit.set_xticks([])
ax_cb_fit.set_xticklabels([])
ax_cb_fit.set_yticks([])
ax_cb_fit.set_yticklabels([])

cmap = plt.get_cmap("Reds")
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=z.max()), cmap=cmap), ax=ax_cb_fit)
cbar.ax.set_ylabel("z ($\mu$m)")

cmap_guess = plt.get_cmap("Greens")
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=z.max()), cmap=cmap_guess), ax=ax_cb_guess)

initialized = False

def load_volume(sizes, centers, vol_index):
    roi_sizes = localize.get_roi_size(sizes, theta, dc, dstage, ensure_odd=False)

    zc, yc, xc = centers
    ind = np.unravel_index(np.argmin((xc - x)**2 + (yc - y)**2 + (zc - z)**2), (x+y+z).shape)

    roi = localize.get_centered_roi(ind, roi_sizes, min_vals=(0, 0, 0), max_vals=(x+y+z).shape)


    files = [os.path.join(root_dir, "Image2_%d.tif") % (ii + 1) for ii in range(nstage * vol_index, nstage * (vol_index + 1))]
    # restrict to only ones we need to load
    files = files[roi[0]:roi[1]]
    imgs = []
    for f in files:
        imgs.append(tifffile.imread(f))
    imgs = np.asarray(imgs)
    imgs = np.flip(imgs, axis=1)

    imgs_roi = imgs[:, roi[2]:roi[3], roi[4]:roi[5]]
    coords = (z[:, roi[2]:roi[3], :], y[roi[0]:roi[1], roi[2]:roi[3], :], x[:, :, roi[4]:roi[5]])

    xroi, yroi, zroi, img_deskew_roi = localize.interp_opm_data(imgs_roi, dc, dstage, theta, mode="ortho-interp")

    xroi += x[0, 0, roi[4]]
    yroi += y[roi[0], roi[2], 0]
    zroi += z[0, roi[2], 0]
    coords_deskew = (zroi, yroi, xroi)

    return coords, coords_deskew, img_deskew_roi

# function called when sliders are moved on plot
def update(val):
    # update indices / centers
    index = sliders.val
    cx = float(cx_slider.val)
    cy = float(cy_slider.val)
    cz, _, _ = localize.find_trapezoid_cz(cy, (z, y, x))
    wx = float(wx_slider.val)
    wy = float(wy_slider.val)
    wz = float(wz_slider.val)

    # load image
    coords, coords_deskew, img = load_volume([wz, wy, wx], [cz, cy, cx], index)
    zroi, yroi, xroi = coords_deskew
    zroi_opm, yroi_opm, xroi_opm = coords

    dx = xroi[1] - xroi[0]
    dy = yroi[1] - yroi[0]
    dz = zroi[1] - zroi[0]

    extent_xy = [xroi[0] - 0.5 * dx, xroi[-1] + 0.5 * dx, yroi[0] - 0.5 * dy, yroi[-1] + 0.5 * dy]
    extent_xz = [xroi[0] - 0.5 * dx, xroi[-1] + 0.5 * dx, zroi[0] - 0.5 * dz, zroi[-1] + 0.5 * dz]
    # pad z-image by n-pixels
    nz_pad = 6
    extent_yz = [zroi[0] - nz_pad * dz - 0.5 * dz, zroi[-1] + nz_pad * dz + 0.5 * dz, yroi[0] - 0.5 * dy,
                 yroi[-1] + 0.5 * dy]

    img_xy = np.nanmax(img, axis=0)
    img_yz = np.pad(np.transpose(np.nanmax(img, axis=2)), ((0, 0), (nz_pad, nz_pad)), mode="constant", constant_values=np.nan)
    img_xz = np.pad(np.nanmax(img, axis=1), ((nz_pad, nz_pad), (0, 0)), mode="constant", constant_values=np.nan)

    # get figure aspect ratio
    fig_aspect_ratio = figh.get_size_inches()[0] / figh.get_size_inches()[1]

    # set aspect ratios for axes
    # aspect ratio = horizontal / vertical size
    aspect_ratio_xy = (extent_xy[1] - extent_xy[0]) / (extent_xy[3] - extent_xy[2]) / fig_aspect_ratio
    xy_hsize = xy_vsize * aspect_ratio_xy

    xz_hsize = xy_hsize
    aspect_ratio_xz = (extent_xz[1] - extent_xz[0]) / (extent_xz[3] - extent_xz[2]) / fig_aspect_ratio
    xz_vsize = xy_hsize / aspect_ratio_xz

    yz_vsize = xy_vsize
    aspect_ratio_yz = (extent_yz[1] - extent_yz[0]) / (extent_yz[3] - extent_yz[2]) / fig_aspect_ratio
    yz_hsize = yz_vsize * aspect_ratio_yz

    ax_xy.clear()
    ax_xy.set_facecolor("grey")
    ax_xy.set_position([xy_hstart, xy_vstart, xy_hsize, xy_vsize])
    ax_xy.set_xticklabels([])
    ax_xy.set_yticklabels([])
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])

    ax_xz.clear()
    ax_xz.set_facecolor("grey")
    ax_xz.set_position([xy_hstart, xy_vstart - xz_vsize - plot_sep, xz_hsize, xz_vsize])
    ax_xz.set_xlabel("x ($\mu$m)")
    ax_xz.set_yticklabels([])
    ax_xz.set_yticks([])

    ax_yz.clear()
    ax_yz.set_facecolor("grey")
    ax_yz.set_position([xy_hstart - yz_hsize - plot_sep / fig_aspect_ratio, xy_vstart, yz_hsize, yz_vsize])
    ax_yz.set_xlabel("z ($\mu$m)")
    ax_yz.set_ylabel("y ($\mu$m)")

    # load localization data
    loc_data_dir = os.path.join(analysis_dir, "localization_results_vol_%d.pkl" % index)
    with open(loc_data_dir, "rb") as f:
        loc_data = pickle.load(f)

    centers = loc_data["centers"]
    # centers_to_plot = localize.point_in_trapezoid(centers, xroi_opm, yroi_opm, zroi_opm)
    centers_to_plot = np.logical_and.reduce((centers[:, 0] <= zroi_opm.max() + dz * nz_pad, centers[:, 0] >= zroi_opm.min() - dz * nz_pad,
                                             centers[:, 1] <= yroi_opm.max(), centers[:, 1] >= yroi_opm.min(),
                                             centers[:, 2] <= xroi_opm.max(), centers[:, 2] >= xroi_opm.min()))
    centers = centers[centers_to_plot]

    plt.suptitle("%s\n%s\n%d localizations in ROI" % (root_dir, os.path.split(analysis_dir)[-1], len(centers)))

    # plot images
    vmin = np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 1)
    vmax = np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 99.9)

    ax_xy.imshow(img_xy, extent=extent_xy, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, interpolation='none')

    ax_yz.imshow(img_yz, extent=extent_yz, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')

    ax_xz.imshow(img_xz, extent=extent_xz, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')

    # plot guesses before fitting
    if plot_guesses:
        centers_guess = loc_data["centers_guess"]
        # centers_guess_in_trap = localize.point_in_trapezoid(centers_guess, xroi_opm, yroi_opm, zroi_opm)
        centers_guess_to_plot = np.logical_and.reduce((centers_guess[:, 0] <= zroi_opm.max() + dz * nz_pad, centers_guess[:, 0] >= zroi_opm.min() - dz * nz_pad,
                                                       centers_guess[:, 1] <= yroi_opm.max(), centers_guess[:, 1] >= yroi_opm.min(),
                                                       centers_guess[:, 2] <= xroi_opm.max(), centers_guess[:, 2] >= xroi_opm.min()))
        centers_guess = centers_guess[centers_guess_to_plot]

        ax_xy.scatter(centers_guess[:, 2], centers_guess[:, 1], facecolors=cmap_guess(centers_guess[:, 0] / z.max()),
                      marker='+', label="guesses")
        ax_yz.scatter(centers_guess[:, 0], centers_guess[:, 1], facecolors=cmap_guess(1.), marker='+')
        ax_xz.scatter(centers_guess[:, 2], centers_guess[:, 0], facecolors=cmap_guess(1.), marker='+')

    # plot fit points with z-position displayed via colorbar, and display number of point
    ax_xy.scatter(centers[:, 2], centers[:, 1], facecolors=cmap(centers[:, 0] / z.max()), marker='x', label="fits")
    ax_yz.scatter(centers[:, 0], centers[:, 1], facecolors=cmap(1.), marker='x')
    ax_xz.scatter(centers[:, 2], centers[:, 0], facecolors=cmap(1.), marker='x')
    for ii in range(len(centers)):
        ax_xy.annotate("%d" % ii, xy=(centers[ii, 2], centers[ii, 1]), color=[1., 1., 0.2])
        ax_yz.annotate("%d" % ii, xy=(centers[ii, 0], centers[ii, 1]), color=[1., 1., 0.2])
        ax_xz.annotate("%d" % ii, xy=(centers[ii, 2], centers[ii, 0]), color=[1., 1., 0.2])


    if plot_tracks:
        # remove track if does not occur at this time point
        cols_not_nan = np.logical_not(np.all(np.isnan(track_data[:, :index + 1, 0]), axis=1))

        # remove track if not in ROI
        in_trap = localize.point_in_trapezoid(track_data[:, index], xroi_opm, yroi_opm, zroi_opm)
        cols_to_plot = np.logical_and(in_trap, cols_not_nan)

        dat = track_data[cols_to_plot, :index + 1]

        ax_xy.plot(dat[..., 2].transpose(), dat[..., 1].transpose(), '.-')
        ax_yz.plot(dat[..., 0].transpose(), dat[..., 1].transpose(), '.-')
        ax_xz.plot(dat[..., 2].transpose(), dat[..., 0].transpose(), '.-')

    figh.canvas.draw_idle()

def backward(val):
    pos = sliders.val
    if pos > 0:
        new_pos = pos - 1
    else:
        new_pos = pos
    sliders.set_val(new_pos)

def forward(val):
    pos = sliders.val
    sliders.set_val(pos + 1)

figh.canvas.mpl_connect("resize_event", update)
sliders.on_changed(update)
button1.on_clicked(backward)
button2.on_clicked(forward)
button3.on_clicked(update)
# call once to ensure displays something
sliders.set_val(0)

plt.show()