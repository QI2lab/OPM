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
import localize
import load_dataset

plot_guesses = False
plot_tracks = True
annotate_points = False
auto_track_cz = False

# root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210309", "crowders-10x-50glyc")
# analysis_dir = os.path.join(root_dir, "2021_03_10_17;57;53_localization")
# analysis_dir = glob.glob(os.path.join(root_dir, "2021_04*"))[-1]
# analysis_dir = os.path.join(root_dir, "test_localization")
# mode = "hcimage"

# root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210408n", "glycerol60x_1", "Full resolution")
# analysis_dir = glob.glob(os.path.join(root_dir, "..", "2021_04*"))[-1]
# mode = "ndtiff"

# root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210408m", "glycerol50x_1", "Full resolution")
# analysis_dir = glob.glob(os.path.join(root_dir, "..", "2021_04*"))[-1]
# mode = "ndtiff"

root_dir = os.path.join(r"\\10.206.26.21", "opm2", "20210408n", "glycerol60x_1", "Full resolution")
analysis_dir = glob.glob(os.path.join(root_dir, "..", "2021_04*"))[-1]
mode = "ndtiff"

# scan data
if mode == "hcimage":

    scan_data_fname = os.path.join(root_dir, "galvo_scan_params.pkl")
    with open(scan_data_fname, "rb") as f:
        scan_data = pickle.load(f)

    dc = scan_data["pixel size"][0] / 1000
    theta = scan_data["theta"][0] * np.pi/180
    nstage = 25
    ni1 = 256
    ni2 = 1600
    dstage = scan_data["scan step"][0] / 1000
    nvols = 10000

    fnames = [os.path.join(root_dir, "Image2_%d.tif") % (ii + 1) for ii in range(0, nstage * nvols)]

elif mode == "ndtiff":
    scan_data_fname = os.path.join(root_dir, "..", "..", "scan_metadata.csv")
    scan_data = load_dataset.read_metadata(scan_data_fname)
    dc = scan_data["pixel_size"] / 1000
    theta = scan_data["theta"] * np.pi/180
    nstage = scan_data["scan_axis_positions"]
    ni1 = scan_data["y_pixels"]
    ni2 = scan_data["x_pixels"]
    dstage = scan_data["scan_step"] / 1000
    nvols = scan_data["num_t"]

    fnames = glob.glob(os.path.join(root_dir, "*.tif"))
else:
    raise ValueError("mode must be 'hcimage' or 'ndtiff'")

x, y, z = localize.get_lab_coords(ni2, ni1, dc, theta, dstage * np.arange(nstage))

# initial roi values to plot
# z, y, x # these are not literal sizes, rather the computed ROI contains this cube
# sizes_roi = (3., 3., 3.)
# sizes_roi = (1e10, 1e10, 1e10)
sizes_roi = (4., 4., 12.)
# (y, x)
# centers_roi = (18., 180.)
centers_roi = (21., 190.)
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

plots_vstart = 0.06
plots_hstart = 0.05
plots_vsize_max = 0.93
plots_hsize_max = 0.8
plot_sep = 0.0

# [left, bottom, width, height]
ax_xy = plt.axes([0, 0, 0, 0])
ax_xz = plt.axes([0, 0.1, 0, 0])
ax_yz = plt.axes([0, 0.2, 0, 0])

# sliders and buttons
button_hsize = 0.04
button_vsize = 0.03
slider_hsize = 0.1
slider_vsize = 0.03
hpos_center = 0.9 # center for row of sliders/button
hspace = 0.02 # separation between buttons on one rwo
vsep = 0.04 # center to center distance of lines

ax_index_slider = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - vsep, slider_hsize, slider_vsize], facecolor='lightgoldenrodyellow')
index_slider = matplotlib.widgets.Slider(ax_index_slider, 'time index', 0, 1000, valinit=0, valstep=1, closedmin=True, closedmax=False)

ax_dec_index = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 2 * vsep, button_hsize, button_vsize])
ax_adv_index = plt.axes([hpos_center + 0.5 * hspace, 1 - 2 * vsep, button_hsize, button_vsize])
button_dec_index = Button(ax_dec_index, '<', color='w', hovercolor='b')
button_adv_index = Button(ax_adv_index, '>', color='w', hovercolor='b')

# cx
ax_cx = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 3 * vsep, slider_hsize, slider_vsize])
cx_slider = matplotlib.widgets.Slider(ax_cx, "cx", x.min(), x.max(), valinit=centers_roi[1], valstep=1, closedmin=True, closedmax=True)

ax_dec_cx = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 4 * vsep, button_hsize, button_vsize])
ax_adv_cx = plt.axes([hpos_center + 0.5 * hspace, 1 - 4 * vsep, button_hsize, button_vsize])
button_dec_cx = Button(ax_dec_cx, '<', color="w", hovercolor='b')
button_adv_cx = Button(ax_adv_cx, '>', color="w", hovercolor='b')

# cy
ax_cy = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 5 * vsep, slider_hsize, slider_vsize])
cy_slider = matplotlib.widgets.Slider(ax_cy, "cy", y.min(), y.max(), valinit=centers_roi[0], valstep=1, closedmin=True, closedmax=True)

ax_dec_cy = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 6 * vsep, button_hsize, button_vsize])
ax_adv_cy = plt.axes([hpos_center + 0.5 * hspace, 1 - 6 * vsep, button_hsize, button_vsize])
button_dec_cy = Button(ax_dec_cy, '<', color="w", hovercolor='b')
button_adv_cy = Button(ax_adv_cy, '>', color="w", hovercolor='b')

# cz
cz_start, _, _ = localize.find_trapezoid_cz(centers_roi[0], (z, y, x))

ax_cz = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 7 * vsep, slider_hsize, slider_vsize])
cz_slider = matplotlib.widgets.Slider(ax_cz, "cz", z.min(), z.max(), valinit=float(cz_start),
                                      valstep=1, closedmin=True, closedmax=True)

ax_dec_cz = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 8 * vsep, button_hsize, button_vsize])
ax_adv_cz = plt.axes([hpos_center + 0.5 * hspace, 1 - 8 * vsep, button_hsize, button_vsize])
button_dec_cz = Button(ax_dec_cz, '<', color="w", hovercolor='b')
button_adv_cz = Button(ax_adv_cz, '>', color="w", hovercolor='b')


# wx
ax_wx = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 9 * vsep, slider_hsize, slider_vsize])
wx_slider = matplotlib.widgets.Slider(ax_wx, "wx", 0, x.max() - x.min(), valinit=sizes_roi[2], valstep=1, closedmin=True, closedmax=True)

ax_dec_wx = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 10 * vsep, button_hsize, button_vsize])
ax_adv_wx = plt.axes([hpos_center + 0.5 * hspace, 1 - 10 * vsep, button_hsize, button_vsize])
button_dec_wx = Button(ax_dec_wx, '<', color="w", hovercolor='b')
button_adv_wx = Button(ax_adv_wx, '>', color="w", hovercolor='b')

# wy
ax_wy = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 11 * vsep, slider_hsize, slider_vsize])
wy_slider = matplotlib.widgets.Slider(ax_wy, "wy", 0, y.max() - y.min(), valinit=sizes_roi[1], valstep=1, closedmin=True, closedmax=True)

ax_dec_wy = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 12 * vsep, button_hsize, button_vsize])
ax_adv_wy = plt.axes([hpos_center + 0.5 * hspace, 1 - 12 * vsep, button_hsize, button_vsize])
button_dec_wy = Button(ax_dec_wy, '<', color="w", hovercolor='b')
button_adv_wy = Button(ax_adv_wy, '>', color="w", hovercolor='b')

# wz
ax_wz = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 13 * vsep, slider_hsize, slider_vsize])
wz_slider = matplotlib.widgets.Slider(ax_wz, "wz", 0, z.max() - z.min(), valinit=sizes_roi[0], valstep=1, closedmin=True, closedmax=True)

ax_dec_wz = plt.axes([hpos_center - button_hsize - 0.5 * hspace, 1 - 14 * vsep, button_hsize, button_vsize])
ax_adv_wz = plt.axes([hpos_center + 0.5 * hspace, 1 - 14 * vsep, button_hsize, button_vsize])
button_dec_wz = Button(ax_dec_wz, '<', color="w", hovercolor='b')
button_adv_wz = Button(ax_adv_wz, '>', color="w", hovercolor='b')

# update button
ax_update = plt.axes([hpos_center - 0.5 * slider_hsize, 1 - 15 * vsep, slider_hsize, slider_vsize])
button_update = Button(ax_update, "update", color="w", hovercolor="b")

# colorbar axes
ax_cb_guess = plt.axes([0.85, 0.05, 0.04, 0.4])
ax_cb_guess.set_frame_on(False)
ax_cb_guess.set_xticks([])
ax_cb_guess.set_xticklabels([])
ax_cb_guess.set_yticks([])
ax_cb_guess.set_yticklabels([])

ax_cb_fit = plt.axes([0.9, 0.05, 0.04, 0.4])
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

    imgs, _ = load_dataset.load_volume(fnames, vol_index, nstage, chunk_index=0, mode=mode)
    imgs = np.flip(imgs, axis=1)

    imgs_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    coords = (z[:, roi[2]:roi[3], :], y[roi[0]:roi[1], roi[2]:roi[3], :], x[:, :, roi[4]:roi[5]])

    xroi, yroi, zroi, img_deskew_roi = localize.interp_opm_data(imgs_roi, dc, dstage, theta, mode="ortho-interp")

    xroi += x[0, 0, roi[4]]
    yroi += y[roi[0], roi[2], 0]
    zroi += z[0, roi[2], 0]
    coords_deskew = (zroi, yroi, xroi)

    return coords, coords_deskew, img_deskew_roi

# function to update data displayed on plot based on slider values
def update(val):
    # update indices / centers
    index = index_slider.val
    cx = float(cx_slider.val)
    cy = float(cy_slider.val)
    if auto_track_cz:
        cz, _, _ = localize.find_trapezoid_cz(cy, (z, y, x))
    else:
        cz = float(cz_slider.val)
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
    nz_pad = 8
    extent_yz = [zroi[0] - nz_pad * dz - 0.5 * dz, zroi[-1] + nz_pad * dz + 0.5 * dz, yroi[0] - 0.5 * dy,
                 yroi[-1] + 0.5 * dy]

    img_xy = np.nanmax(img, axis=0)
    img_yz = np.pad(np.transpose(np.nanmax(img, axis=2)), ((0, 0), (nz_pad, nz_pad)), mode="constant", constant_values=np.nan)
    img_xz = np.pad(np.nanmax(img, axis=1), ((nz_pad, nz_pad), (0, 0)), mode="constant", constant_values=np.nan)

    # get figure aspect ratio
    fig_aspect_ratio = figh.get_size_inches()[0] / figh.get_size_inches()[1]

    # set aspect ratios for axes
    # aspect ratio = horizontal / vertical size
    # aspect ratio for all axes
    aspect_ratio = (extent_xy[1] - extent_xy[0] + extent_yz[1] - extent_yz[0]) / \
                   (extent_xy[3] - extent_xy[2] + extent_xz[3] - extent_xz[2]) / fig_aspect_ratio

    hsize = plots_hsize_max
    vsize = hsize / aspect_ratio
    if vsize > plots_vsize_max:
        vsize = plots_vsize_max
        hsize = vsize * aspect_ratio

    aspect_ratio_xy = (extent_xy[1] - extent_xy[0]) / (extent_xy[3] - extent_xy[2]) / fig_aspect_ratio
    aspect_ratio_xz = (extent_xz[1] - extent_xz[0]) / (extent_xz[3] - extent_xz[2]) / fig_aspect_ratio
    aspect_ratio_yz = (extent_yz[1] - extent_yz[0]) / (extent_yz[3] - extent_yz[2]) / fig_aspect_ratio

    # set individual sizes
    # xy_size is the xy graph fraction of the total size
    xy_vsize = vsize * (extent_xy[1] - extent_xy[0]) / (extent_xy[1] - extent_xy[0] + extent_yz[1] - extent_yz[0]) - 0.5 * plot_sep
    xy_hsize = xy_vsize * aspect_ratio_xy

    xz_hsize = xy_hsize
    xz_vsize = xy_hsize / aspect_ratio_xz

    yz_vsize = xy_vsize
    yz_hsize = yz_vsize * aspect_ratio_yz

    ax_xy.clear()
    ax_xy.set_facecolor("grey")
    ax_xy.set_position([plots_hstart + yz_hsize + plot_sep/fig_aspect_ratio,
                        plots_vstart + xz_vsize + plot_sep, xy_hsize, xy_vsize])
    ax_xy.set_xticklabels([])
    ax_xy.set_yticklabels([])
    ax_xy.set_xticks([])
    ax_xy.set_yticks([])

    ax_xz.clear()
    ax_xz.set_facecolor("grey")
    ax_xz.set_position([plots_hstart + yz_hsize + plot_sep/fig_aspect_ratio,
                        plots_vstart, xz_hsize, xz_vsize])
    ax_xz.set_xlabel("x ($\mu$m)")
    ax_xz.set_yticklabels([])
    ax_xz.set_yticks([])

    ax_yz.clear()
    ax_yz.set_facecolor("grey")
    ax_yz.set_position([plots_hstart, plots_vstart + xz_vsize + plot_sep, yz_hsize, yz_vsize])
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
    vmin = np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 0.5)
    vmax = np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 99.99)

    ax_xy.imshow(img_xy, extent=extent_xy, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, interpolation='none')
    xlim = ax_xy.get_xlim()
    ylim = ax_xy.get_ylim()

    ax_yz.imshow(img_yz, extent=extent_yz, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')
    zlim = ax_yz.get_xlim()

    ax_xz.imshow(img_xz, extent=extent_xz, cmap="bone", origin="lower", vmin=vmin, vmax=vmax, aspect="auto", interpolation='none')

    # plot boundaries
    ax_xy.plot([x.min(), x.max(), x.max(), x.min(), x.min()], [y.min(), y.min(), y.max(), y.max(), y.min()], 'r')
    ax_xz.plot([x.min(), x.max(), x.max(), x.min(), x.min()], [z.min(), z.min(), z.max(), z.max(), z.min()], 'r')
    ax_yz.plot([z.min(), z.min(), z.max(), z.max(), z.min()], [y[0, 0], y[-1, 0], y[-1, -1], y[0, -1], y[0, 0]], 'r')

    # plot guesses before fitting
    if plot_guesses:
        centers_guess = loc_data["centers_guess"]
        # centers_guess_in_trap = localize.point_in_trapezoid(centers_guess, xroi_opm, yroi_opm, zroi_opm)
        centers_guess_to_plot = np.logical_and.reduce((centers_guess[:, 0] <= zroi_opm.max() + dz * nz_pad, centers_guess[:, 0] >= zroi_opm.min() - dz * nz_pad,
                                                       centers_guess[:, 1] <= yroi_opm.max(), centers_guess[:, 1] >= yroi_opm.min(),
                                                       centers_guess[:, 2] <= xroi_opm.max(), centers_guess[:, 2] >= xroi_opm.min()))
        centers_guess = centers_guess[centers_guess_to_plot]

        ax_xy.scatter(centers_guess[:, 2], centers_guess[:, 1], edgecolors=cmap_guess(centers_guess[:, 0] / z.max()),
                      facecolors="none", marker='o', label="guesses")
        ax_yz.scatter(centers_guess[:, 0], centers_guess[:, 1], edgecolors=cmap_guess(1.), facecolors="none", marker='o')
        ax_xz.scatter(centers_guess[:, 2], centers_guess[:, 0], edgecolors=cmap_guess(1.), facecolors="none", marker='o')

    # plot fit points with z-position displayed via colorbar, and display number of point
    ax_xy.scatter(centers[:, 2], centers[:, 1], edgecolors=cmap(centers[:, 0] / z.max()), facecolors="none", marker='o', label="fits")
    ax_yz.scatter(centers[:, 0], centers[:, 1], edgecolors=cmap(1.), facecolors="none", marker='o')
    ax_xz.scatter(centers[:, 2], centers[:, 0], edgecolors=cmap(1.), facecolors="none", marker='o')
    if annotate_points:
        for ii in range(len(centers)):
            ax_xy.annotate("%d" % ii, xy=(centers[ii, 2], centers[ii, 1]), color=[1., 1., 0.2])
            ax_yz.annotate("%d" % ii, xy=(centers[ii, 0], centers[ii, 1]), color=[1., 1., 0.2])
            ax_xz.annotate("%d" % ii, xy=(centers[ii, 2], centers[ii, 0]), color=[1., 1., 0.2])


    if plot_tracks:
        # remove track if does not occur at this time point
        cols_not_nan = np.logical_not(np.all(np.isnan(track_data[:, :index + 1, 0]), axis=1))

        # remove track if not in ROI
        # in_trap = localize.point_in_trapezoid(track_data[:, index], xroi_opm, yroi_opm, zroi_opm)
        centers_now = track_data[:, index]
        cols_to_plot = np.logical_and.reduce((centers_now[:, 0] <= zroi_opm.max() + dz * nz_pad, centers_now[:, 0] >= zroi_opm.min() - dz * nz_pad,
                                         centers_now[:, 1] <= yroi_opm.max(), centers_now[:, 1] >= yroi_opm.min(),
                                         centers_now[:, 2] <= xroi_opm.max(), centers_now[:, 2] >= xroi_opm.min(),
                                         cols_not_nan))

        dat = track_data[cols_to_plot, :index + 1]

        ax_xy.plot(dat[..., 2].transpose(), dat[..., 1].transpose(), 'o-', alpha=0.4, markerfacecolor="none", markeredgewidth=2)
        ax_yz.plot(dat[..., 0].transpose(), dat[..., 1].transpose(), 'o-', alpha=0.4, markerfacecolor="none", markeredgewidth=2)
        ax_xz.plot(dat[..., 2].transpose(), dat[..., 0].transpose(), 'o-', alpha=0.4, markerfacecolor="none", markeredgewidth=2)

    # ensure none of our plotting changed the axes limits
    ax_xy.set_xlim(xlim)
    ax_xy.set_ylim(ylim)
    ax_yz.set_xlim(zlim)
    ax_yz.set_ylim(ylim)
    ax_xz.set_xlim(xlim)
    ax_xz.set_ylim(zlim)

    figh.canvas.draw_idle()

def dec_cx(val):
    cx_slider.set_val(cx_slider.val - 1)

def adv_cx(val):
    cx_slider.set_val(cx_slider.val + 1)

def dec_cy(val):
    cy_slider.set_val(cy_slider.val - 1)

def adv_cy(val):
    cy_slider.set_val(cy_slider.val + 1)

def dec_cz(val):
    cz_slider.set_val(cz_slider.val - 1)

def adv_cz(val):
    cz_slider.set_val(cz_slider.val + 1)

def dec_wx(val):
    wx_slider.set_val(wx_slider.val - 1)

def adv_wx(val):
    wx_slider.set_val(wx_slider.val + 1)

def dec_wy(val):
    wy_slider.set_val(wy_slider.val - 1)

def adv_wy(val):
    wy_slider.set_val(wy_slider.val + 1)

def dec_wz(val):
    wz_slider.set_val(wz_slider.val - 1)

def adv_wz(val):
    wz_slider.set_val(wz_slider.val + 1)

def dec_index(val):
    pos = index_slider.val
    if pos > 0:
        new_pos = pos - 1
    else:
        new_pos = pos
    index_slider.set_val(new_pos)

def adv_index(val):
    pos = index_slider.val
    index_slider.set_val(pos + 1)

# assign button actions
button_dec_index.on_clicked(dec_index)
button_adv_index.on_clicked(adv_index)
button_dec_cx.on_clicked(dec_cx)
button_adv_cx.on_clicked(adv_cx)
button_dec_cy.on_clicked(dec_cy)
button_adv_cy.on_clicked(adv_cy)
button_dec_cz.on_clicked(dec_cz)
button_adv_cz.on_clicked(adv_cz)
button_dec_wx.on_clicked(dec_wx)
button_adv_wx.on_clicked(adv_wx)
button_dec_wy.on_clicked(dec_wy)
button_adv_wy.on_clicked(adv_wy)
button_dec_wz.on_clicked(dec_wz)
button_adv_wz.on_clicked(adv_wz)

# conditions to update
# replot if figure size is changed
figh.canvas.mpl_connect("resize_event", update)
index_slider.on_changed(update)
button_update.on_clicked(update)

# call once to ensure displays something
index_slider.set_val(0)

plt.show()