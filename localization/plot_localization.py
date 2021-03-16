import numpy as np
import matplotlib
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import pickle
import os
import tifffile

index = 1
root_dir = r"\\10.206.26.21\opm2\20210309\crowders-10x-50glyc\2021_03_10_17;57;53_localization"

# localization data
loc_data_dir = os.path.join(root_dir, "localization_results_vol_%d.pkl" % index)
with open(loc_data_dir, "rb") as f:
    loc_data = pickle.load(f)
centers = loc_data["centers"]

# scan data
scan_data_dir = os.path.join(root_dir, "..", "galvo_scan_params.pkl")
with open(scan_data_dir, "rb") as f:
    scan_data = pickle.load(f)

# image and coords
img_xy_dir = os.path.join(root_dir, "max_proj_xy_vol=%d_chunk=0.tiff" % index)
img_xy = tifffile.imread(img_xy_dir)

ny, nx = img_xy.shape
dc = scan_data["pixel size"][0] / 1000
theta = scan_data["theta"][0] * np.pi/180
nstage = 25
dstage = scan_data["scan step"][0] / 1000
dx = dc
dy = dc * np.cos(theta)
x = np.arange(nx) * dx
y = np.arange(ny) * dy

# create plot
plt.set_cmap("bone")
figh = plt.figure()

slider_axes = plt.axes([0.3, 0.1, 0.4, 0.1], facecolor='lightgoldenrodyellow')
sliders = matplotlib.widgets.Slider(slider_axes, 'index', 0, 100, valinit=0, valstep=1)

ax_b1 = plt.axes([0.2, 0.1, 0.05, 0.1])
ax_b2 = plt.axes([0.75, 0.1, 0.05, 0.1])
button1 = Button(ax_b1, '<', color='w', hovercolor='b')
button2 = Button(ax_b2, '>', color='w', hovercolor='b')

# [left, bottom, width, height]
ax = plt.axes([0.2, 0.25, 0.6, 0.6])
initialized = False

# function called when sliders are moved on plot
def update(val):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()


    ax.clear()
    index = sliders.val

    img_xy_dir = os.path.join(root_dir, "max_proj_xy_vol=%d_chunk=0.tiff" % index)
    img_xy = tifffile.imread(img_xy_dir)

    loc_data_dir = os.path.join(root_dir, "localization_results_vol_%d.pkl" % index)
    with open(loc_data_dir, "rb") as f:
        loc_data = pickle.load(f)
    centers = loc_data["centers"]

    ax.imshow(img_xy, extent=[x[0] - 0.5 * dx, x[-1] + 0.5 * dx, y[0] - 0.5 * dy, y[-1] + 0.5 * dy], cmap="bone",
               origin="lower",
               vmin=np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 1),
               vmax=np.percentile(img_xy[np.logical_not(np.isnan(img_xy))], 99.9))
    ax.plot(centers[:, 2], centers[:, 1], 'rx')

    if initialized:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    figh.canvas.draw_idle()

def backward(val):
    pos = sliders.val
    sliders.set_val(pos - 1)

def forward(val):
    pos = sliders.val
    sliders.set_val(pos + 1)

sliders.on_changed(update)
button1.on_clicked(backward)
button2.on_clicked(forward)
# call once to ensure displays something
sliders.set_val(0)
initialized = True

plt.show()