"""
test different scan interp methods
"""
import time
import numpy as np
import localize
from localize import deskew
import matplotlib.pyplot as plt

# set scan parameters
theta = 30 * np.pi/180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel
# size and pixel size
nx = 15
ny = 31
npos = 31
dc = 0.115
dstep = dc * np.cos(theta) * 1
stage_pos = dstep * np.arange(npos)

# set physical parameters
na = 1.3
ni = 1.4
emission_wavelength = 0.605
sxy = 0.22 * emission_wavelength / na
sz = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

# coordinates
x, y, z = localize.get_lab_coords(nx, ny, dc, theta, stage_pos)

# simulated image
center = np.expand_dims(np.array([z.mean(), y.mean(), x.mean()]), axis=0)
gt, c_gt = localize.simulate_img({"dc": dc, "dstep": dstep, "theta": theta, "shape": (npos, ny, nx)},
                           {"na": na, "ni": ni, "peak_photons": 1000, "background": 0.1, "emission_wavelength": emission_wavelength},
                           centers=center)
# camera and photon shot noise
img, _, _ = localize.simulate_img_noise(gt, 1, cam_gains=2, cam_offsets=100, cam_readout_noise_sds=5)

tstart = time.perf_counter()
x1, y1, z1, deskew1 = localize.interp_opm_data(img, dc, dstep, theta, mode="ortho-interp")
tend = time.perf_counter()
print("unvectorized deskew ran in %0.5gs" % (tend - tstart))

tstart = time.perf_counter()
deskew2 = localize.deskew(img, [theta * 180/np.pi, dstep, dc])
tend = time.perf_counter()
print("numba deskew ran in %0.5gs" % (tend - tstart))
deskew2[deskew2 == 0] = np.nan

figh = plt.figure()
vmin = 100
vmax = 2e3
grid = plt.GridSpec(2, 3)

ax = plt.subplot(grid[0, 0])
plt.imshow(np.nanmax(deskew1, axis=0), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("x")
plt.ylabel("no numba implementation\ny")

ax = plt.subplot(grid[0, 1])
plt.imshow(np.nanmax(deskew1, axis=1), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("x")
plt.ylabel("z")

ax = plt.subplot(grid[0, 2])
plt.imshow(np.nanmax(deskew1, axis=2), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("y")
plt.ylabel("z")

ax = plt.subplot(grid[1, 0])
plt.imshow(np.nanmax(deskew2, axis=0), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("x")
plt.ylabel("numb implementation\ny")

ax = plt.subplot(grid[1, 1])
plt.imshow(np.nanmax(deskew2, axis=1), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("x")
plt.ylabel("z")

ax = plt.subplot(grid[1, 2])
plt.imshow(np.nanmax(deskew2, axis=2), cmap="bone", vmin=vmin, vmax=vmax)
plt.xlabel("y")
plt.ylabel("z")