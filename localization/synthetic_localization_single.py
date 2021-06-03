"""
Test localization methods on a single synthetic ROI
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import localize

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
x, y, z = localize.get_skewed_coords((npos, ny, nx), dc, dstep, theta)

# simulated image
gt, c_gt = localize.simulate_img({"dc": dc, "dstep": dstep, "theta": theta, "shape": (npos, ny, nx)},
                           {"na": na, "ni": ni, "peak_photons": 1000, "background": 0.1, "emission_wavelength": emission_wavelength},
                           ncenters=1)
# camera and photon shot noise
img, _, _ = localize.simulate_img_noise(gt, 1, cam_gains=2, cam_offsets=100, cam_readout_noise_sds=5)

# localize using gauss nonlinear fit
tstart = time.process_time()

results = localize.fit_roi(img, (z, y, x), dc=dc, angles=np.array([0, theta, 0]))
c_nl = np.array([results["fit_params"][3], results["fit_params"][2], results["fit_params"][1]])

tend = time.process_time()
print("Fit took %0.3gs" % (tend - tstart))

# localize with radial symmetry method
rad_params = localize.localize_radial_symm(img, (z, y, x), mode="radial-symmetry")
c_rad = np.array([rad_params[3], rad_params[2], rad_params[1]])

# localize with centroid
cent_params = localize.localize_radial_symm(img, (z, y, x), mode="centroid")
c_cent = np.array([cent_params[3], cent_params[2], cent_params[1]])

# deskew image
xi, yi, zi, img_unskew = localize.interp_opm_data(img, dc, dstep, theta, mode="ortho-interp")
dxi = xi[1] - xi[0]
dyi = yi[1] - yi[0]
dzi = zi[1] - zi[0]
vmin = np.nanmin(img_unskew)
vmax = np.nanmax(img_unskew)

# plot results
figh_interp = plt.figure()
plt.suptitle("Fit, max projections, interpolated")
grid = plt.GridSpec(2, 2)

ax = plt.subplot(grid[0, 0])
plt.imshow(np.nanmax(img_unskew, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower", cmap="bone",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
plt.plot(c_nl[1], c_nl[2], 'rx', label="gauss nlsq")
plt.plot(c_gt[:, 1], c_gt[:, 2], 'b+', label="gt")
plt.plot(c_rad[1], c_rad[2], 'g1', label="rad-symm")
plt.plot(c_cent[1], c_cent[2], 'm2', label="centroid")
plt.xlabel("Y (um)")
plt.ylabel("X (um)")
plt.title("XY")
plt.legend()

ax = plt.subplot(grid[0, 1])
plt.imshow(np.nanmax(img_unskew, axis=1), vmin=vmin, vmax=vmax, origin="lower", cmap="bone",
           extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
plt.plot(c_nl[2], c_nl[0], 'rx')
plt.plot(c_gt[:, 2], c_gt[:, 0], 'b+')
plt.plot(c_rad[2], c_rad[0], 'g1')
plt.plot(c_cent[2], c_cent[0], 'm2')
plt.xlabel("X (um)")
plt.ylabel("Z (um)")
plt.title("XZ")

ax = plt.subplot(grid[1, 0])
plt.imshow(np.nanmax(img_unskew, axis=2), vmin=vmin, vmax=vmax, origin="lower", cmap="bone",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
plt.plot(c_nl[1], c_nl[0], 'rx')
plt.plot(c_gt[:, 1], c_gt[:, 0], 'b+')
plt.plot(c_rad[1], c_rad[0], 'g1')
plt.plot(c_cent[1], c_cent[0], 'm2')
plt.xlabel("Y (um)")
plt.ylabel("Z (um)")
plt.title("YZ")

