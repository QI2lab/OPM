"""
Test localization methods on single ROI
"""
import numpy as np
import matplotlib.pyplot as plt
import localize

# angle
theta = 30 * np.pi/180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel
# size and pixel size
nx = 31
ny = 31
npos = 21
dc = 0.115
dstep = dc * np.cos(theta) * 1
stage_pos = dstep * np.arange(npos)

#
na = 1.3
ni = 1.4
emission_wavelength = 0.605
sxy = 0.22 * emission_wavelength / na
sz = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2


# coordinates
x, y, z = localize.get_lab_coords(nx, ny, dc, theta, stage_pos)

# define center
xc = np.random.uniform(x.mean() - dc, x.mean() + dc)
yc = np.random.uniform(y.mean() - dc, y.mean() + dc)
zc = np.random.uniform(z.mean() - dc * np.sin(theta), z.mean() + dc * np.sin(theta))
amp = 1000
bg = 0.1
params = [amp, xc, yc, zc, sxy, sz, bg]
c_gt = np.array([params[3], params[2], params[1]])

img, _, _ = localize.simulated_img(localize.gaussian3d_pixelated_psf(x, y, z, [dc, dc], normal, params, sf=3), 1, 2, 100, 5, use_otf=False)

# localize
results = localize.fit_roi(img, x, y, z)
c_nl = np.array([results["fit_params"][3], results["fit_params"][2], results["fit_params"][1]])

c_rad = localize.localize_radial_symm(img, theta, dc, dstep)

# plot results
xi, yi, zi, img_unskew = localize.interp_opm_data(img, dc, dstep, theta, mode="ortho-interp")
dxi = xi[1] - xi[0]
dyi = yi[1] - yi[0]
dzi = zi[1] - zi[0]
vmin = np.nanmin(img_unskew)
vmax = np.nanmax(img_unskew)

figh_interp = plt.figure()
plt.suptitle("Fit, max projections, interpolated")
grid = plt.GridSpec(2, 2)

ax = plt.subplot(grid[0, 0])
plt.imshow(np.nanmax(img_unskew, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
plt.plot(c_nl[1], c_nl[2], 'rx')
plt.plot(c_gt[1], c_gt[2], 'k.')
plt.plot(c_rad[1], c_rad[2], 'g1')
plt.xlabel("Y (um)")
plt.ylabel("X (um)")
plt.title("XY")

ax = plt.subplot(grid[0, 1])
plt.imshow(np.nanmax(img_unskew, axis=1), vmin=vmin, vmax=vmax, origin="lower",
           extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
plt.plot(c_nl[2], c_nl[0], 'rx')
plt.plot(c_gt[2], c_gt[0], 'k.')
plt.plot(c_rad[2], c_rad[0], 'g1')
plt.xlabel("X (um)")
plt.ylabel("Z (um)")
plt.title("XZ")

ax = plt.subplot(grid[1, 0])
plt.imshow(np.nanmax(img_unskew, axis=2), vmin=vmin, vmax=vmax, origin="lower",
           extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
plt.plot(c_nl[1], c_nl[0], 'rx')
plt.plot(c_gt[1], c_gt[0], 'k.')
plt.plot(c_rad[1], c_rad[0], 'g1')
plt.xlabel("Y (um)")
plt.ylabel("Z (um)")
plt.title("YZ")

