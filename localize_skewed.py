"""
Code for localization in native OPM frame

This file stores tools specific to the OPM skewed geometry. Most localization functions are found in localize.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import warnings
import time
import joblib

import fit_psf
import rois
import localize

# geometry functions
def nearest_pt_line(pt, slope, pt_line):
    """
    Get shortest distance between a point and a line.
    :param pt: (xo, yo), point of interest
    :param slope: slope of line
    :param pt_line: (xl, yl), point the line passes through

    :return pt: (x_near, y_near), nearest point on line
    :return d: shortest distance from point to line
    """
    xo, yo = pt
    xl, yl = pt_line
    b = yl - slope * xl

    x_int = (xo + slope * (yo - b)) / (slope ** 2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xo - x_int) ** 2 + (yo - y_int) ** 2)

    return (x_int, y_int), d


def point_in_trapezoid(pts, x, y, z):
    """
    Test if a point is in the trapzoidal region described by x,y,z
    :param pts: np.array([[cz0, cy0, cx0], [cz1, cy1, cx1], ...[czn, cyn, cxn]])
    :param x:
    :param y:
    :param z:
    :return:
    """
    if pts.ndim == 1:
        pts = pts[None, :]

    # get theta
    dc = x[0, 0, 1] - x[0, 0, 0]
    dz = z[0, 1, 0] - z[0, 0, 0]
    theta = np.arcsin(dz / dc)

    # get edges
    zstart = z.min()
    ystart = y[0, 0, 0]
    yend = y[-1, 0, 0]

    # need to round near machine precision, or can get strange results when points right on boundary
    decimals = 10
    not_in_region_x = np.logical_or(np.round(pts[:, 2], decimals) < np.round(x.min(), decimals),
                                    np.round(pts[:, 2], decimals) > np.round(x.max(), decimals))
    not_in_region_z = np.logical_or(np.round(pts[:, 0], decimals) < np.round(z.min(), decimals),
                                    np.round(pts[:, 0], decimals) > np.round(z.max(), decimals))
    # tilted lines describing ends
    not_in_region_yz = np.logical_or(np.round(pts[:, 0] - zstart, decimals) > np.round((pts[:, 1] - ystart) * np.tan(theta), decimals),
                                     np.round(pts[:, 0] - zstart, decimals) < np.round((pts[:, 1] - yend) * np.tan(theta), decimals))

    in_region = np.logical_not(np.logical_or(not_in_region_yz, np.logical_or(not_in_region_x, not_in_region_z)))

    return in_region

# coordinate transformations between OPM and coverslip frames
def get_skewed_coords(sizes, dc, ds, theta, scan_direction="lateral"):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param sizes: (n0, n1, n2)
    :param dc: camera pixel size
    :param ds: stage step size
    :param theta: in radians
    :return x, y, z:
    """
    nimgs, ny_cam, nx_cam = sizes

    if scan_direction == "lateral":
        x = dc * np.arange(nx_cam)[None, None, :]
        # y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        y = ds * np.arange(nimgs)[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    elif scan_direction == "axial":
        x = dc * np.arange(nx_cam)[None, None, :]
        y = dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = ds * np.arange(nimgs)[:, None, None] + dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    else:
        raise ValueError("scan_direction must be `lateral` or `axial` but was `%s`" % scan_direction)

    return x, y, z


def get_skewed_coords_deriv(sizes, dc, ds, theta):
    """
    derivative with respect to theta
    :param sizes:
    :param dc:
    :param ds:
    :param theta:
    :return:
    """
    nimgs, ny_cam, nx_cam = sizes
    dxdt = 0 * dc * np.arange(nx_cam)[None, None, :]
    dydt = 0 * ds * np.arange(nimgs)[:, None, None] - dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    dzdt = dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]

    dxds = 0 * dc * np.arange(nx_cam)[None, None, :]
    dyds = np.arange(nimgs)[:, None, None] + 0 * dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
    dzds = 0 * dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]

    dxdc = np.arange(nx_cam)[None, None, :]
    dydc = 0 * ds * np.arange(nimgs)[:, None, None] + np.cos(theta) * np.arange(ny_cam)[None, :, None]
    dzdc = np.sin(theta) * np.arange(ny_cam)[None, :, None]

    return [dxdt, dydt, dzdt], [dxds, dyds, dzds], [dxdc, dydc, dzdc]


def lab2cam(x, y, z, theta):
    """
    Convert xyz coordinates to camera coordinates sytem, x', y', and stage position.

    :param x:
    :param y:
    :param z:
    :param theta:

    :return xp:
    :return yp: yp coordinate
    :return stage_pos: distance of leading edge of camera frame from the y-axis
    """
    xp = x
    stage_pos = y - z / np.tan(theta)
    yp = (y - stage_pos) / np.cos(theta)
    return xp, yp, stage_pos


def xy_lab2cam(x, y, stage_pos, theta):
    """
    Convert xy coordinates to x', y' coordinates at a certain stage position
    
    :param x: 
    :param y: 
    :param stage_pos: 
    :param theta: 
    :return: 
    """
    xp = x
    yp = (y - stage_pos) / np.cos(theta)

    return xp, yp


def get_trapezoid_zbound(cy, coords):
    """
    Find z-range of trapezoid for given center position cy
    :param cy:
    :param coords: (z, y, x)
    :return zmax, zmin:
    """
    cy = np.array(cy)

    z, y, x = coords
    slope = (z[:, -1, 0] - z[:, 0, 0]) / (y[0, -1, 0] - y[0, 0, 0])

    # zmax
    zmax = np.zeros(cy.shape)
    cy_greater = cy > y[0, -1]
    zmax[cy_greater] = z.max()
    zmax[np.logical_not(cy_greater)] = slope * (cy[np.logical_not(cy_greater)] - y[0, 0])
    # if cy > y[0, -1]:
    #     zmax = z.max()
    # else:
    #     zmax = slope * (cy - y[0, 0])

    # zmin
    zmin = np.zeros(cy.shape)
    cy_less = cy < y[-1, 0]
    zmin[cy_less] = z.min()
    zmin[np.logical_not(cy_less)] = slope * (cy[np.logical_not(cy_less)] - y[-1, 0])

    # if cy < y[-1, 0]:
    #     zmin = z.min()
    # else:
    #     zmin = slope * (cy - y[-1, 0])

    return zmax, zmin


def get_trapezoid_ybound(cz, coords):
    """
    Find y-range of trapezoid for given center position cz
    :param cz:
    :param coords: (z, y, x)
    :return cy, ymax, ymin:
    """
    cz = np.array(cz)

    z, y, x = coords
    slope = (z[:, -1, 0] - z[:, 0, 0]) / (y[0, -1, 0] - y[0, 0, 0])

    ymin = cz / slope
    ymax = cz / slope + y[-1, 0]

    return ymax, ymin


# deskew
def interp_opm_data(imgs, dc, ds, theta, mode="ortho-interp"):
    """
    Interpolate OPM stage-scan data to be equally spaced in coverslip frame

    :param imgs: nz x ny x nx
    :param dc: image spacing in camera space, i.e. camera pixel size reference to object space
    :param ds: distance stage moves between frames
    :param theta:
    :return:
    """

    # fix y-positions from raw images
    nxp = imgs.shape[2]
    nyp = imgs.shape[1]
    nimgs = imgs.shape[0]

    # set up interpolated coordinates
    dx = dc
    dy = dc * np.cos(theta)
    dz = dc * np.sin(theta)
    x = dx * np.arange(0, nxp)
    y = dy * np.arange(0, nyp + int(ds / dy) * (nimgs - 1))
    z = dz * np.arange(0, nyp)
    nz = len(z)
    ny = len(y)

    img_unskew = np.nan * np.zeros((z.size, y.size, x.size))

    # todo: using loops for a start ... optimize later
    if mode == "row-interp":  # interpolate using nearest two points on same row
        for ii in range(nz):
            for jj in range(ny):
                # find coordinates of nearest two OPM images
                jeff = (jj * dy - ii * dc * np.cos(theta)) / ds
                jlow = int(np.floor(jeff))
                if (jlow + 1) >= (imgs.shape[0]):
                    continue

                # interpolate
                img_unskew[ii, jj, :] = (imgs[jlow, ii, :] * (jj * dy - jlow * ds) + imgs[jlow + 1, ii, :] * ((jlow + 1) * ds - jj * dy)) / ds

    # todo: this mode can be generalized to not use dy a multiple of dx
    elif mode == "ortho-interp":  # interpolate using nearest four points.
        for ii in range(nz):  # loop over z-positions
            for jj in range(ny):  # loop over large y-position steps (moving distance btw two real frames)
                # find coordinates of nearest two OPM images
                jeff = (jj * dy - ii * dc * np.cos(theta)) / ds
                jlow = int(np.floor(jeff))

                if (jlow + 1) >= (imgs.shape[0]) or jlow < 0:
                    continue

                pt_now = (y[jj], z[ii])

                # find nearest point to line along frame index jlow
                # this line passes through the point (jlow * ds, 0)
                pt_n1, dist_1 = nearest_pt_line(pt_now, np.tan(theta), (jlow * ds, 0))
                dist_along_line1 = np.sqrt((pt_n1[0] - jlow * ds) ** 2 + pt_n1[1] ** 2) / dc
                # as usual, need to round to avoid finite precision floor/ceiling issues if number is already an integer
                i1_low = int(np.floor(np.round(dist_along_line1, 14)))
                i1_high = int(np.ceil(np.round(dist_along_line1, 14)))

                if i1_high >= (imgs.shape[1] - 1) or i1_low < 0:
                    continue

                if np.round(dist_1, 14) == 0:
                    q1 = imgs[jlow, i1_low, :]
                elif i1_low < 0 or i1_high >= nyp:
                    q1 = np.nan
                else:
                    d1 = dist_along_line1 - i1_low
                    q1 = (1 - d1) * imgs[jlow, i1_low, :] + d1 * imgs[jlow, i1_high, :]

                # find nearest point to line passing along frame index (jlow + 1)
                # this line passes through the point ( (jlow + 1) * ds, 0)
                pt_no, dist_o = nearest_pt_line(pt_now, np.tan(theta), ( (jlow + 1) * ds, 0))
                dist_along_line0 = np.sqrt((pt_no[0] - (jlow + 1) * ds) ** 2 + pt_no[1] ** 2) / dc
                io_low = int(np.floor(np.round(dist_along_line0, 14)))
                io_high = int(np.ceil(np.round(dist_along_line0, 14)))

                if io_high >= (imgs.shape[1] - 1) or io_low < 0:
                    continue

                if np.round(dist_o, 14) == 0:
                    qo = imgs[jlow + 1, i1_low, :]
                elif io_low < 0 or io_high >= nyp:
                    qo = np.nan
                else:
                    do = dist_along_line0 - io_low
                    qo = (1 - do) * imgs[jlow + 1, io_low, :] + do * imgs[jlow + 1, io_high, :]

                # weighted average of qo and q1 based on their distance
                img_unskew[ii, jj, :] = (q1 * dist_o + qo * dist_1) / (dist_o + dist_1)

    else:
        raise Exception("mode must be 'row-interp' or 'ortho-interp' but was '%s'" % mode)

    return x, y, z, img_unskew


# point spread function model and fitting
def gaussian3d_angle(shape, dc, p):
    """
    
    :param shape: 
    :param dc:
    :param p: [A, cx, cy, cz, sxy, sz, bg, theta, ds]
    :return: 
    """

    x, y, z = get_skewed_coords(shape, dc, p[8], p[7])

    val = p[0] * np.exp(-(x - p[1])**2 / 2 / p[4]**2 - (y - p[2])**2 / 2 / p[4]**2 - (z - p[3])**2 / 2 / p[5]**2) + p[6]

    return val


def gaussian3d_angle_jacobian(shape, dc, p):
    x, y, z = get_skewed_coords(shape, dc, p[8], p[7])
    [dxdt, dydt, dzdt], [dxds, dyds, dzds], _ = get_skewed_coords_deriv(shape, dc, p[8], p[7])

    exp = np.exp(-(x - p[1])**2 / 2 / p[4]**2 - (y - p[2])**2 / 2 / p[4]**2 - (z - p[3])**2 / 2 / p[5]**2)

    jac = [exp,
           p[0] * exp * (x - p[1]) / p[4]**2,
           p[0] * exp * (y - p[2]) / p[4]**2,
           p[0] * exp * (z - p[3]) / p[5]**2,
           p[0] * exp * ((x - p[1])**2 + (y - p[2])**2) / p[4]**3,
           p[0] * exp * (z - p[3])**2 / p[5]**3,
           np.ones(shape),
           p[0] * exp * ((-1) * (x - p[1]) / p[4]**2 * dxdt +
                         (-1) * (y - p[2]) / p[4]**2 * dydt +
                         (-1) * (z - p[3]) / p[5]**2 * dzdt),
           p[0] * exp * ((-1) * (x - p[1]) / p[4] ** 2 * dxds +
                         (-1) * (y - p[2]) / p[4] ** 2 * dyds +
                         (-1) * (z - p[3]) / p[5] ** 2 * dzds)
           ]

    return jac


# generate synthetic image
def simulate_img(scan_params, physical_params, ncenters=1, centers=None, sf=3):

    # size and pixel size
    dc = scan_params["dc"]
    theta = scan_params["theta"]
    dstep = scan_params["dstep"]
    npos, ny, nx = scan_params["shape"]

    normal = np.array([0, -np.sin(theta), np.cos(theta)])  # normal of camera pixel

    # physical data
    na = physical_params["na"]
    ni = physical_params["ni"]
    emission_wavelength = physical_params["emission_wavelength"]
    # ideal sigmas
    sxy = 0.22 * emission_wavelength / na
    sz = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

    amp = physical_params["peak_photons"]
    bg = physical_params["background"]

    # coordinates
    x, y, z = get_skewed_coords((npos, ny, nx), dc, dstep, theta)

    # define centers
    if centers is None:
        centers = []
        while len(centers) < ncenters:
            xc = np.random.uniform(x.min(), x.max())
            yc = np.random.uniform(y.min(), y.max())
            zc = np.random.uniform(z.min(), z.max())
            c_proposed = np.array([zc, yc, xc])
            if point_in_trapezoid(c_proposed, x, y, z):
                centers.append(c_proposed)
        centers = np.asarray(centers)

    img_gt = np.zeros((x+y+z).shape)
    for c in centers:
        params = [amp, c[2], c[1], c[0], sxy, sz, bg]
        img_gt += fit_psf.gaussian3d_psf(x, y, z, dc, params, sf=sf, angles=np.array([0, theta, 0]))

    return img_gt, centers


def simulate_img_noise(ground_truth, max_photons, cam_gains=2, cam_offsets=100, cam_readout_noise_sds=5, photon_shot_noise=True):
    """
    Convert ground truth image (with values between 0-1) to simulated camera image, including the effects of
    photon shot noise and camera readout noise.

    :param use_otf:
    :param ground_truth: Relative intensity values of image
    :param max_photons: Mean photons emitted by ber of photons will be different than expected. Furthermore, due to
    the "blurring" of the point spread function and possible binning of the image, no point in the image
     may realize "max_photons"
    :param cam_gains: gains at each camera pixel
    :param cam_offsets: offsets of each camera pixel
    :param cam_readout_noise_sds: standard deviation characterizing readout noise at each camera pixel
    :param pix_size: pixel size of ground truth image in ums. Note that the pixel size of the output image will be
    pix_size * bin_size
    :param otf: optical transfer function. If None, use na and wavelength to set values
    :param na: numerical aperture. Only used if otf=None
    :param wavelength: wavelength in microns. Only used if otf=None
    :param photon_shot_noise: turn on/off photon shot-noise
    :param bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
    finer pixel grid.

    :return img:
    :return snr:
    :return max_photons_real:
    """
    if np.any(ground_truth > 1) or np.any(ground_truth < 0):
        warnings.warn('ground_truth image values should be in the range [0, 1] for max_photons to be correct')

    img = max_photons * ground_truth
    max_photons_real = img.max()
    # signal, used later to get SNR
    sig = cam_gains * img

    # add shot noise
    if photon_shot_noise:
        img = np.random.poisson(img)

    # add camera noise and convert from photons to ADU
    readout_noise = np.random.standard_normal(img.shape) * cam_readout_noise_sds
    img = cam_gains * img + readout_noise + cam_offsets

    # calculate SNR
    # assuming photon number large enough ~gaussian
    noise = np.sqrt(cam_readout_noise_sds ** 2 + cam_gains ** 2 * img)
    snr = sig / noise

    return img, snr, max_photons_real


# ROI tools
def get_skewed_roi_size(sizes, theta, dc, dstep, ensure_odd=True):
    """
    Get ROI size in OPM matrix that includes sufficient xy and z points

    :param sizes: [z-size, y-size, x-size] in same units as dc, dstep
    :param theta: angle in radians
    :param dc: camera pixel size
    :param dstep: step size
    :param bool ensure_odd:

    :return [no, n1, n2]: integer size of roi in skewed coordinates
    """

    # x-size determines n2 size
    n2 = int(np.ceil(sizes[2] / dc))

    # z-size determines n1
    n1 = int(np.ceil(sizes[0] / dc / np.sin(theta)))

    # set so that @ top and bottom z-points, ROI includes the full y-size
    n0 = int(np.ceil((0.5 * (n1 + 1)) * dc * np.cos(theta) + sizes[1]) / dstep)

    if ensure_odd:
        if np.mod(n2, 2) == 0:
            n2 += 1

        if np.mod(n1, 2) == 0:
            n1 += 1

        if np.mod(n0, 2) == 0:
            n0 += 1

    return [n0, n1, n2]


def get_skewed_roi(center, imgs, coords, sizes):
    """
    Given a center value (not necessarily aligned to the coordinates), find the closest region of interest (ROI)
    centered around that point.

    :param float center: [cz, cy, cx] in same units as x, y, z
    :param coords: a tuple of coordinates (z, y, x), where z, y, and x are broadcastable to the same shape as imgs.
    These coordinates are supplied as produced by get_skewed_coords()
    :param list[int] sizes: [n0, n1, n2], the size of the desired ROI in number of pixels along the skewed coordinate
    directions. This can be calculated with the help of get_skewed_roi_size()
    :return roi, img_roi, x_roi, y_roi, z_roi:
    """
    z, y, x = coords
    shape = imgs.shape
    i2 = np.argmin(np.abs(x[0, 0, :].ravel() - center[2]))
    i1 = np.argmin(np.abs(z[0, :, 0] - center[0]))
    i0 = np.argmin(np.abs(y[:, i1, 0] - center[1]))
    # i0, i1, _ = np.unravel_index(np.argmin((y - center[1]) ** 2 + (z - center[0]) ** 2), y.shape)
    roi = np.array(rois.get_centered_roi([i0, i1, i2], sizes, min_vals=[0, 0, 0], max_vals=np.array(shape)))

    img_roi = rois.cut_roi(roi, imgs)

    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]
    z_roi, y_roi, x_roi = np.broadcast_arrays(z_roi, y_roi, x_roi)
    # x_roi = rois.cut_roi(roi, x)
    # y_roi = rois.cut_roi(roi, y)
    # z_roi = rois.cut_roi(roi, z)

    return roi, img_roi, x_roi, y_roi, z_roi


def get_roi_mask(center, max_seps, coords):
    """
    Get mask to exclude points in the ROI that are far from the center. We do not want to include regions at the edges
    of the trapezoidal ROI in processing.

    :param center: (cz, cy, cx)
    :param max_seps: (dz, dxy)
    :param coords: (z, y, x) sizes must be broadcastable
    :return mask: same size as roi, 1 where point is allowed and nan otherwise
    """
    z_roi, y_roi, x_roi = coords
    x_roi_full, y_roi_full, z_roi_full = np.broadcast_arrays(x_roi, y_roi, z_roi)
    mask = np.ones(x_roi_full.shape, dtype=bool)

    # roi is parallelogram, so still want to cut out points which are too far from center
    too_far_xy = np.sqrt((x_roi - center[2]) ** 2 + (y_roi - center[1]) ** 2) > max_seps[1]
    too_far_z = np.abs(z_roi - center[0]) > max_seps[0]
    too_far = np.logical_or(too_far_xy, too_far_z)

    mask[too_far] = False

    return mask


# filtering
def get_filter_kernel_skewed(sigmas, dc, theta, dstage, sigma_cutoff=2):
    """
    Get gaussian filter convolution kernel in skewed coordinates

    :param sigmas: (sz, sy, sx) in the same units as dc and stage
    :param dc: pixel size
    :param theta: angle in radians
    :param dstage: stage step
    :param sigma_cutoff: number of standard deviations to include in the filter. This parameter determines the fitler size
    :return kernel:
    """
    # normalize everything to camera pixel size
    sigma_x_pix = sigmas[2] / dc
    sigma_y_pix = sigmas[2] / dc
    sigma_z_pix = sigmas[0] / dc
    nk_x = 2 * int(np.round(sigma_x_pix * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(sigma_y_pix * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(sigma_z_pix * sigma_cutoff)) + 1
    # determine how large the OPM geometry ROI needs to be to fit the desired filter
    roi_sizes = get_skewed_roi_size([nk_z, nk_y, nk_x], theta, 1, dstage / dc, ensure_odd=True)

    # get coordinates to evaluate kernel at
    xk, yk, zk = get_skewed_coords(roi_sizes, 1, dstage / dc, theta)
    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kernel = np.exp(-xk ** 2 / 2 / sigma_x_pix ** 2 - yk ** 2 / 2 / sigma_y_pix ** 2 - zk ** 2 / 2 / sigma_z_pix ** 2)
    kernel = kernel / np.sum(kernel)

    return kernel


def get_lapl_gauss_filter_kernel_skewed(sigmas_small, sigmas_large, dc, theta, dstage, sigma_cutoff=2):
    """
    Get Laplacian of Gaussian filter convolution kernel in skewed coordinates

    :param sigmas_small: (sz, sy, sx) in the same units as dc and stage
    :param sigmas_large: (sz, sy, sx) in the same units as dc and stage
    :param dc: pixel size
    :param theta: angle in radians
    :param dstage: stage step
    :param sigma_cutoff: number of standard deviations to include in the filter. This parameter determines the fitler size
    :return kernel:
    """
    # normalize everything to camera pixel size
    sxs_pix = sigmas_small[2] / dc
    sys_pix = sigmas_small[2] / dc
    szs_pix = sigmas_small[0] / dc

    sxl_pix = sigmas_large[2] / dc
    syl_pix = sigmas_large[2] / dc
    szl_pix = sigmas_large[0] / dc
    nk_x = 2 * int(np.round(sxl_pix * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(syl_pix * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(szl_pix * sigma_cutoff)) + 1
    # determine how large the OPM geometry ROI needs to be to fit the desired filter
    roi_sizes = get_skewed_roi_size([nk_z, nk_y, nk_x], theta, 1, dstage / dc, ensure_odd=True)

    # get coordinates to evaluate kernel at
    xk, yk, zk = get_skewed_coords(roi_sizes, 1, dstage / dc, theta)
    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kl = np.exp(-xk ** 2 / 2 / sxl_pix ** 2 - yk ** 2 / 2 / syl_pix ** 2 - zk ** 2 / 2 / szl_pix ** 2)
    kl = kl / np.sum(kl)
    ks = np.exp(-xk ** 2 / 2 / sxs_pix ** 2 - yk ** 2 / 2 / sys_pix ** 2 - zk ** 2 / 2 / szs_pix ** 2)
    ks = ks / np.sum(ks)
    kernel = ks - kl

    return kernel


# identify peaks
def get_skewed_footprint(min_sep_allowed, dc, ds, theta):
    """
    Get footprint for maximum filter in skewed coordinates

    :param min_sep_allowed: (dz, dy, dx)
    :param dc: pixel size
    :param ds: stage step
    :param theta: angle in radians
    :return footprint:
    """
    footprint_roi_size = get_skewed_roi_size(min_sep_allowed, theta, dc, ds, ensure_odd=True)
    footprint_form = np.ones(footprint_roi_size, dtype=bool)
    xf, yf, zf = get_skewed_coords(footprint_form.shape, dc, ds, theta)
    xf = xf - xf.mean()
    yf = yf - yf.mean()
    zf = zf - zf.mean()
    footprint_mask = get_roi_mask((0, 0, 0), min_sep_allowed, (zf, yf, xf))
    footprint_mask[np.isnan(footprint_mask)] = 0
    footprint_mask = footprint_mask.astype(np.bool)

    return footprint_form * footprint_mask


# localization functions
def localize_radial_symm(img, coords, mode="radial-symmetry"):
    """

    """
    # todo: check quality of localizations
    if img.ndim != 3:
        raise ValueError("img must be a 3D array, but was %dD" % img.ndim)

    nstep, ni1, ni2 = img.shape
    z, y, x = coords

    if mode == "centroid":
        w = np.nansum(img)
        xc = np.nansum(img * x) / w
        yc = np.nansum(img * y) / w
        zc = np.nansum(img * z) / w
    elif mode == "radial-symmetry":
        yk = 0.5 * (y[:-1, :-1, :] + y[1:, 1:, :])
        xk = 0.5 * (x[:, :, :-1] + x[:, :, 1:])
        zk = 0.5 * (z[:, :-1] + z[:, 1:])
        coords = (zk, yk, xk)

        # take a cube of 8 voxels, and compute gradients at the center, using the four pixel diagonals that pass
        # through the center
        grad_n1 = img[1:, 1:, 1:] - img[:-1, :-1, :-1]
        # vectors go [nz, ny, nx]
        n1 = np.array([zk[0, 1, 0] - zk[0, 0, 0], yk[1, 1, 0] - yk[0, 0, 0], xk[0, 0, 1] - xk[0, 0, 0]])
        n1 = n1 / np.linalg.norm(n1)

        grad_n2 = img[1:, :-1, 1:] - img[:-1, 1:, :-1]
        n2 = np.array([zk[0, 0, 0] - zk[0, 1, 0], yk[1, 0, 0] - yk[0, 1, 0], xk[0, 0, 1]- xk[0, 0, 0]])
        n2 = n2 / np.linalg.norm(n2)

        grad_n3 = img[1:, :-1, :-1] - img[:-1, 1:, 1:]
        n3 = np.array([zk[0, 0, 0] - zk[0, 1, 0], yk[1, 0, 0] - yk[0, 1, 0], xk[0, 0, 0] - xk[0, 0, 1]])
        n3 = n3 / np.linalg.norm(n3)

        grad_n4 = img[1:, 1:, :-1] - img[:-1, :-1, 1:]
        n4 = np.array([zk[0, 1, 0] - zk[0, 0, 0], yk[1, 1, 0] - yk[0, 0, 0], xk[0, 0, 0] - xk[0, 0, 1]])
        n4 = n4 / np.linalg.norm(n4)

        # compute the gradient xyz components
        # 3 unknowns and 4 eqns, so use pseudo-inverse to optimize overdetermined system
        uvec_mat = np.concatenate((n1[None, :], n2[None, :], n3[None, :], n4[None, :]), axis=0)
        dat_mat = np.concatenate((grad_n1.ravel()[None, :], grad_n2.ravel()[None, :],
                                  grad_n3.ravel()[None, :], grad_n4.ravel()[None, :]), axis=0)
        gradk = np.linalg.pinv(uvec_mat).dot(dat_mat)
        gradk = np.reshape(gradk, [3, nstep - 1, ni1 - 1, ni2 - 1])

        # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
        # from the centroid (as small slope errors can become large as the line is extended to the centroi)
        # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
        grad_norm = np.sqrt(np.sum(gradk ** 2, axis=0))
        centroid_gns = np.array([np.nansum(zk * grad_norm), np.nansum(yk * grad_norm), np.nansum(xk * grad_norm)]) / \
                       np.nansum(grad_norm)
        dk_centroid = np.sqrt((zk - centroid_gns[0]) ** 2 + (yk - centroid_gns[1]) ** 2 + (xk - centroid_gns[2]) ** 2)
        # weights
        wk = grad_norm ** 2 / dk_centroid

        # in 3D, parameterize a line passing through point Po along normal n by
        # V(t) = Pk + n * t
        # distance between line and point Pc minimized at
        # tmin = -\sum_{i=1}^3 (Pk_i - Pc_i) / \sum_i n_i^2
        # dk^2 = \sum_k \sum_i (Pk + n * tmin - Pc)^2
        # again, we want to minimize the quantity
        # chi^2 = \sum_k dk^2 * wk
        # so we take the derivatives of chi^2 with respect to Pc_x, Pc_y, and Pc_z, which gives a system of linear
        # equations, which we can recast into a matrix equation
        # np.array([[A, B, C], [D, E, F], [G, H, I]]) * np.array([[Pc_z], [Pc_y], [Pc_x]]) = np.array([[J], [K], [L]])
        with np.errstate(invalid="ignore"):
            nk = gradk / np.linalg.norm(gradk, axis=0)

        # def chi_sqr(xc, yc, zc):
        #     cs = (zc, yc, xc)
        #     chi = 0
        #     for ii in range(3):
        #         chi += np.sum((coords[ii] + nk[ii] * (cs[jj] - coords[jj]) - cs[ii]) ** 2 * wk)
        #     return chi

        # build 3x3 matrix from above
        mat = np.zeros((3, 3))
        for ll in range(3):  # rows of matrix
            for ii in range(3):  # columns of matrix
                if ii == ll:
                    mat[ll, ii] += np.nansum(-wk * (nk[ii] * nk[ll] - 1))
                else:
                    mat[ll, ii] += np.nansum(-wk * nk[ii] * nk[ll])

                for jj in range(3):  # internal sum
                    if jj == ll:
                        mat[ll, ii] += np.nansum(wk * nk[ii] * nk[jj] * (nk[jj] * nk[ll] - 1))
                    else:
                        mat[ll, ii] += np.nansum(wk * nk[ii] * nk[jj] * nk[jj] * nk[ll])

        # build vector from above
        vec = np.zeros((3, 1))
        coord_sum = zk * nk[0] + yk * nk[1] + xk * nk[2]
        for ll in range(3):  # sum over J, K, L
            for ii in range(3):  # internal sum
                if ii == ll:
                    vec[ll] += -np.nansum((coords[ii] - nk[ii] * coord_sum) * (nk[ii] * nk[ll] - 1) * wk)
                else:
                    vec[ll] += -np.nansum((coords[ii] - nk[ii] * coord_sum) * nk[ii] * nk[ll] * wk)

        # invert matrix
        zc, yc, xc = np.linalg.inv(mat).dot(vec)
        zc, yc, xc = np.float(zc), np.float(yc), np.float(xc)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    # compute useful parameters
    # amplitude
    amp = np.nanmax(img)

    # compute standard devs to estimate sizes
    w = np.nansum(img)
    sx = np.sqrt(np.nansum((x - xc) ** 2 * img) / w)
    sy = np.sqrt(np.nansum((y - yc) ** 2 * img) / w)
    sigma_xy = np.sqrt(sx * sy)
    sz = np.sqrt(np.nansum((z - zc) ** 2 * img) / w)

    return np.array([amp, xc, yc, zc, sigma_xy, sz, np.nan])


# plotting functions
def plot_skewed_roi(fit_params, roi, imgs, theta, x, y, z, init_params=None, same_color_scale=True, figsize=(16, 8),
                    prefix="", save_dir=None):
    """
    plot results from fit_roi()

    :param fit_params:
    :param roi:
    :param imgs:
    :param dc:
    :param theta:
    :param x:
    :param y:
    :param z:
    :param figsize:
    :return:
    """
    # extract useful coordinate info
    dstage = y[1, 0] - y[0, 0]
    dc = x[0, 0, 1] - x[0, 0, 0]
    stage_pos = y[:, 0]

    if init_params is not None:
        center_guess = np.array([init_params[3], init_params[2], init_params[1]])

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])
    #normal = np.array([0, -np.sin(theta), np.cos(theta)])

    # get ROI and coordinates
    img_roi = rois.cut_roi(roi, imgs)
    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]

    vmin_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)
    vmax_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # get fit
    fit_volume = fit_psf.gaussian3d_psf(x_roi, y_roi, z_roi, dc, fit_params, sf=3, angles=np.array([0., theta, 0.]))

    if same_color_scale:
        vmin_fit = vmin_roi
        vmax_fit = vmax_roi
    else:
        vmin_fit = np.percentile(fit_volume, 1)
        vmax_fit = np.percentile(fit_volume, 99.9)

    # interpolate on regular grid
    xi_roi, yi_roi, zi_roi, img_roi_unskew = interp_opm_data(img_roi, dc, dstage, theta, mode="ortho-interp")
    xi_roi += x_roi.min()
    dxi_roi = xi_roi[1] - xi_roi[0]
    yi_roi += y_roi.min()
    dyi_roi = yi_roi[1] - yi_roi[0]
    zi_roi += z_roi.min()
    dzi_roi = zi_roi[1] - zi_roi[0]

    # fit on regular grid
    fit_roi_unskew = fit_psf.gaussian3d_psf(xi_roi[None, None, :], yi_roi[None, :, None], zi_roi[:, None, None]
                                              , dc, fit_params, sf=1)

    # ################################
    # plot results interpolated on regular grid
    # ################################
    figh_interp = plt.figure(figsize=figsize)
    st_str = "Fit, max projections, interpolated, ROI = [%d, %d, %d, %d, %d, %d]\n" \
             "      A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % \
             (roi[0], roi[1], roi[2], roi[3], roi[4], roi[5],
              fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], fit_params[6])
    if init_params is not None:
        st_str += "\nguess A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % \
                  (init_params[0], init_params[1], init_params[2], init_params[3], init_params[4], init_params[5], init_params[6])
    plt.suptitle(st_str)
    grid = plt.GridSpec(2, 3)

    ax = plt.subplot(grid[0, 0])
    plt.imshow(np.nanmax(img_roi_unskew, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi],
               cmap="bone")
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    ax = plt.subplot(grid[0, 1])
    plt.imshow(np.nanmax(img_roi_unskew, axis=1), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi],
               cmap="bone")
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[2], center_guess[0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[0, 2])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        plt.imshow(np.nanmax(img_roi_unskew, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
                   extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi],
                   cmap="bone")
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    ax = plt.subplot(grid[1, 0])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=0).transpose(), vmin=vmin_fit, vmax=vmax_fit, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi],
               cmap="bone")
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[ 1], center_guess[2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")

    ax = plt.subplot(grid[1, 1])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=1), vmin=vmin_fit, vmax=vmax_fit, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi],
               cmap="bone")
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[2], center_guess[0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")

    ax = plt.subplot(grid[1, 2])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=2), vmin=vmin_fit, vmax=vmax_fit, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi],
               cmap="bone")
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")

    if save_dir is not None:
        figh_interp.savefig(os.path.join(save_dir, "%smax_projection.png" % prefix))
        plt.close(figh_interp)

    # ################################
    # plot fits in raw OPM coords
    # ################################
    figh_raw = plt.figure(figsize=figsize)
    st_str = "ROI single PSF fit, ROI = [%d, %d, %d, %d, %d, %d]\n" \
             "A=%0.5g, cx=%0.5g, cy=%0.5g, cz=%0.5g, sxy=%0.5g, sz=%0.5g, bg=%0.5g" % \
                 (roi[0], roi[1], roi[2], roi[3], roi[4], roi[5],
                  fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], fit_params[6])
    if init_params is not None:
        st_str += "\nguess A=%0.5g, cx=%0.5g, cy=%0.5g, cz=%0.5g, sxy=%0.5g, sz=%0.5g, bg=%0.5g" % \
                  (init_params[0], init_params[1], init_params[2], init_params[3], init_params[4], init_params[5],
                   init_params[6])

    plt.suptitle(st_str)
    grid = plt.GridSpec(3, roi[1] - roi[0])

    # todo: need to correct these coordinates for offsets
    xp = np.arange(imgs.shape[2]) * dc + x.min()
    yp = np.arange(imgs.shape[1]) * dc
    extent_roi = [xp[roi[4]] - 0.5 * dc, xp[roi[5] - 1] + 0.5 * dc,
                  yp[roi[2]] - 0.5 * dc, yp[roi[3] - 1] + 0.5 * dc]

    # stage positions contained in this ROI
    stage_pos_roi = stage_pos[roi[0]:roi[1]]
    # find the one closest to the center
    _, _, closest_stage_pos = lab2cam(fit_params[1], fit_params[2], fit_params[3], theta)
    jj_min = np.argmin(np.abs(closest_stage_pos - stage_pos_roi))
    for jj in range(len(stage_pos_roi)):
        xp, yp = xy_lab2cam(fit_params[1], fit_params[2], stage_pos_roi[jj], theta)

        ax = plt.subplot(grid[0, jj])
        plt.imshow(img_roi[jj], vmin=vmin_roi, vmax=vmax_roi, extent=extent_roi, origin="lower", cmap="bone")
        if jj != jj_min:
            plt.plot(xp, yp, 'mx')
        else:
            plt.plot(xp, yp, 'rx')

        plt.title("%0.2fum" % stage_pos_roi[jj])

        if jj == 0:
            plt.ylabel("Data\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        ax = plt.subplot(grid[1, jj])
        plt.imshow(fit_volume[jj], vmin=vmin_fit, vmax=vmax_fit, extent=extent_roi, origin="lower", cmap="bone")
        if jj != jj_min:
            plt.plot(xp, yp, 'mx')
        else:
            plt.plot(xp, yp, 'rx')

        if jj == 0:
            plt.ylabel("Fit\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        ax = plt.subplot(grid[2, jj])
        plt.imshow(img_roi[jj] - fit_volume[jj], extent=extent_roi, origin="lower", cmap="bone")
        if jj != jj_min:
            plt.plot(xp, yp, 'mx')
        else:
            plt.plot(xp, yp, 'rx')

        if jj == 0:
            plt.ylabel("Data - fit\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

    if save_dir is not None:
            figh_raw.savefig(os.path.join(save_dir, "%sraw.png" % prefix))
            plt.close(figh_raw)

    return figh_interp, figh_raw


# orchestration functions
def localize_skewed(imgs, params, abs_threshold, roi_size, filter_sigma_small, filter_sigma_large,
                    min_spot_sep, offsets=(0, 0, 0), allowed_polygon=None, sf=3,
                    mode="fit", use_gpu_fit=True, use_gpu_filter=True):
    """
    :param imgs: raw OPM data
    :param params: {"dc", "dstage", "theta"}
    :param abs_threshold:
    :param roi_size: (sz, sy, sx) size to include in xyz directions for fit rois. Note: currently sy=sx and sx is unused
    :param filter_sigma_small: (sz, sy, sx) sigmas for small size filter to be used in difference of gaussian filter
    :param filter_sigma_large:
    :param min_spot_sep: (dz, dxy) assume points separated by less than this distance come from one spot
    :param offsets: offset to apply to y-coordinates. Useful for analyzing datasets in chunks
    :return:
    """

    dz_min, dxy_min = min_spot_sep
    # ###################################################
    # set up geometry
    # ###################################################
    npos, ny, nx = imgs.shape

    dc = params["dc"]
    theta = params["theta"]
    stage_step = params["dstep"]

    x, y, z = get_skewed_coords((npos, ny, nx), dc, stage_step, theta)
    x += offsets[2]
    y += offsets[1]
    z += offsets[0]

    # ###################################################
    # smooth image and remove background with difference of gaussians filter
    # ###################################################
    tstart = time.perf_counter()

    ks = get_filter_kernel_skewed(filter_sigma_small, dc, theta, stage_step, sigma_cutoff=2)
    kl = get_filter_kernel_skewed(filter_sigma_large, dc, theta, stage_step, sigma_cutoff=2)
    imgs_hp = localize.filter_convolve(imgs, ks, use_gpu=use_gpu_filter)
    imgs_lp = localize.filter_convolve(imgs, kl, use_gpu=use_gpu_filter)
    imgs_filtered = imgs_hp - imgs_lp

    print("Filtered images in %0.2fs" % (time.perf_counter() - tstart))

    # ###################################################
    # mask off region of each camera frame
    # ###################################################
    tstart = time.perf_counter()

    if allowed_polygon is None:
        mask = np.expand_dims(np.ones(imgs_filtered[0].shape, dtype=bool), axis=0)
    else:
        p = Path(allowed_polygon)
        xx, yy = np.meshgrid(range(imgs.shape[2]), range(imgs.shape[1]))
        rs = np.concatenate((xx.ravel()[:, None], yy.ravel()[:, None]), axis=1)
        mask = p.contains_points(rs).reshape([1, imgs.shape[1], imgs.shape[2]])

    print("Masked region in %0.2fs" % (time.perf_counter() - tstart))

    # ###################################################
    # identify candidate beads
    # ###################################################
    tstart = time.perf_counter()

    footprint = get_skewed_footprint((dz_min, dxy_min, dxy_min), dc, stage_step, theta)
    centers_guess_inds, amps = localize.find_peak_candidates(imgs_filtered * mask, footprint, abs_threshold, use_gpu_filter=use_gpu_filter)

    # convert to xyz coordinates
    xc = x[0, 0, centers_guess_inds[:, 2]]
    yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
    zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
    centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

    print("Found %d points above threshold in %0.2fs" % (len(centers_guess), time.perf_counter() - tstart))

    # ###################################################
    # average multiple points too close together. Necessary bc if naive threshold, may identify several points
    # from same spot. Particularly important if spots have very different brightness levels.
    # ###################################################
    tstart = time.perf_counter()

    inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
    weights = imgs_filtered.ravel()[inds]
    centers_guess, inds_comb = localize.filter_nearby_peaks(centers_guess, dxy_min, dz_min, weights=weights, mode="average")

    amps = amps[inds_comb]
    print("Found %d points separated by dxy > %0.5g and dz > %0.5g in %0.1fs" %
          (len(centers_guess), dxy_min, dz_min, time.perf_counter() - tstart))

    # ###################################################
    # prepare ROIs
    # ###################################################
    tstart = time.perf_counter()

    # cut rois out
    roi_size_skew = get_skewed_roi_size(roi_size, theta, dc, stage_step, ensure_odd=True)
    rois, img_rois, xrois, yrois, zrois = zip(*[get_skewed_roi(c, imgs, x, y, z, roi_size_skew) for c in centers_guess])
    rois = np.asarray(rois)

    # exclude some regions of roi
    roi_masks = [get_roi_mask(c, (np.inf, 0.5 * roi_size[1]), (zrois[ii], yrois[ii], xrois[ii])) for ii, c in enumerate(centers_guess)]

    # mask regions
    xrois, yrois, zrois, img_rois = zip(*[(xr[rm][None, :], yr[rm][None, :], zr[rm][None, :], ir[rm][None, :])
                                        for xr, yr, zr, ir, rm in zip(xrois, yrois, zrois, img_rois, roi_masks)])

    # extract guess values
    bgs = np.array([np.mean(r) for r in img_rois])
    sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in zip(img_rois, xrois, centers_guess)])
    sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in zip(img_rois, yrois, centers_guess)])
    sxys = np.expand_dims(0.5 * (sxs + sys), axis=1)
    szs = np.expand_dims(np.array([np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in zip(img_rois, zrois, centers_guess)]), axis=1)


    # get initial parameter guesses
    init_params = np.concatenate((np.expand_dims(amps, axis=1),
                                  centers_guess[:, 2][:, None],
                                  centers_guess[:, 1][:, None],
                                  centers_guess[:, 0][:, None],
                                  sxys, szs,
                                  np.expand_dims(bgs, axis=1)),
                                 axis=1)

    print("Prepared %d rois in %0.2fs" % (len(rois), time.perf_counter() - tstart))

    # ###################################################
    # localization
    # ###################################################
    if mode == "fit":
        print("starting fitting for %d rois" % centers_guess.shape[0])
        tstart = time.perf_counter()

        fit_params, fit_states, chi_sqrs, niters, fit_t = localize.fit_gauss_rois(img_rois, (zrois, yrois, xrois),
                                                                   init_params, estimator="LSE",
                                                                   sf=sf, dc=dc, angles=(0., theta, 0.),
                                                                   use_gpu=use_gpu_fit)

    elif mode == "radial-symmetry":
        print("starting radial-symmetry localization for %d rois" % len(centers_guess))

        img_roi_masked = [rois.cut_roi(rois[ii], imgs_filtered) * roi_masks[ii] for ii in range(len(centers_guess))]
        for ii in range(len(img_roi_masked)):
            img_roi_masked[ii][img_roi_masked[ii] < 0] = 0

        results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                  joblib.delayed(localize_radial_symm)(img_roi_masked[ii], (zrois[ii], yrois[ii], xrois[ii]),
                                                       mode="radial-symmetry")
                  for ii in range(len(centers_guess)))

        fit_params = np.asarray(results)

    else:
        raise ValueError("'mode' must be 'fit' or 'radial-symmetry' but was '%s'" % mode)

    tend = time.perf_counter()
    print("Localization took %0.2fs" % (tend - tstart))

    fit_results = np.concatenate((np.expand_dims(fit_states, axis=1),
                                  np.expand_dims(chi_sqrs, axis=1),
                                  np.expand_dims(niters, axis=1)), axis=1)

    return rois, fit_params, init_params, fit_results, imgs_filtered, (z, y, x)

def filter_localizations(fit_params, init_params, coords, fit_dist_max_err, min_spot_sep,
                         sigma_bounds, amp_min=0, dist_boundary_min=(0, 0), mode="skewed"):
    """
    Given a collection of fits, determine which fits are plausible localizations based on the fit parameters.

    :param fit_params: nfits x 7 array
    :param init_params: nfits x 7 array
    :param coords: (z, y, x)
    :param fit_dist_max_err = (dz_max, dxy_max)
    :param min_spot_sep: (dz, dxy) assume points separated by less than this distance come from one spot
    :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max)) exclude fits with sigmas that fall outside
    these ranges
    :param amp_min: exclude fits with smaller amplitude
    :param dist_boundary_min: (dz_min, dxy_min)
    :return to_keep, conditions, condition_names, filter_settings:
    """

    filter_settings = {"fit_dist_max_err": fit_dist_max_err, "min_spot_sep": min_spot_sep,
                       "sigma_bounds": sigma_bounds, "amp_min": amp_min, "dist_boundary_min": dist_boundary_min}

    z, y, x = coords
    centers_guess = np.concatenate((init_params[:, 3][:, None], init_params[:, 2][:, None], init_params[:, 1][:, None]), axis=1)
    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]), axis=1)

    # ###################################################
    # only keep points if size and position were reasonable
    # ###################################################
    dz_min, dxy_min = dist_boundary_min

    if mode == "skewed":
        in_bounds = point_in_trapezoid(centers_fit, x, y, z)

        zmax, zmin = get_trapezoid_zbound(centers_fit[:, 1], coords)
        far_from_boundary_z = np.logical_and(centers_fit[:, 0] > zmin + dz_min, centers_fit[:, 0] < zmax - dz_min)

        ymax, ymin = get_trapezoid_ybound(centers_fit[:, 0], coords)
        far_from_boundary_y = np.logical_and(centers_fit[:, 1] > ymin + dxy_min, centers_fit[:, 0] < ymax - dxy_min)

        xmin = np.min(x)
        xmax = np.max(x)
        far_from_boundary_x = np.logical_and(centers_fit[:, 2] > xmin + dxy_min, centers_fit[:, 2] < xmax - dxy_min)

        in_bounds = np.logical_and.reduce((in_bounds, far_from_boundary_x, far_from_boundary_y, far_from_boundary_z))
    elif mode == "straight":
        in_bounds = np.logical_and.reduce((fit_params[:, 1] >= x.min() + dxy_min,
                                           fit_params[:, 1] <= x.max() - dxy_min,
                                           fit_params[:, 2] >= y.min() + dxy_min,
                                           fit_params[:, 2] <= y.max() - dxy_min,
                                           fit_params[:, 3] >= z.min() + dz_min,
                                           fit_params[:, 3] <= z.max() - dz_min))
    else:
        raise ValueError("mode must be 'skewed' or 'straight' but was '%s'" % mode)

    # maximum distance between fit center and guess center
    z_err_fit_max, xy_fit_err_max = fit_dist_max_err
    center_close_to_guess_xy = np.sqrt((centers_guess[:, 2] - fit_params[:, 1])**2 +
                                       (centers_guess[:, 1] - fit_params[:, 2])**2) <= xy_fit_err_max
    center_close_to_guess_z = np.abs(centers_guess[:, 0] - fit_params[:, 3]) <= z_err_fit_max

    # maximum/minimum sigmas AND combine all conditions
    (sz_min, sxy_min), (sz_max, sxy_max) = sigma_bounds
    conditions = np.stack((in_bounds, center_close_to_guess_xy, center_close_to_guess_z,
                            fit_params[:, 4] <= sxy_max, fit_params[:, 4] >= sxy_min,
                            fit_params[:, 5] <= sz_max, fit_params[:, 5] >= sz_min,
                            fit_params[:, 0] >= amp_min), axis=1)


    condition_names = ["in_bounds", "center_close_to_guess_xy", "center_close_to_guess_z",
                       "xy_size_small_enough", "xy_size_big_enough", "z_size_small_enough",
                       "z_size_big_enough", "amp_ok"]

    to_keep_temp = np.logical_and.reduce(conditions, axis=1)

    # ###################################################
    # check for unique points
    # ###################################################

    dz, dxy = min_spot_sep
    if np.sum(to_keep_temp) > 0:

        # only keep unique center if close enough
        _, unique_inds = localize.filter_nearby_peaks(centers_fit[to_keep_temp], dxy, dz, mode="keep-one")

        # unique mask for those in to_keep_temp
        is_unique = np.zeros(np.sum(to_keep_temp), dtype=bool)
        is_unique[unique_inds] = True

        # get indices of non-unique points among all points
        not_unique_inds_full = np.arange(len(to_keep_temp), dtype=int)[to_keep_temp][np.logical_not(is_unique)]

        # get mask in full space
        unique = np.ones(len(fit_params), dtype=bool)
        unique[not_unique_inds_full] = False
    else:
        unique = np.ones(len(fit_params), dtype=bool)

    conditions = np.concatenate((conditions, np.expand_dims(unique, axis=1)), axis=1)
    condition_names += ["unique"]
    to_keep = np.logical_and(to_keep_temp, unique)

    return to_keep, conditions, condition_names, filter_settings
