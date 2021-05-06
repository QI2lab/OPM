"""
Code for localization in native OPM frame
"""
import os
import numpy as np
import scipy.optimize
import scipy.ndimage
import skimage.feature
import skimage.filters
import matplotlib.pyplot as plt
from matplotlib.path import Path
import warnings
import time
import joblib

# for filtering on GPU
try:
    import cupy as cp
    import cupyx.scipy.signal # new in v9, compiled github source
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# for fitting on GPU
try:
    import pygpufit.gpufit as gf
    GPUFIT_AVAILABLE = True
except ImportError:
    GPUFIT_AVAILABLE = False

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
def get_skewed_coords(sizes, dc, ds, theta):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param nx_cam:
    :param ny_cam:
    :param dc: camera pixel size
    :param theta:
    :param stage_pos: list of y-displacements for each scan position
    :return:
    """
    nimgs, ny_cam, nx_cam = sizes
    x = dc * np.arange(nx_cam)[None, None, :]
    # y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
    y = ds * np.arange(nimgs)[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
    z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]

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


def trapezoid_cz(cy, coords):
    """
    Find z-range of trapezoid for given center position cy
    :param cy:
    :param coords: (z, y, x)
    :return:
    """
    z, y, x = coords
    slope = (z[:, -1] - z[:, 0]) / (y[0, -1] - y[0, 0])

    if cy > y[0, -1]:
        zmax = z.max()
    else:
        zmax = slope * (cy - y[0, 0])

    if cy < y[-1, 0]:
        zmin = z.min()
    else:
        zmin = slope * (cy - y[-1, 0])

    cz = 0.5 * (zmax + zmin)

    return cz, zmax, zmin


def trapezoid_cy(cz, coords):
    """
    Find y-range of trapezoid for given center position cz
    :param cz:
    :param coords: (z, y, x)
    :return:
    """
    z, y, x = coords
    slope = (z[:, -1] - z[:, 0]) / (y[0, -1] - y[0, 0])

    ymin = cz / slope
    ymax = cz / slope + y[-1, 0]
    cy = 0.5 * (ymax + ymin)

    return cy, ymax, ymin


def get_coords(sizes, dc, dz):
    """
    Non-tilted coordinates
    :param sizes: (sz, sy, sx)
    :param dc:
    :param dz:
    :return:
    """
    x = np.expand_dims(np.arange(sizes[2]) * dc, axis=(0, 1))
    y = np.expand_dims(np.arange(sizes[1]) * dc, axis=(0, 2))
    z = np.expand_dims(np.arange(sizes[0]) * dz, axis=(1, 2))
    return x, y, z


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
def gaussian3d_pixelated_psf(x, y, z, ds, normal, p, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

    vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param ds: [dx, dy]
    :param normal:
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """
    if len(ds) != 2:
        raise ValueError("ds = [dx, dy] must have length was not 2")

    # generate new points in pixel
    if sf != 1:
        pts = np.arange(1 / sf / 2, 1 - 1 / sf / 2, 1 / sf) - 0.5
    else:
        pts = np.array([0])

    xp, yp = np.meshgrid(ds[0] * pts, ds[1] * pts)
    zp = np.zeros(xp.shape)

    # rotate points to correct position using normal vector
    # for now we will fix x, but lose generality
    eyp = np.cross(normal, np.array([1, 0, 0]))
    mat_r2rp = np.concatenate((np.array([1, 0, 0])[:, None], eyp[:, None], normal[:, None]), axis=1)
    result = mat_r2rp.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
    xs, ys, zs = result

    # now must add these to each point x, y, z
    xx_s = x[..., None] + xs[None, ...]
    yy_s = y[..., None] + ys[None, ...]
    zz_s = z[..., None] + zs[None, ...]

    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    psf = p[0] * np.mean(psf_s, axis=-1) + p[-1]

    return psf


def gaussian3d_pixelated_psf_jac(x, y, z, ds, normal, p, sf):
    """
    Jacobian of gaussian3d_pixelated_psf()

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param ds: [dx, dy]
    :param normal:
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """

    if len(ds) != 2:
        raise ValueError("ds = (dx, dy) length as not 2")

    # generate new points in pixel
    if sf != 1:
        pts = np.arange(1 / sf / 2, 1 - 1 / sf / 2, 1 / sf) - 0.5
    else:
        pts = np.array([0])
    xp, yp = np.meshgrid(ds[0] * pts, ds[1] * pts)
    zp = np.zeros(xp.shape)

    # rotate points to correct position using normal vector
    # for now we will fix x, but lose generality
    eyp = np.cross(normal, np.array([1, 0, 0]))
    mat_r2rp = np.concatenate((np.array([1, 0, 0])[:, None], eyp[:, None], normal[:, None]), axis=1)
    result = mat_r2rp.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
    xs, ys, zs = result

    # now must add these to each point x, y, z
    xx_s = x[..., None] + xs[None, ...]
    yy_s = y[..., None] + ys[None, ...]
    zz_s = z[..., None] + zs[None, ...]

    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    # psf = p[0] * psf_sum + p[-1]

    bcast_shape = (x + y + z).shape
    # [A, cx, cy, cz, sxy, sz, bg]
    jac = [np.mean(psf_s, axis=-1),
           p[0] * np.mean(2 * (xx_s - p[1]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (yy_s - p[2]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (zz_s - p[3]) / 2/ p[5]**2 * psf_s, axis=-1),
           p[0] * np.mean((2 / p[4]**3 * (xx_s - p[1])**2 / 2 +
                           2 / p[4]**3 * (yy_s - p[2])**2 / 2) * psf_s, axis=-1),
           p[0] * np.mean(2 / p[5]**3 * (zz_s - p[3])**2 / 2 * psf_s, axis=-1),
           np.ones(bcast_shape)]

    return jac


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


def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Fit 2D model function with capability of fixing some parameters

    :param np.array img: nd array
    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
    they will not affect the optimization results, although they will affect chi squared.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None, no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return:
    """
    to_use = np.logical_not(np.isnan(img))

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
        sd = np.ones(img.shape)

    # handle uncertainties that will cause fitting to fail
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        sd[sd == 0] = np.nanmean(sd[sd != 0])
        sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds. If within small tolerance, assume supposed to be equal and force
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            if init_params[ii] > bounds[1][ii] and np.round(init_params[ii] - bounds[1][ii], 12) == 0:
                init_params[ii] = bounds[1][ii]
            elif (init_params[ii] < bounds[0][ii] and np.round(init_params[ii] - bounds[0][ii], 12) == 0):
                init_params[ii] = bounds[0][ii]
            else:
                raise ValueError(
                    "Initial parameter at index %d had value %0.5g, which was outside of bounds (%0.5g, %0.5g)" %
                    (ii, init_params[ii], bounds[0][ii], bounds[1][ii]))

    if np.any(np.isnan(init_params)):
        raise ValueError("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise ValueError("bounds cannot include nans")

    def err_fn(p):
        return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())

    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # awful list comprehension. The idea is this: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]

    def pfree2pfull(pfree):
        return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])

    # map full parameters to reduced set
    def pfull2pfree(pfull):
        return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree):
        return err_fn(pfree2pfull(pfree))

    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(jac_fn(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    red_chi_sq = np.sum(np.square(err_fn(pfit))) / (img[to_use].size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result


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
        img_gt += gaussian3d_pixelated_psf(x, y, z, [dc, dc], normal, params, sf=sf)

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
def cut_roi(roi, img):
    return img[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]


def get_centered_roi(centers, sizes, min_vals=None, max_vals=None):
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: list of centers [c1, c2, ..., cn]
    :param sizes: list of sizes [s1, s2, ..., sn]
    :param min_values: list of minimimum allowed index values for each dimension
    :param max_values: list of maximum allowed index values for each dimension
    :return roi: [start_0, end_0, start_1, end_1, ..., start_n, end_n]
    """
    roi = []
    # for c, n in zip(centers, sizes):
    for ii in range(len(centers)):
        c = centers[ii]
        n = sizes[ii]

        # get ROI closest to centered
        end_test = np.round(c + (n - 1) / 2) + 1
        end_err = np.mod(end_test, 1)
        start_test = np.round(c - (n - 1) / 2)
        start_err = np.mod(start_test, 1)

        if end_err > start_err:
            start = start_test
            end = start + n
        else:
            end = end_test
            start = end - n

        if min_vals is not None:
            if start < min_vals[ii]:
                start = min_vals[ii]

        if max_vals is not None:
            if end > max_vals[ii]:
                end = max_vals[ii]

        roi.append(int(start))
        roi.append(int(end))

    return roi


def get_roi_size(sizes, dc, dz, ensure_odd=True):
    n0 = int(np.ceil(sizes[0] / dz))
    n1 = int(np.ceil(sizes[1] / dc))
    n2 = int(np.ceil(sizes[2] / dc))

    if ensure_odd:
        n0 += (1 - np.mod(n0, 2))
        n1 += (1 - np.mod(n1, 2))
        n2 += (1 - np.mod(n2, 2))

    return [n0, n1, n2]

def get_roi(center, x, y, z, sizes, shape):
    """

    :param center: [cz, cy, cx] in same units as x, y, z.
    :param x:
    :param y:
    :param z:
    :param sizes: [i0, i1, i2] integers
    :param shape: shape of full array ... todo: should be able to infer from x,y,z
    :return:
    """
    i0 = np.argmin(np.abs(z.ravel() - center[0]))
    i1 = np.argmin(np.abs(y.ravel() - center[1]))
    i2 = np.argmin(np.abs(x.ravel() - center[2]))

    roi = np.array(get_centered_roi([i0, i1, i2], sizes, min_vals=[0, 0, 0], max_vals=np.array(shape)))

    z_roi = z[roi[0]:roi[1]]
    y_roi = y[:, roi[2]:roi[3], :]
    x_roi = x[:, :, roi[4]:roi[5]]

    return roi, x_roi, y_roi, z_roi


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


def get_skewed_roi(center, x, y, z, sizes, shape):
    """

    :param float center: [cz, cy, cx] in same units as x, y, z
    :param x:
    :param y:
    :param z:
    :param int sizes: [n0, n1, n2]
    :param shape: shape of full image ... todo: should infer from x, y, z
    :return:
    """
    i2 = np.argmin(np.abs(x.ravel() - center[2]))
    i0, i1, _ = np.unravel_index(np.argmin((y - center[1]) ** 2 + (z - center[0]) ** 2), y.shape)
    roi = np.array(get_centered_roi([i0, i1, i2], sizes, min_vals=[0, 0, 0], max_vals=np.array(shape)))

    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]

    return roi, x_roi, y_roi, z_roi


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
    mask = np.ones(x_roi_full.shape)

    # roi is parallelogram, so still want to cut out points which are too far from center
    too_far_xy = np.sqrt((x_roi - center[2]) ** 2 + (y_roi - center[1]) ** 2) > max_seps[1]
    too_far_z = np.abs(z_roi - center[0]) > max_seps[0]
    too_far = np.logical_or(too_far_xy, too_far_z)

    mask[too_far] = np.nan

    return mask


# filtering
def get_filter_kernel_skewed(sigmas, dc, theta, dstage, sigma_cutoff=2):
    # normalize everything to camera pixel size
    sigma_x_pix = sigmas[2] / dc
    sigma_y_pix = sigmas[2] / dc
    sigma_z_pix = sigmas[0] / dc
    nk_x = 2 * int(np.round(sigmas[2] * sigma_cutoff)) + 1
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


def get_filter_kernel(sigmas, dc, dz, sigma_cutoff=2):
    # compute kernel size
    nk_x = 2 * int(np.round(sigmas[2] / dc * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(sigmas[1] / dc * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(sigmas[0] / dz * sigma_cutoff)) + 1

    # get coordinates to evaluate kernel at
    xk = np.expand_dims(np.arange(nk_x) * dc, axis=(0, 1))
    yk = np.expand_dims(np.arange(nk_y) * dc, axis=(0, 2))
    zk = np.expand_dims(np.arange(nk_z) * dz, axis=(1, 2))

    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kernel = np.exp(-xk ** 2 / 2 / sigmas[2] ** 2 - yk ** 2 / 2 / sigmas[1] ** 2 - zk ** 2 / 2 / sigmas[0] ** 2)
    kernel = kernel / np.sum(kernel)

    return kernel


def filter_convolve(imgs, kernel, use_gpu=True):
    """
    Gaussian filter accounting for OPM geometry

    :param imgs:
    :param sigmas: (sigma_z, sigma_y, sigma_x)
    :param dc:
    :param theta:
    :param dstage:
    :param sigma_cutoff: number of sigmas to cutoff the kernel at # todo: allow different in different directions...
    :return:
    """

    # convolve, and deal with edges by normalizing
    # todo: is there a wrap around issue, where this blends opposite edges?
    if use_gpu:
        kernel_cp = cp.asarray(kernel, dtype=cp.float32)
        imgs_cp = cp.asarray(imgs, dtype=cp.float32)
        imgs_filtered = cp.asnumpy(cupyx.scipy.signal.fftconvolve(imgs_cp, kernel_cp, mode="same")/
                                   cupyx.scipy.signal.fftconvolve(cp.ones(imgs.shape), kernel_cp, mode="same"))
    else:
        imgs_filtered = scipy.signal.fftconvolve(imgs, kernel, mode="same") / \
             scipy.signal.fftconvolve(np.ones(imgs.shape), kernel, mode="same")

    # this method too slow for large filter sizes
    # imgs_filtered = scipy.ndimage.convolve(imgs, kernel, mode="constant", cval=0) / \
    #                 scipy.ndimage.convolve(np.ones(imgs.shape), kernel, mode="constant", cval=0)

    return imgs_filtered


# identify peaks
def get_skewed_footprint(min_sep_allowed, dc, ds, theta):
    """
    Get footprint for maximum filter
    :param min_sep_allowed: (dz, dy, dx)
    :param dc:
    :param ds:
    :param theta:
    :return:
    """
    footprint_roi_size = get_skewed_roi_size(min_sep_allowed, theta, dc, ds, ensure_odd=True)
    footprint_form = np.ones(footprint_roi_size, dtype=np.bool)
    xf, yf, zf = get_skewed_coords(footprint_form.shape, dc, ds, theta)
    xf = xf - xf.mean()
    yf = yf - yf.mean()
    zf = zf - zf.mean()
    footprint_mask = get_roi_mask((0, 0, 0), min_sep_allowed, (zf, yf, xf))
    footprint_mask[np.isnan(footprint_mask)] = 0
    footprint_mask = footprint_mask.astype(np.bool)

    return footprint_form * footprint_mask


def get_footprint(min_sep_allowed, dc, dz):
    """
    Get footprint for maximum filter
    :param min_sep_allowed: (sz, sy, sx)
    :param dc:
    :param dz:
    :return:
    """

    sz, sy, sx = min_sep_allowed

    nz = int(np.ceil(sz / dz))
    if np.mod(nz, 2) == 0:
        nz += 1

    ny = int(np.ceil(sy / dc))
    if np.mod(ny, 2) == 0:
        ny += 1

    nx = int(np.ceil(sx / dc))
    if np.mod(nx, 2) == 0:
        nx += 1

    footprint = np.ones((nz, ny, nx), dtype=np.bool)

    return footprint


def find_peak_candidates(imgs, footprint, threshold, use_gpu_filter=CUPY_AVAILABLE):
    """
    Find peak candidates using maximum filter

    :param imgs:
    :param footprint: footprint to use for maximum filter
    :param threshold: only pixels with values greater than or equal to the threshold will be considered
    :param use_gpu_filter:
    :return centers_guess_inds: np.array([[i0, i1, i2], ...]) array indices of local maxima
    """

    if use_gpu_filter:
        img_max_filtered = cp.asnumpy(
            cupyx.scipy.ndimage.maximum_filter(cp.asarray(imgs, dtype=cp.float32), footprint=cp.asarray(footprint)))
        # need to compare imgs as float32 because img_max_filtered will be ...
        centers_guess_inds = np.argwhere(np.logical_and(imgs.astype(np.float32) == img_max_filtered, imgs >= threshold))
    else:
        img_max_filtered = scipy.ndimage.maximum_filter(imgs, footprint=footprint)
        centers_guess_inds = np.argwhere(np.logical_and(imgs == img_max_filtered, imgs >= threshold))

    return centers_guess_inds


#
def combine_nearby_peaks(centers, min_xy_dist, min_z_dist, mode="average", weights=None):
    """
    Combine multiple peaks above threshold into reduced set, where assume all peaks separated by no more than
    min_xy_dist and min_z_dist come from the same feature.
    :param centers:
    :param min_xy_dist:
    :param min_z_dist:
    :param mode:
    :param weights:
    :return:
    """
    # todo: looks like this might be easier if use some tools from scipy.spatial, scipy.spatial.cKDTree

    centers_unique = np.array(centers, copy=True)
    inds = np.arange(len(centers), dtype=np.int)

    if weights is None:
        weights = np.ones(len(centers_unique))

    counter = 0
    while 1:
        # compute distances to all other beads
        z_dists = np.abs(centers_unique[counter][0] - centers_unique[:, 0])
        xy_dists = np.sqrt((centers_unique[counter][1] - centers_unique[:, 1]) ** 2 +
                           (centers_unique[counter][2] - centers_unique[:, 2]) ** 2)

        # beads which are close enough we will combine
        combine = np.logical_and(z_dists < min_z_dist, xy_dists < min_xy_dist)
        if mode == "average":
            denom = np.nansum(np.logical_not(np.isnan(np.sum(centers_unique[combine], axis=1))) * weights[combine])
            centers_unique[counter] = np.nansum(centers_unique[combine] * weights[combine][:, None], axis=0, dtype=np.float) / denom
            weights[counter] = denom
        elif mode == "keep-one":
            pass
        else:
            raise ValueError("mode must be 'average' or 'keep-one', but was '%s'" % mode)

        # remove all points from list except for one representative
        combine[counter] = False

        inds = inds[np.logical_not(combine)]
        centers_unique = centers_unique[np.logical_not(combine)]
        weights = weights[np.logical_not(combine)]

        counter += 1
        if counter >= len(centers_unique):
            break

    return centers_unique, inds


# localization functions
def localize_radial_symm(img, coords, mode="radial-symmetry"):
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


def fit_roi(img_roi, coords, init_params=None, fixed_params=None, bounds=None, sf=3):
    """
    Fit single ROI
    :param img_roi:
    :param coords: (z_roi, y_roi, x_roi)
    :param init_params:
    :param fixed_params:
    :param bounds:
    :return:
    """
    z_roi, y_roi, x_roi = coords

    to_use = np.logical_not(np.isnan(img_roi))
    x_roi_full, y_roi_full, z_roi_full = np.broadcast_arrays(x_roi, y_roi, z_roi)

    # get parameters from coordinates
    dc = x_roi[0, 0, 1] - x_roi[0, 0, 0]
    theta = np.arcsin((z_roi[0, 1, 0] - z_roi[0, 0, 0]) / dc)
    normal = np.array([0, -np.sin(theta), np.cos(theta)])

    if init_params is None:
        init_params = [None] * 7

    if np.any([ip is None for ip in init_params]):
        # set initial parameters
        min_val = np.nanmin(img_roi)
        img_roi -= min_val  # so will get ok values for moments
        mx1 = np.nansum(img_roi * x_roi) / np.nansum(img_roi)
        mx2 = np.nansum(img_roi * x_roi ** 2) / np.nansum(img_roi)
        my1 = np.nansum(img_roi * y_roi) / np.nansum(img_roi)
        my2 = np.nansum(img_roi * y_roi ** 2) / np.nansum(img_roi)
        sxy = np.sqrt(np.sqrt(my2 - my1 ** 2) * np.sqrt(mx2 - mx1 ** 2))
        mz1 = np.nansum(img_roi * z_roi) / np.nansum(img_roi)
        mz2 = np.nansum(img_roi * z_roi ** 2) / np.nansum(img_roi)
        sz = np.sqrt(mz2 - mz1 ** 2)
        img_roi += min_val  # put back to before

        ip_default = [np.nanmax(img_roi) - np.nanmean(img_roi), mx1, my1, mz1, sxy, sz, np.nanmean(img_roi)]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        # set bounds
        bounds = [[0, x_roi_full[to_use].min(), y_roi_full[to_use].min(), z_roi_full[to_use].min(), 0, 0, -np.inf],
                  [np.inf, x_roi_full[to_use].max(), y_roi_full[to_use].max(), z_roi_full[to_use].max(), np.inf, np.inf, np.inf]]

    # gaussian fitting localization
    def model_fn(p):
        return gaussian3d_pixelated_psf(x_roi, y_roi, z_roi, [dc, dc], normal, p, sf=sf)

    def jac_fn(p):
        return gaussian3d_pixelated_psf_jac(x_roi, y_roi, z_roi, [dc, dc], normal, p, sf=sf)

    # do fitting
    results = fit_model(img_roi, model_fn, init_params, bounds=bounds, fixed_params=fixed_params, model_jacobian=jac_fn)

    return results


def plot_roi(fit_params, roi, imgs, theta, x, y, z, center_guess=None, figsize=(16, 8),
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

    if center_guess is not None:
        if center_guess.ndim == 1:
            center_guess = center_guess[None, :]

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])
    normal = np.array([0, -np.sin(theta), np.cos(theta)])

    # get ROI and coordinates
    img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]

    vmin_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)
    vmax_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # git fit
    fit_volume = gaussian3d_pixelated_psf(x_roi, y_roi, z_roi, [dc, dc], normal, fit_params, sf=3)

    # interpolate on regular grid
    xi_roi, yi_roi, zi_roi, img_roi_unskew = interp_opm_data(img_roi, dc, dstage, theta, mode="ortho-interp")
    xi_roi += x_roi.min()
    dxi_roi = xi_roi[1] - xi_roi[0]
    yi_roi += y_roi.min()
    dyi_roi = yi_roi[1] - yi_roi[0]
    zi_roi += z_roi.min()
    dzi_roi = zi_roi[1] - zi_roi[0]

    # fit on regular grid
    fit_roi_unskew = gaussian3d_pixelated_psf(xi_roi[None, None, :], yi_roi[None, :, None], zi_roi[:, None, None]
                                              , [dc, dc * np.cos(theta)], np.array([0, 0, 1]), fit_params, sf=3)

    # ################################
    # plot results interpolated on regular grid
    # ################################
    figh_interp = plt.figure(figsize=figsize)
    plt.suptitle("Fit, max projections, interpolated\nA=%0.5g, cx=%0.5g, cy=%0.5g, cz=%0.5g, sxy=%0.5g, sz=%0.5g, bg=%0.5g" %
                 (fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], fit_params[6]))
    grid = plt.GridSpec(2, 3)

    ax = plt.subplot(grid[0, 0])
    plt.imshow(np.nanmax(img_roi_unskew, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi])
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    ax = plt.subplot(grid[0, 1])
    plt.imshow(np.nanmax(img_roi_unskew, axis=1), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 2], center_guess[:, 0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[0, 2])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        plt.imshow(np.nanmax(img_roi_unskew, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
                   extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    ax = plt.subplot(grid[1, 0])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi])
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 2], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")

    ax = plt.subplot(grid[1, 1])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=1), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[xi_roi[0] - 0.5 * dxi_roi, xi_roi[-1] + 0.5 * dxi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[2], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 2], center_guess[:, 0], 'gx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")

    ax = plt.subplot(grid[1, 2])
    plt.imshow(np.nanmax(fit_roi_unskew, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=[yi_roi[0] - 0.5 * dyi_roi, yi_roi[-1] + 0.5 * dyi_roi,
                       zi_roi[0] - 0.5 * dzi_roi, zi_roi[-1] + 0.5 * dzi_roi])
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if center_guess is not None:
        plt.plot(center_guess[:, 1], center_guess[:, 0], 'gx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")

    if save_dir is not None:
        figh_interp.savefig(os.path.join(save_dir, "%smax_projection.png" % prefix))
        plt.close(figh_interp)

    # ################################
    # plot fits in raw OPM coords
    # ################################
    figh_raw = plt.figure(figsize=figsize)
    plt.suptitle("ROI single PSF fit\nA=%0.5g, cx=%0.5g, cy=%0.5g, cz=%0.5g, sxy=%0.5g, sz=%0.5g, bg=%0.5g" %
                 (fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], fit_params[6]))
    grid = plt.GridSpec(3, roi[1] - roi[0])

    xp = np.arange(imgs.shape[2]) * dc
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
        plt.imshow(img_roi[jj], vmin=vmin_roi, vmax=vmax_roi, extent=extent_roi, origin="lower")
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
        plt.imshow(fit_volume[jj], vmin=vmin_roi, vmax=vmax_roi, extent=extent_roi, origin="lower")
        if jj != jj_min:
            plt.plot(xp, yp, 'mx')
        else:
            plt.plot(xp, yp, 'rx')

        if jj == 0:
            plt.ylabel("Fit\ny' (um)")
        else:
            ax.axes.yaxis.set_ticks([])

        ax = plt.subplot(grid[2, jj])
        plt.imshow(img_roi[jj] - fit_volume[jj], extent=extent_roi, origin="lower")
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
def localize(imgs, params, abs_threshold, roi_size, filter_sigma_small, filter_sigma_large,
             min_sep_allowed, sigma_bounds, offsets=(0, 0, 0), allowed_polygon=None, sf=3, use_max_filter=True,
             mode="fit", use_gpu_fit=GPUFIT_AVAILABLE, use_gpu_filter=CUPY_AVAILABLE):
    """
    :param imgs: raw OPM data
    :param params: {"dc", "dstage", "theta"}
    :param abs_threshold:
    :param roi_size: (sz, sy, sx) size to include in xyz directions for fit rois. Note: currently sy=sx and sx is unused
    :param filter_sigma_small: (sz, sy, sx) sigmas for small size filter to be used in difference of gaussian filter
    :param filter_sigma_large:
    :param min_sep_allowed: (dz, dy, dx) assume points separated by less than this distance come from one spot
    :param sigma_bounds: ((sz_min, sy_min, sx_min), (sz_max, sy_max, sx_max)) exclude fits with sigmas that fall outside
    these ranges
    :param offsets: offset to apply to y-coordinates. Useful for analyzing datasets in chunks
    :return:
    """
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
    imgs_hp = filter_convolve(imgs, ks, use_gpu=use_gpu_filter)
    imgs_lp = filter_convolve(imgs, kl, use_gpu=use_gpu_filter)
    imgs_filtered = imgs_hp - imgs_lp

    tend = time.perf_counter()
    print("Filtered images in %0.2fs" % (tend - tstart))

    # ###################################################
    # mask off region of each camera frame
    # ###################################################
    tstart = time.perf_counter()

    if allowed_polygon is None:
        mask = np.expand_dims(np.ones(imgs_filtered[0].shape, dtype=np.bool), axis=0)
    else:
        p = Path(allowed_polygon)
        xx, yy = np.meshgrid(range(imgs.shape[2]), range(imgs.shape[1]))
        rs = np.concatenate((xx.ravel()[:, None], yy.ravel()[:, None]), axis=1)
        mask = p.contains_points(rs).reshape([1, imgs.shape[1], imgs.shape[2]])
    tend = time.perf_counter()
    print("Masked region in %0.2fs" % (tend - tstart))

    # ###################################################
    # identify candidate beads
    # ###################################################
    tstart = time.perf_counter()
    if False:
        centers_guess_inds = skimage.feature.peak_local_max(imgs_filtered * mask, min_distance=1, threshold_abs=abs_threshold,
                                            exclude_border=False, num_peaks=np.inf)
    elif False:
        ispeak = imgs_filtered * mask > abs_threshold

        # get indices of points above threshold
        coords = np.meshgrid(*[range(imgs.shape[ii]) for ii in range(imgs.ndim)], indexing="ij")
        centers_guess_inds = np.concatenate([c[ispeak][:, None] for c in coords], axis=1)
    else:
        footprint = get_skewed_footprint(min_sep_allowed, dc, stage_step, theta)
        centers_guess_inds = find_peak_candidates(imgs_filtered * mask, footprint, abs_threshold, use_gpu_filter=use_gpu_filter)

        # footprint_roi_size = get_skewed_roi_size(min_sep_allowed, theta, dc, stage_step, ensure_odd=True)
        # footprint_form = np.ones(footprint_roi_size, dtype=np.bool)
        # xf, yf, zf = get_skewed_coords(footprint_form.shape, dc, stage_step, theta)
        # xf = xf - xf.mean()
        # yf = yf - yf.mean()
        # zf = zf - zf.mean()
        # footprint_mask = get_roi_mask((0, 0, 0), min_sep_allowed, (zf, yf, xf))
        # footprint_mask[np.isnan(footprint_mask)] = 0
        # footprint_mask = footprint_mask.astype(np.bool)
        # if use_gpu_filter:
        #     imgs_filtered = imgs_filtered.astype(np.float32)
        #     img_max_filtered = cp.asnumpy(cupyx.scipy.ndimage.maximum_filter(cp.asarray(imgs_filtered * mask, dtype=cp.float32),
        #                                                                      footprint=cp.asarray(footprint_form * footprint_mask)))
        # else:
        #     img_max_filtered = scipy.ndimage.maximum_filter(imgs_filtered * mask, footprint=footprint_form * footprint_mask)
        #
        # centers_guess_inds = np.argwhere(np.logical_and(imgs_filtered == img_max_filtered, imgs_filtered > abs_threshold))


    # convert to xyz coordinates
    xc = x[0, 0, centers_guess_inds[:, 2]]
    yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
    zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
    centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

    tend = time.perf_counter()
    print("Found %d points above threshold in %0.2fs" % (len(centers_guess), tend - tstart))

    # ###################################################
    # average multiple points too close together. Necessary bc if naive threshold, may identify several points
    # from same spot. Particularly important if spots have very different brightness levels.
    # ###################################################
    tstart = time.perf_counter()

    inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
    weights = imgs_filtered.ravel()[inds]
    centers_guess, _ = combine_nearby_peaks(centers_guess, min_sep_allowed[1], min_sep_allowed[0], weights=weights, mode="average")

    # remove point outside boundaries
    in_region = point_in_trapezoid(centers_guess, x, y, z)
    centers_guess = centers_guess[in_region]

    tend = time.perf_counter()
    print("Found %d points separated by dxy > %0.5g and dz > %0.5g and in allowed region in %0.1fs" %
          (len(centers_guess), min_sep_allowed[1], min_sep_allowed[0], tend - tstart))

    tstart = time.perf_counter()
    # get rois and coordinates for each center
    roi_size_skew = get_skewed_roi_size(roi_size, theta, dc, stage_step, ensure_odd=True)
    rois, xrois, yrois, zrois = zip(*[get_skewed_roi(c, x, y, z, roi_size_skew, imgs.shape) for c in centers_guess])
    rois = np.asarray(rois)

    # exclude some regions of roi
    roi_masks = [get_roi_mask(c, (np.inf, 0.5 * roi_size[1]), (zrois[ii], yrois[ii], xrois[ii])) for ii, c in enumerate(centers_guess)]
    tend = time.perf_counter()
    print("Prepared %d rois in %0.2fs" % (len(rois), tend - tstart))

    # ###################################################
    # localization
    # ###################################################
    if mode == "fit":
        print("starting fitting for %d rois" % centers_guess.shape[0])
        tstart = time.perf_counter()

        if use_gpu_fit:

            # exclude rois near edge
            roi_sizes = np.array([(r[1] - r[0]) * (r[3] - r[2]) * (r[5] - r[4]) for r in rois])
            roi_masks = [r.astype(np.bool) for r in roi_masks]
            roi_mask_sizes = np.array([np.sum(r) for r in roi_masks])
            nmax = np.max(roi_mask_sizes)

            print("starting fitting in parallel on GPU")
            imgs_roi = [cut_roi(r, imgs)[rm].astype(np.float32)[None, :] for r, rm in zip(rois, roi_masks)]
            imgs_roi = [np.pad(im, ((0, 0), (0, nmax - im.size)), mode="constant") for im in imgs_roi]

            data = np.concatenate(imgs_roi, axis=0)
            data = data.astype(np.float32)

            coords = [np.broadcast_arrays(x, y, z) for x, y, z in zip(xrois, yrois, zrois)]
            user_info = [(c[0][rm], c[1][rm], c[2][rm]) for c, rm in zip(coords, roi_masks)]
            user_info = [np.concatenate((np.pad(c[0], (0, nmax - c[0].size), mode="constant"),
                                         np.pad(c[1], (0, nmax - c[1].size), mode="constant"),
                                         np.pad(c[2], (0, nmax - c[1].size), mode="constant"))) for c in user_info]
            user_info = np.concatenate(user_info).astype(np.float32)

            user_info = np.concatenate((user_info, roi_mask_sizes.astype(np.float32)))

            nfits, n_pts_per_fit = data.shape
            init_params = np.concatenate((100 * np.ones((nfits, 1)), centers_guess[:, 2][:, None],
                                          centers_guess[:, 1][:, None], centers_guess[:, 0][:, None],
                                          0.14 * np.ones((nfits, 1)), 0.4 * np.ones((nfits, 1)), 0 * np.ones((nfits, 1))), axis=1)
            init_params = init_params.astype(np.float32)

            if data.ndim != 2:
                raise ValueError
            if init_params.ndim != 2 or init_params.shape != (nfits, 7):
                raise ValueError
            if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
                raise ValueError

            fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data, None, gf.ModelID.GAUSS_3D_ARB, init_params,
                                                                     max_number_iterations=100,
                                                                     estimator_id=gf.EstimatorID.LSE,
                                                                     user_info=user_info)
        else:
            print("using multiprocessor parallelization on CPU")
            results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                      joblib.delayed(fit_roi)(cut_roi(rois[ii], imgs_filtered) * roi_masks[ii], (zrois[ii], yrois[ii], xrois[ii]), sf=sf,
                                              init_params=[None, centers_guess[ii, 2], centers_guess[ii, 1], centers_guess[ii, 0], None, None, None])
                      for ii in range(len(centers_guess)))
            # results = []
            # for ii in range(len(centers_guess)):
            #     print(ii)
            #     results.append(fit_roi(get_roi(rois[ii], imgs) * roi_masks[ii], (zrois[ii], yrois[ii], xrois[ii]), sf=sf,
            #     init_params=[None, centers_guess[ii, 2], centers_guess[ii, 1], centers_guess[ii, 0], None, None, None]))

            fit_params = np.asarray([r["fit_params"] for r in results])

    elif mode == "radial-symmetry":
        print("starting radial-symmetry localization for %d rois" % len(centers_guess))

        img_roi_masked = [cut_roi(rois[ii], imgs_filtered) * roi_masks[ii] for ii in range(len(centers_guess))]
        for ii in range(len(img_roi_masked)):
            img_roi_masked[ii][img_roi_masked[ii] < 0] = 0

        results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                  joblib.delayed(localize_radial_symm)(img_roi_masked[ii], (zrois[ii], yrois[ii], xrois[ii]),
                                                       mode="radial-symmetry")
                  for ii in range(len(centers_guess)))

        fit_params = np.asarray(results)

        # tstart = time.process_time()
        # fit_params = np.zeros((len(rois), 7))
        # for ii in range(len(rois)):
        #     img_roi = get_roi(rois[ii], imgs_filtered) * roi_masks[ii]
        #     img_roi[img_roi < 0] = 0
        #     fit_params[ii] = localize_radial_symm(img_roi, (zrois[ii], yrois[ii], xrois[ii]), mode="radial-symmetry")

        # tend = time.process_time()
        # print("radial-symmetry localzation for %d rois in %0.2fs" % (centers_guess.shape[0], tend - tstart))
    else:
        raise ValueError
    tend = time.perf_counter()
    print("Localization took %0.2fs" % (tend - tstart))

    tstart = time.perf_counter()
    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]), axis=1)

    # ###################################################
    # only keep points if size and position were reasonable
    # ###################################################
    amp_min = 0.1 * abs_threshold
    sigmas_min_keep = sigma_bounds[0]
    sigmas_max_keep = sigma_bounds[1]
    to_keep = np.logical_and.reduce((point_in_trapezoid(centers_fit, x, y, z),
                                     fit_params[:, 0] >= amp_min,
                                     fit_params[:, 4] <= sigmas_max_keep[2],
                                     fit_params[:, 4] >= sigmas_min_keep[2],
                                     fit_params[:, 4] <= sigmas_max_keep[1],
                                     fit_params[:, 4] >= sigmas_min_keep[1],
                                     fit_params[:, 5] <= sigmas_max_keep[0],
                                     fit_params[:, 5] >= sigmas_min_keep[0]))

    tend = time.perf_counter()
    print("identified %d valid localizations with:\n"
          "centers in bounds, amp >= %.5g\n"
          "%0.5g <= sx <= %0.5g, %0.5g <= sy <= %0.5g and %0.5g <= sz <= %0.5g in %0.3f" %
          (np.sum(to_keep), amp_min, sigmas_min_keep[2], sigmas_max_keep[2],
           sigmas_min_keep[1], sigmas_max_keep[1],
           sigmas_min_keep[0], sigmas_max_keep[0], tend - tstart))

    if np.sum(to_keep) > 0:
        tstart = time.perf_counter()
        # only keep unique center if close enough
        centers_unique, unique_inds = combine_nearby_peaks(centers_fit[to_keep], min_sep_allowed[1], min_sep_allowed[0], mode="keep-one")
        fit_params_unique = fit_params[to_keep][unique_inds]
        rois_unique = rois[to_keep][unique_inds]
        tend = time.perf_counter()
        print("identified %d unique points, dxy > %0.5g, and dz > %0.5g in %0.3f" %
              (len(centers_unique), min_sep_allowed[1], min_sep_allowed[0], tend - tstart))
    else:
        centers_unique = np.zeros((0, 3))
        fit_params_unique = np.zeros((0, 7))
        rois_unique = np.zeros((0, 6))

    return imgs_filtered, centers_unique, fit_params_unique, rois_unique, centers_guess
