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
import cupy as cp
import cupyx.scipy.signal # new in v9, compiled github source

# misc helper functions
def nearest_pt_line(pt, slope, pt_line):
    """
    Get shortest distance between a point and a line.
    :param pt: (xo, yo), point of itnerest
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


# coordinate transformations between OPM and coverslip frames
def get_lab_coords(nx_cam, ny_cam, dc, theta, stage_pos):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param nx_cam:
    :param ny_cam:
    :param dc: camera pixel size
    :param theta:
    :param stage_pos: list of y-displacements for each scan position
    :return:
    """
    x = dc * np.arange(nx_cam)[None, None, :]
    y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
    z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]

    return x, y, z


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


def find_trapezoid_cz(cy, coords):
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


def find_trapezoid_cy(cz, coords):
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

def interp_opm_data(imgs, dc, ds, theta, mode="row-interp"):
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
        raise ValueError("ds = (dx, dy) length as not 2")

    # generate new points in pixel
    pts = np.arange(1 / sf / 2, 1 - 1 / sf / 2, 1 / sf) - 0.5
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
    pts = np.arange(1 / sf / 2, 1 - 1 / sf / 2, 1 / sf) - 0.5
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


def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    # todo: to be fully general, maybe should get rid of the img argument and only take model_fn. Then this function
    # todo: can be used as a wrapper for least_squares, but adding the ability to fix parameters
    # todo: added function below fit_least_squares(). This function should now call that one
    Fit 2D model function
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
def simulate_img(scan_params, physical_params, ncenters=1, sf=3):

    # size and pixel size
    dc = scan_params["dc"]
    theta = scan_params["theta"]
    dstep = scan_params["dstep"]
    npos, ny, nx = scan_params["shape"]

    normal = np.array([0, -np.sin(theta), np.cos(theta)])  # normal of camera pixel
    stage_pos = dstep * np.arange(npos)

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
    x, y, z = get_lab_coords(nx, ny, dc, theta, stage_pos)

    # define centers
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
def get_roi(roi, img):
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


def get_roi_size(sizes, theta, dc, dstep, ensure_odd=True):
    """
    Get ROI size in OPM matrix that includes sufficient xy and z points

    :param sizes: [z-size, y-size, x-size]
    :param theta:
    :param dc:
    :param dstep:
    :param bool ensure_odd:
    :return:
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


def get_tilted_roi(center, x, y, z, sizes, shape):
    """

    :param center: [cz, cy, cx]
    :param x:
    :param y:
    :param z:
    :param sizes: [n0, n1, n2]
    :param shape: shape of full image
    :return:
    """
    i2 = np.argmin(np.abs(x.ravel() - center[2]))
    i0, i1, _ = np.unravel_index(np.argmin((y - center[1]) ** 2 + (z - center[0]) ** 2), y.shape)
    roi = np.array(get_centered_roi([i0, i1, i2], sizes, min_vals=[0, 0, 0], max_vals=np.array(shape)))

    x_roi = x[:, :, roi[4]:roi[5]]  # only roi on last one because x has only one entry on first two dims
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]

    return roi, x_roi, y_roi, z_roi


def get_roi_mask(center, xy_size, z_size, x_roi, y_roi, z_roi):
    """
    Get mask to exclude points in the ROI that are far from the center. We do not want to include regions at the edges
    of the trapezoidal ROI in processing.

    :param center:
    :param xy_size:
    :param z_size:
    :param x_roi:
    :param y_roi:
    :param z_roi:
    :return:
    """
    x_roi_full, y_roi_full, z_roi_full = np.broadcast_arrays(x_roi, y_roi, z_roi)
    mask = np.ones(x_roi_full.shape)

    # roi is parallelogram, so still want to cut out points which are too far from center
    too_far_xy = np.sqrt((x_roi - center[2]) ** 2 + (y_roi - center[1]) ** 2) > xy_size
    too_far_z = np.abs(z_roi - center[0]) > z_size
    too_far = np.logical_or(too_far_xy, too_far_z)

    mask[too_far] = np.nan

    return mask

# fitting ROI
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


def iterative_fit_roi(c_guess, imgs, dc, theta, x, y, z, xy_roi_size, z_roi_size, center_max_dist_good_guess, nmax_try=3):
    """
    Fit ROI until center converges
    :param imgs:
    :param c_guess:
    :param x:
    :param y:
    :param z:
    :param xy_roi_size: required xy half distance to be included in ROI
    :param z_roi_size: required z half distance to be included in ROI
    :param center_max_dist_good_guess:
    :param nmax_try:
    :return:
    """
    dstep = y[1, 0, 0] - y[0, 0, 0]
    nzp, nyp, nxp = get_roi_size([z_roi_size, xy_roi_size, xy_roi_size], theta, dc, dstep, ensure_odd=True)

    centers_sequence = np.zeros((nmax_try, 3)) * np.nan
    centers_sequence[0] = c_guess

    ntry = 0
    while 1:
        # get ROI from center guess
        roi, x_roi, y_roi, z_roi = get_tilted_roi(centers_sequence[ntry], x, y, z, [nzp, nyp, nxp], imgs.shape)
        img_roi = np.array(imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]], dtype=np.float)

        # roi is parallelogram, so still want to cut out points which are too far from center
        too_far_xy = np.sqrt((x_roi - centers_sequence[ntry, 2]) ** 2 + (y_roi - centers_sequence[ntry, 1]) ** 2) > 0.5 * xy_roi_size
        img_roi[too_far_xy] = np.nan

        # e.g. if point is outside the image
        if not point_in_trapezoid(centers_sequence[ntry], x_roi, y_roi, z_roi) and ntry > 1:
            break

        init_params = [None, centers_sequence[ntry, 2], centers_sequence[ntry, 1], centers_sequence[ntry, 0], None, None, None]
        # gaussian fitting localization
        results = fit_roi(img_roi, (z_roi, y_roi, x_roi), init_params=init_params)

        # if fit center is close enough to guess, assume ROI was a good choice and move on
        # if not, update ROI and try again
        c_fit = np.array([results["fit_params"][3], results["fit_params"][2], results["fit_params"][1]])
        c_fit_guess_dist = np.sqrt(np.sum((c_fit - centers_sequence[ntry]) ** 2))

        # if exceeded maximum number of tries or fit is high quality, break out of loop
        ntry += 1
        if ntry >= nmax_try or c_fit_guess_dist < center_max_dist_good_guess:
            break
        else:
            # otherwise, update center guess and re-fit
            centers_sequence[ntry] = c_fit
    # except:
    #     results = {"fit_params", np.zeros(7)}

    # store results
    fit_params = results["fit_params"]

    return fit_params, ntry, roi, centers_sequence


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

# localization functions
def find_candidate_beads(img, filter_xy_pix=1, filter_z_pix=0.5, min_distance=1, abs_thresh_std=1,
                         max_thresh=-np.inf, max_num_peaks=np.inf,
                         mode="max_filter"):
    """
    Find candidate beads in image. Based on function from mesoSPIM-PSFanalysis

    :param img: 2D or 3D image
    :param filter_xy_pix: standard deviation of Gaussian filter applied to image in xy plane before peak finding
    :param filter_z_pix:
    :param min_distance: minimum allowable distance between peaks
    :param abs_thresh_std: absolute threshold for identifying peaks, as a multiple of the image standard deviation
    :param abs_thresh: absolute threshold, in raw counts. If both abs_thresh_std and abs_thresh are provided, the
    maximum value will be used
    :param max_num_peaks: maximum number of peaks to find.

    :return centers: np.array([[cz, cy, cx], ...])
    """

    # gaussian filter to smooth image before peak finding
    if img.ndim == 3:
        filter_sds = [filter_z_pix, filter_xy_pix, filter_xy_pix]
    elif img.ndim == 2:
        filter_sds = filter_xy_pix
    else:
        raise ValueError("img should be a 2 or 3 dimensional array, but was %d dimensional" % img.ndim)

    smoothed = skimage.filters.gaussian(img, filter_sds, output=None, mode='nearest', cval=0,
                                        multichannel=None, preserve_range=True)

    # set threshold value
    abs_threshold = np.max([smoothed.mean() + abs_thresh_std * img.std(), max_thresh])

    if mode == "max_filter":
        centers = skimage.feature.peak_local_max(smoothed, min_distance=min_distance, threshold_abs=abs_threshold,
                                                 exclude_border=False, num_peaks=max_num_peaks)
    elif mode == "threshold":
        ispeak = smoothed > abs_threshold
        # get indices of points above threshold
        coords = np.meshgrid(*[range(img.shape[ii]) for ii in range(img.ndim)], indexing="ij")
        centers = np.concatenate([c[ispeak][:, None] for c in coords], axis=1)
    else:
        raise ValueError("mode must be 'max_filter', or 'threshold', but was '%s'" % mode)

    return centers


def filter_gauss(imgs, sigmas, dc, theta, dstage, sigma_cutoff=2, use_gpu=True):
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

    # normalize everything to camera pixel size
    sigma_x_pix = sigmas[2] / dc
    sigma_y_pix = sigmas[2] / dc
    sigma_z_pix = sigmas[0] / dc
    nk_x = 2 * int(np.round(sigma_x_pix * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(sigma_y_pix * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(sigma_z_pix * sigma_cutoff)) + 1
    # determine how large the OPM geometry ROI needs to be to fit the desired filter
    roi_sizes = get_roi_size([nk_z, nk_y, nk_x], theta, 1, dstage / dc, ensure_odd=True)

    # get coordinates to evaluate kernel at
    xk, yk, zk = get_lab_coords(roi_sizes[2], roi_sizes[1], 1, theta, dstage / dc * np.arange(roi_sizes[0]))
    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kernel = np.exp(-xk ** 2 / 2 / sigma_x_pix ** 2 - yk ** 2 / 2 / sigma_y_pix ** 2 - zk ** 2 / 2 / sigma_z_pix ** 2)
    kernel = kernel / np.sum(kernel)

    # convolve, and deal with edges by normalizing
    # todo: is there a wrap around issue, where this blends opposite edges?
    # todo: can replace this with CuPy
    if use_gpu:
        kernel_cp = cp.asarray(kernel)
        imgs_cp = cp.asarray(imgs)
        imgs_filtered = cp.asnumpy(cupyx.scipy.signal.fftconvolve(imgs_cp, kernel_cp, mode="same")/
                                   cupyx.scipy.signal.fftconvolve(cp.ones(imgs.shape), kernel_cp, mode="same"))
    else:
        imgs_filtered = scipy.signal.fftconvolve(imgs, kernel, mode="same") / \
             scipy.signal.fftconvolve(np.ones(imgs.shape), kernel, mode="same")

    # this method too slow for large fitler sizes
    # imgs_filtered = scipy.ndimage.convolve(imgs, kernel, mode="constant", cval=0) / \
    #                 scipy.ndimage.convolve(np.ones(imgs.shape), kernel, mode="constant", cval=0)

    return imgs_filtered, kernel

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

def point_in_trapezoid(pts, x, y, z):
    """
    Test if a point is in the trapzoidal region described by x,y,z
    :param pts:
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

def localize_radial_symm(img, coords, mode="radial-symmetry"):
    # todo: check quality of localizations
    if img.ndim != 3:
        raise ValueError("img must be a 3D array, but was %dD" % img.ndim)

    nstep, ni1, ni2 = img.shape
    z, y, x = coords
    # x, y, z = get_lab_coords(ni2, ni1, dc, theta, stage_step * np.arange(nstep))

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

def localize(imgs, params, abs_threshold, roi_size, filter_sigma_small, filter_sigma_large,
             min_sep_allowed, sigma_bounds, y_offset=0, allowed_polygon=None, sf=3,
             mode="fit"):
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
    :param nmax_try: maximum number of times to try fitting a single ROI # todo: git rid of this...
    :param y_offset: offset to apply to y-coordinates. Useful for analyzing datasets in chunks
    :return:
    """

    # set up geometry
    npos, ny, nx = imgs.shape

    dc = params["dc"]
    theta = params["theta"]
    stage_step = params["dstep"]
    stage_pos = stage_step * np.arange(npos)

    x, y, z = get_lab_coords(nx, ny, dc, theta, stage_pos)
    y += y_offset

    # smooth image and remove background with difference of gaussians filter
    tstart = time.process_time()

    imgs_filtered_small, ks = filter_gauss(imgs, filter_sigma_small, dc, theta, stage_step, sigma_cutoff=2)
    imgs_filtered_large, kl = filter_gauss(imgs, filter_sigma_large, dc, theta, stage_step, sigma_cutoff=2)
    imgs_filtered = imgs_filtered_small - imgs_filtered_large

    tend = time.process_time()
    print("Filtered images in %0.2fs" % (tend - tstart))

    # mask will filter out some points
    if allowed_polygon is None:
        mask = np.expand_dims(np.ones(imgs_filtered[0].shape), axis=0)
    else:
        p = Path(allowed_polygon)
        xx, yy = np.meshgrid(range(imgs.shape[2]), range(imgs.shape[1]))
        rs = np.concatenate((xx.ravel()[:, None], yy.ravel()[:, None]), axis=1)
        mask = p.contains_points(rs).reshape([1, imgs.shape[1], imgs.shape[2]])

    tstart = time.process_time()
    # identify points above threshold (either naively or using max filter)
    if False:
        ispeak = imgs_filtered * mask > abs_threshold

        # get indices of points above threshold
        coords = np.meshgrid(*[range(imgs.shape[ii]) for ii in range(imgs.ndim)], indexing="ij")
        centers_guess_inds = np.concatenate([c[ispeak][:, None] for c in coords], axis=1)
    else:
        centers_guess_inds = skimage.feature.peak_local_max(imgs_filtered * mask, min_distance=1, threshold_abs=abs_threshold,
                                            exclude_border=False, num_peaks=np.inf)

    # convert to xyz coordinates
    xc = x[0, 0, centers_guess_inds[:, 2]]
    yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
    zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
    centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)

    tend = time.process_time()
    print("Found %d points above threshold in %0.2fs" % (len(centers_guess), tend - tstart))

    # average multiple points too close together. Necessary bc if naive threshold, may identify several points
    # from same spot. Particularly important if spots have very different brightness levels.
    tstart = time.process_time()

    inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
    weights = imgs_filtered.ravel()[inds]
    centers_guess, _ = combine_nearby_peaks(centers_guess, min_sep_allowed[1], min_sep_allowed[0], weights=weights, mode="average")

    # remove point outside boundaries
    in_region = point_in_trapezoid(centers_guess, x, y, z)
    centers_guess = centers_guess[in_region]

    tend = time.process_time()
    print("Found %d points separated by dxy > %0.5g and dz > %0.5g and in allowed region in %0.1fs" %
          (len(centers_guess), min_sep_allowed[1], min_sep_allowed[0], tend - tstart))

    # get rois and coordinates for each center
    roi_size_skew = get_roi_size(roi_size, theta, dc, stage_step, ensure_odd=True)
    rois, xrois, yrois, zrois = zip(*[get_tilted_roi(c, x, y, z, roi_size_skew, imgs.shape) for c in centers_guess])
    rois = np.asarray(rois)
    # exclude some regions of roi
    roi_masks = [get_roi_mask(c, 0.5 * roi_size[1], np.inf, xrois[ii], yrois[ii], zrois[ii]) for ii, c in enumerate(centers_guess)]

    if mode == "fit":
        print("starting fitting for %d rois" % centers_guess.shape[0])
        results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
                  joblib.delayed(fit_roi)(get_roi(rois[ii], imgs_filtered) * roi_masks[ii], (zrois[ii], yrois[ii], xrois[ii]), sf=sf,
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

        img_roi_masked = [get_roi(rois[ii], imgs_filtered) * roi_masks[ii] for ii in range(len(centers_guess))]
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

    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]), axis=1)

    # only keep points if size and position were reasonable
    sigmas_min_keep = sigma_bounds[0]
    sigmas_max_keep = sigma_bounds[1]
    to_keep = np.logical_and.reduce((point_in_trapezoid(centers_fit, x, y, z),
                                    fit_params[:, 4] <= sigmas_max_keep[2],
                                    fit_params[:, 4] >= sigmas_min_keep[2],
                                    fit_params[:, 4] <= sigmas_max_keep[1],
                                    fit_params[:, 4] >= sigmas_min_keep[1],
                                    fit_params[:, 5] <= sigmas_max_keep[0],
                                    fit_params[:, 5] >= sigmas_min_keep[0]))

    print("identified %d valid localizations with centers in bounds,"
          " %0.5g <= sx <= %0.5g, %0.5g <= sy <= %0.5g and %0.5g <= sz <= %0.5g " %
          (np.sum(to_keep), sigmas_min_keep[2], sigmas_max_keep[2], sigmas_min_keep[1], sigmas_max_keep[1],
           sigmas_min_keep[0], sigmas_max_keep[0]))

    # only keep unique center if close enough
    centers_unique, unique_inds = combine_nearby_peaks(centers_fit[to_keep], min_sep_allowed[1], min_sep_allowed[0], mode="keep-one")
    fit_params_unique = fit_params[to_keep][unique_inds]
    rois_unique = rois[to_keep][unique_inds]
    print("identified %d unique points, dxy > %0.5g, and dz > %0.5g" %
          (len(centers_unique), min_sep_allowed[1], min_sep_allowed[0]))

    return imgs_filtered, centers_unique, fit_params_unique, rois_unique, centers_guess

