"""
2/5/2021, Peter T. Brown
"""
import numpy as np
import scipy
from scipy import fft
import scipy.sparse as sp
import skimage.feature
import skimage.filters
import matplotlib.pyplot as plt
import warnings

# misc helper functions
def get_centered_roi(centers, sizes):
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: list of centers [c1, c2, ..., cn]
    :param sizes: list of sizes [s1, s2, ..., sn]
    :return roi: [start_0, end_0, start_1, end_1, ..., start_n, end_n]
    """
    roi = []
    for c, n in zip(centers, sizes):

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

        roi.append(int(start))
        roi.append(int(end))

    return roi

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

    x_int = (xo + slope * (yo - b)) / (slope**2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xo - x_int)**2 + (yo - y_int)**2)

    return (x_int, y_int), d

# coordinate transformations between OPM and coverslip frames
def get_lab_coords(nx, dx, theta, gn):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param nx:
    :param dx:
    :param theta:
    :param gn: list of y-displacements for each scan position
    :return:
    """
    x = dx * np.arange(nx)[None, None, :]
    y = gn[:, None, None] + dx * np.cos(theta) * np.arange(nx)[None, :, None]
    z = dx * np.sin(theta) * np.arange(nx)[None, :, None]

    return x, y, z

def lab2cam(x, y, z, theta):
    """
    Get camera coordinates.
    :param x:
    :param y:
    :param z:
    :param theta:

    :return xp:
    :return yp: yp coordinate
    :return gn: distance of leading edge of camera frame from the y-axis, measured along the z-axis
    """
    xp = x
    gn = y - z / np.tan(theta)
    yp = (y - gn) / np.cos(theta)
    return xp, yp, gn

def interp_opm_data(imgs, dc, ds, theta, mode="row-interp"):
    """
    Interpolate OPM stage-scan data to be equally spaced in coverslip frame

    :param imgs: nz x ny x nx
    :param dc: image spacing in camera space, i.e. camera pixel size reference to object space
    :param ds: distance stage moves between frames
    :param theta:
    :return:
    """
    # ds/ (dx * np.cos(theta) ) should be an integer.
    # todo: relax this constraint if ortho-interp is used
    step_size = int(ds / (dc * np.cos(theta)))

    # fix y-positions from raw images
    nx = imgs.shape[2]
    nyp = imgs.shape[1]
    nimgs = imgs.shape[0]

    # interpolated sizes
    x = dc * np.arange(0, nx)
    y = dc * np.cos(theta) * np.arange(0, nx + step_size * nimgs)
    z = dc * np.sin(theta) * np.arange(0, nyp)
    ny = len(y)
    nz = len(z)
    # interpolated sampling spacing
    dx = dc
    dy = dc * np.cos(theta)
    dz = dc * np.sin(theta)

    img_unskew = np.nan * np.zeros((z.size, y.size, x.size))

    # todo: using loops for a start ... optimize later
    if mode == "row-interp": # interpolate using nearest two points on same row
        for ii in range(nz): # loop over z-positions
            for jj in range(nimgs): # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size): # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size], z[ii])
                        img_unskew[ii, ii + jj * step_size + kk, :] = imgs[jj, ii, :] * (step_size - kk) / step_size + \
                                                                      imgs[jj + 1, ii, :] * kk / step_size
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]

    # todo: this mode can be generalized to not use dy a multiple of dx
    elif mode == "ortho-interp": # interpolate using nearest four points.
        for ii in range(nz):  # loop over z-positions
            for jj in range(nimgs):  # loop over large y-position steps (moving distance btw two real frames)
                if jj < (nimgs - 1):
                    for kk in range(step_size):  # loop over y-positions in between two frames
                        # interpolate to estimate value at point (y, z) = (y[ii + jj * step_size + kk], z[ii])
                        pt_now = (y[ii + jj * step_size + kk], z[ii])

                        # find nearest point on line passing through (y[jj * step_size], 0)
                        pt_n1, dist_1 = nearest_pt_line(pt_now, np.tan(theta), (y[jj * step_size], 0))
                        dist_along_line1 = np.sqrt( (pt_n1[0] - y[jj * step_size])**2 + pt_n1[1]**2) / dc
                        # as usual, need to round to avoid finite precision floor/ceiling issues if number is already an integer
                        i1_low = int(np.floor(np.round(dist_along_line1, 14)))
                        i1_high = int(np.ceil(np.round(dist_along_line1, 14)))

                        if np.round(dist_1, 14) == 0:
                            q1 = imgs[jj, i1_low, :]
                        elif i1_low < 0 or i1_high >= nyp:
                            q1 = np.nan
                        else:
                            d1 = dist_along_line1 - i1_low
                            q1 = (1 - d1) * imgs[jj, i1_low, :] + d1 * imgs[jj, i1_high, :]

                        # find nearest point on line passing through (y[(jj + 1) * step_size], 0)
                        pt_no, dist_o = nearest_pt_line(pt_now, np.tan(theta), (y[(jj + 1) * step_size], 0))
                        dist_along_line0 = np.sqrt((pt_no[0] - y[(jj + 1) * step_size]) ** 2 + pt_no[1] ** 2) / dc
                        io_low = int(np.floor(np.round(dist_along_line0, 14)))
                        io_high = int(np.ceil(np.round(dist_along_line0, 14)))

                        if np.round(dist_o, 14) == 0:
                            qo = imgs[jj + 1, i1_low, :]
                        elif io_low < 0 or io_high >= nyp:
                            qo = np.nan
                        else:
                            do = dist_along_line0 - io_low
                            qo = (1 - do) * imgs[jj + 1, io_low, :] + do * imgs[jj + 1, io_high, :]

                        # weighted average of qo and q1 based on their distance
                        img_unskew[ii, ii + jj * step_size + kk, :] = (q1 * dist_o + qo * dist_1) / (dist_o + dist_1)
                else:
                    img_unskew[ii, ii + jj * step_size, :] = imgs[jj, ii, :]
    else:
        raise Exception("mode must be 'row-interp' or 'ortho-interp' but was '%s'" % mode)

    return x, y, z, img_unskew

# point spread function model and fitting
def gaussian3d_pixelated_psf(x, y, z, ds, normal, p, sf=3):
    """
    Gaussian function, accounting for image pixelation in the xy plane. This function mimics the style of the
    PSFmodels functions.

    vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

    :param dx: pixel size in um
    :param nx: number of pixels (must be odd)
    :param z: coordinates of z-planes to evaluate function at
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param wavelength: in um
    :param ni: refractive index
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :return:
    """

    # generate new points in pixel
    pts = ds * (np.arange(1 / sf / 2, 1 - 1 / sf /2, 1 / sf) - 0.5)
    xp, yp = np.meshgrid(pts, pts)
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

    psf_s = np.exp( - (xx_s - p[1])**2 / 2 / p[4]**2 - (yy_s - p[2])**2 / 2/ p[4]**2 - (zz_s - p[3])**2 / 2/ p[5]**2)
    norm = np.sum(np.exp(-xs**2 / 2 / p[4]**2 - ys**2 / 2 / p[5]**2))

    psf = p[0] / norm * np.sum(psf_s, axis=-1) + p[-1]

    return psf

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

    # init_params = copy.deepcopy(init_params)
    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, but don't touch if parameter is fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            if bounds[0][ii] == -np.inf:
                init_params[ii] = bounds[0][ii] + 1
            elif bounds[1][ii] == np.inf:
                init_params[ii] = bounds[1][ii] - 1
            else:
                init_params[ii] = 0.5 * (bounds[0][ii] + bounds[1][ii])

    if np.any(np.isnan(init_params)):
        raise ValueError("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise ValueError("bounds cannot include nans")

    def err_fn(p): return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())
    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # awful list comprehension. The idea is this: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]
    def pfree2pfull(pfree): return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])
    # map full parameters to reduced set
    def pfull2pfree(pfull): return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree): return err_fn(pfree2pfull(pfree))
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

# simulated image
def bin(img, bin_size, mode='sum'):
    """
    bin image by combining adjacent pixels

    In 1D, this is a straightforward problem. The image is a vector,
    I = (I[0], I[1], ..., I[nx-1])
    and the binning operator is a nx/nx_bin x nx matrix
    M = [[1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
         [0, 0, ..., 0, 1, ..., 1, 0, ..., 0]
         ...
         [0, ...,              0,  1, ..., 1]]
    which has a tensor product structure, which is intuitive because we are operating on each run of x points independently.
    M = identity(nx/nx_bin) \prod ones(1, nx_bin)
    the binned image is obtained from matrix multiplication
    Ib = M * I

    In 2D, this situation is very similar. Here we take the image to be a row stacked vector
    I = (I[0, 0], I[0, 1], ..., I[0, nx-1], I[1, 0], ..., I[ny-1, nx-1])
    the binning operator is a (nx/nx_bin)*(ny/ny_bin) x nx*ny matrix which has a tensor product structure.

    This time the binning matrix has dimension (nx/nx_bin * ny/ny_bin) x (nx * ny)
    The top row starts with nx_bin 1's, then zero until position nx, and then ones until position nx + nx_bin.
    This pattern continues, with nx_bin 1's starting at jj*nx for jj = 0,...,ny_bin-1. The second row follows a similar
    pattern, but shifted by nx_bin pixels
    M = [[1, ..., 1, 0, ..., 0, 1, ..., 1, 0,...]
         [0, ..., 0, 1, ..., 1, ...
    Again, this has tensor product structure. Notice that the first (nx/nx_bin) x nx entries are the same as the 1D case
    and the whole matrix is constructed from blocks of these.
    M = [identity(ny/ny_bin) \prod ones(1, ny_bin)] \prod  [identity(nx/nx_bin) \prod ones(1, nx_bin)]

    Again, Ib = M*I

    Probably this pattern generalizes to higher dimensions!

    :param img: image to be binned
    :param nbin: [ny_bin, nx_bin] where these must evenly divide the size of the image
    :param mode: either 'sum' or 'mean'
    :return:
    """
    # todo: could also add ability to bin in this direction. Maybe could simplify function by allowing binning in
    # arbitrary dimension (one mode), with another mode to bin only certain dimensions and leave others untouched.
    # actually probably don't need to distinguish modes, this can be done by looking at bin_size.
    # still may need different implementation for the cases, as no reason to flatten entire array to vector if not
    # binning. But maybe this is not really a performance hit anyways with the sparse matrices?

    # if three dimensional, bin each image
    if img.ndim == 3:
        ny_bin, nx_bin = bin_size
        nz, ny, nx = img.shape

        # size of image after binning
        nx_s = int(nx / nx_bin)
        ny_s = int(ny / ny_bin)

        m_binned = np.zeros((nz, ny_s, nx_s))
        for ii in range(nz):
            m_binned[ii, :] = bin(img[ii], bin_size, mode=mode)

    # bin 2D image
    elif img.ndim == 2:
        ny_bin, nx_bin = bin_size
        ny, nx = img.shape

        if ny % ny_bin != 0 or nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')

        # size of image after binning
        nx_s = int(nx/nx_bin)
        ny_s = int(ny/ny_bin)

        # matrix which performs binning operation on row stacked matrix
        # need to use sparse matrices to bin even moderately sized images
        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        bin_mat_y = sp.kron(sp.identity(ny_s), np.ones((1, ny_bin)))
        bin_mat_xy = sp.kron(bin_mat_y, bin_mat_x)

        # row stack img. img.ravel() = [img[0, 0], img[0, 1], ..., img[0, nx-1], img[1, 0], ...]
        m_binned = bin_mat_xy.dot(img.ravel()).reshape([ny_s, nx_s])

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / (nx_bin * ny_bin)
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    # 1D "image"
    elif img.ndim == 1:

        nx_bin = bin_size[0]
        nx = img.size

        if nx % nx_bin != 0:
            raise ValueError('bin size must evenly divide image size.')
        nx_s = int(nx / nx_bin)

        bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
        m_binned = bin_mat_x.dot(img)

        if mode == 'sum':
            pass
        elif mode == 'mean':
            m_binned = m_binned / nx_bin
        else:
            raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)

    else:
        raise ValueError("Only 1D, 2D, or 3D arrays allowed")

    return m_binned

def simulated_img(ground_truth, max_photons, cam_gains, cam_offsets, cam_readout_noise_sds, pix_size=None, otf=None, na=1.3,
                  wavelength=0.5, photon_shot_noise=True, bin_size=1, use_otf=False):
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

    img_size = ground_truth.shape

    if use_otf:
        # get OTF
        if otf is None:
            raise ValueError("OTF was None, but use_otf was True")

        # blur image with otf/psf
        # todo: maybe should add an "imaging forward model" function to fit_psf.py and call it here.
        gt_ft = fft.fftshift(fft.fft2(fft.ifftshift(ground_truth)))
        img_blurred = max_photons * fft.fftshift(fft.ifft2(fft.ifftshift(gt_ft * otf))).real
        img_blurred[img_blurred < 0] = 0
    else:
        img_blurred = max_photons * ground_truth

    # resample image by binning
    img_blurred = bin(img_blurred, (bin_size, bin_size), mode='sum')

    max_photons_real = img_blurred.max()

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = np.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # add camera noise and convert from photons to ADU
    readout_noise = np.random.standard_normal(img_shot_noise.shape) * cam_readout_noise_sds

    img = cam_gains * img_shot_noise + readout_noise + cam_offsets

    # signal to noise ratio
    sig = cam_gains * img_blurred
    # assuming photon number large enough ~gaussian
    noise = np.sqrt(cam_readout_noise_sds**2 + cam_gains**2 * img_blurred)
    snr = sig / noise

    return img, snr, max_photons_real

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

if __name__ == "__main__":
    # ###############################
    # setup parametesr
    # ###############################
    figsize = (16, 8)
    na = 1. # numerical aperture
    ni = 1.4 # index of refraction
    wavelength = 0.532 # um
    nx = 201
    dc = 0.115 # camera pixel size, um
    theta = 30 * np.pi / 180 # light sheet angle to coverslip
    normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel
    #
    dz = dc * np.sin(theta) # distance planes above coverslip
    dy = 8 * dc * np.cos(theta)  # stage scanning step
    gn = np.arange(0, 30, dy) # stage positions, todo: allow to be unequally spaced
    npos = len(gn)

    # ###############################
    # get coordinates
    # ###############################
    # picture coordinates in coverslip frame
    x, y, z = get_lab_coords(nx, dc, theta, gn)

    # picture coordinates
    xp = dc * np.arange(nx)
    yp = xp

    # ###############################
    # generate random spots
    # ###############################
    nc = 10
    centers = np.concatenate((np.random.uniform(0.25 * z.max(), 0.75 * z.max(), size=(nc, 1)),
                              np.random.uniform(0.25 * y.max(), 0.75 * y.max(), size=(nc, 1)),
                              np.random.uniform(x.min(), x.max(), size=(nc, 1))), axis=1)
    sigma_xy = 0.22 * wavelength / na
    sigma_z = np.sqrt(6) / np.pi * ni * wavelength / na ** 2

    # ###############################
    # generate synthetic OPM data
    # ###############################
    imgs_opm = np.zeros((npos, nx, nx))
    for kk in range(nc):
        params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], sigma_xy, sigma_z, 0]
        imgs_opm += gaussian3d_pixelated_psf(x, y, z, dc, normal, params, sf=3)

    # add shot-noise and gaussian readout noise
    nphotons = 100
    bg = 100
    gain = 2
    noise = 5
    imgs_opm, _, _ = simulated_img(imgs_opm, nphotons, gain, bg, noise, use_otf=False)
    vmin = bg - 2
    vmax = np.percentile(imgs_opm, 99.999)

    # ###############################
    # identify candidate points in opm data
    # ###############################
    centers_guess_inds = find_candidate_beads(imgs_opm, filter_xy_pix=1, filter_z_pix=0.5, max_thresh=150, mode="threshold")
    xc = x[0, 0, centers_guess_inds[:, 2]]
    yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
    zc = z[0, centers_guess_inds[:, 1], 0] # z-position is determined by the y'-index in OPM image
    centers_guess = np.concatenate((zc[:, None], yc[:, None], xc[:, None]), axis=1)
    # eliminate multiple points too close together
    min_z_dist = 3 * sigma_z
    min_xy_dist = 4 * sigma_xy
    counter = 0
    while 1:
        z_dists = np.abs(centers_guess[counter][0] - centers_guess[:, 0])
        z_dists[counter] = np.inf
        xy_dists = np.sqrt((centers_guess[counter][1] - centers_guess[:, 1])**2 + (centers_guess[counter][2] - centers_guess[:, 2])**2)
        xy_dists[counter] = np.inf

        combine = np.logical_and(z_dists < min_z_dist, xy_dists < min_xy_dist)
        centers_guess[counter] = np.mean(centers_guess[combine], axis=0)
        centers_guess = centers_guess[np.logical_not(combine)]

        centers_guess_inds[counter] = np.round(np.mean(centers_guess_inds[combine], axis=0))
        centers_guess_inds = centers_guess_inds[np.logical_not(combine)]

        counter += 1
        if counter >= len(centers_guess):
            break

    # ###############################
    # do localization
    # ###############################
    centers_fit = np.zeros(centers_guess.shape)
    fit_results = []
    # roi sizes
    xy_size = 3 * sigma_xy
    z_size = 3 * sigma_z
    nxp = int(np.ceil(xy_size / dc))
    nyp = int(np.ceil(z_size / dc / np.cos(theta)))
    nzp = int(np.ceil(xy_size / dc / np.sin(theta)))
    # get rois
    rois = np.array([get_centered_roi(c, [nzp, nyp, nxp]) for c in centers_guess_inds])
    # ensure rois stay within bounds
    for ll in range(3):
        rois[:, 2*ll][rois[:, 2*ll] < 0] = 0
        rois[:, 2*ll + 1][rois[:, 2*ll + 1] >= imgs_opm.shape[ll]] = imgs_opm.shape[ll] - 1

    nroi = len(rois)
    # fit rois
    for ii, roi in enumerate(rois):
        img_roi = imgs_opm[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        x_roi = x[:, :, roi[4]:roi[5]] # only roi on last one because x has only one entry on first two dims
        y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
        z_roi = z[:, roi[2]:roi[3]:, ]

        # gaussian fitting localization
        def model_fn(p): return gaussian3d_pixelated_psf(x_roi, y_roi, z_roi, dc, normal, p, sf=3)
        init_params = [np.max(img_roi), centers_guess[ii, 2], centers_guess[ii, 1], centers_guess[ii, 0], 0.2, 1, np.mean(img_roi)]
        bounds = [[0, x_roi.min(), y_roi.min(), z_roi.min(), 0, 0, 0],
                  [np.inf, x_roi.max(), y_roi.max(), z_roi.max(), np.inf, np.inf, np.inf]]
        results = fit_model(img_roi, model_fn, init_params, bounds=bounds)

        # store results
        fit_results.append(results)
        centers_fit[ii, 0] = results["fit_params"][3]
        centers_fit[ii, 1] = results["fit_params"][2]
        centers_fit[ii, 2] = results["fit_params"][1]

        # plot localization fit diagnostic
        if ii == 0:
            fit_volume = model_fn(results["fit_params"])

            figh = plt.figure()
            grid = plt.GridSpec(2, nzp)
            for ii in range(roi[1] - roi[0]):
                ax = plt.subplot(grid[0, ii])
                plt.imshow(img_roi[ii], vmin=vmin, vmax=vmax)

                ax = plt.subplot(grid[1, ii])
                plt.imshow(fit_volume[ii], vmin=vmin, vmax=vmax)

    # todo: radiallity localization

    # ###############################
    # interpolate images so are on grids in coverslip coordinate system
    # ###############################
    xi, yi, zi, imgs_unskew = interp_opm_data(imgs_opm, dc, dy, theta, mode="row-interp")
    _, _, _, imgs_unskew2 = interp_opm_data(imgs_opm, dc, dy, theta, mode="ortho-interp")
    dxi = xi[1] - xi[0]
    dyi = yi[1] - yi[0]
    dzi = zi[1] - zi[0]

    # ###############################
    # get ground truth image in coverslip coordinates
    # ###############################
    imgs_square = np.zeros((len(zi), len(yi), len(xi)))
    for kk in range(nc):
        params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], sigma_xy, sigma_z, 0]
        imgs_square += gaussian3d_pixelated_psf(xi[None, None, :], yi[None, :, None], zi[:, None, None], dxi, np.array([0, 0, 1]), params, sf=3)
    # add noise
    imgs_square, _, _ = simulated_img(imgs_square, nphotons, gain, bg, noise, use_otf=False)
    # nan-mask region outside what we get from the OPM
    imgs_square[np.isnan(imgs_unskew)] = np.nan

    # ###############################
    # plot results
    # ###############################

    # ###############################
    # plot raw OPM data
    # ###############################
    plt.figure(figsize=figsize)
    plt.suptitle("Raw OPM data")
    ncols = int(np.ceil(np.sqrt(npos)) + 1)
    nrows = int(np.ceil(npos / ncols))
    for ii in range(npos):
        extent = [xp[0] - 0.5 * dc, xp[-1] + 0.5 * dc,
                  yp[-1] + 0.5 * dc, yp[0] - 0.5 * dc]

        ax = plt.subplot(nrows, ncols, ii + 1)
        ax.set_title("dy'=%0.2fum" % gn[ii])
        ax.imshow(imgs_opm[ii], vmin=vmin, vmax=vmax, extent=extent)

        # plot guess localizations
        to_plot = centers_guess_inds[:, 0] == ii
        if np.any(to_plot):
            plt.plot(dc * centers_guess_inds[to_plot][:, 2], dc * centers_guess_inds[to_plot][:, 1], 'gx')

        if ii == 0:
            plt.xlabel("x'")
            plt.ylabel("y'")

    # ###############################
    # maximum intensity projection comparisons
    # plot both interpolated data, ground truth, and localization results
    # ###############################
    plt.figure(figsize=figsize)
    grid = plt.GridSpec(3, 3)
    plt.suptitle("Maximum intensity projection comparison\n"
                 "wavelength=%0.0fnm, NA=%0.3f, n=%0.2f\n"
                 "dc=%0.3fum, stage step=%0.3fum, dx interp=%0.3fum, dy interp=%0.3fum, dz interp =%0.3fum, theta=%0.2fdeg"
                 % (wavelength * 1e3, na, ni, dc, dy, dxi, dyi, dzi, theta * 180 / np.pi))


    ax = plt.subplot(grid[0, 0])
    plt.imshow(np.nanmax(imgs_unskew, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 2], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 2], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("row interp\nX (um)")
    plt.title("XY")

    ax = plt.subplot(grid[0, 1])
    plt.imshow(np.nanmax(imgs_unskew, axis=1), vmin=vmin, vmax=vmax, origin="lower",
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 2], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 2], centers_fit[:, 0], 'mx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[0, 2])
    plt.imshow(np.nanmax(imgs_unskew, axis=2), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 0], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    # orthogonal interp
    ax = plt.subplot(grid[1, 0])
    plt.imshow(np.nanmax(imgs_unskew2, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 2], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 2], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("othogonal interp\nX (um)")

    ax = plt.subplot(grid[1, 1])
    plt.imshow(np.nanmax(imgs_unskew2, axis=1), vmin=vmin, vmax=vmax, origin="lower",
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 2], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 2], centers_fit[:, 0], 'mx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")

    ax = plt.subplot(grid[1, 2])
    plt.imshow(np.nanmax(imgs_unskew2, axis=2), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 0], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")

    # ground truth in these coords
    ax = plt.subplot(grid[2, 0])
    plt.imshow(np.nanmax(imgs_square, axis=0).transpose(), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi])
    plt.plot(centers[:, 1], centers[:, 2], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 2], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 2], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("ground truth\nX (um)")

    ax = plt.subplot(grid[2, 1])
    plt.imshow(np.nanmax(imgs_square, axis=1), vmin=vmin, vmax=vmax, origin="lower",
               extent=[xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 2], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 2], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 2], centers_fit[:, 0], 'mx')
    plt.xlabel("X (um)")
    plt.ylabel("Z (um)")

    ax = plt.subplot(grid[2, 2])
    plt.imshow(np.nanmax(imgs_square, axis=2), vmin=vmin, vmax=vmax, origin="lower",
               extent=[yi[0] - 0.5 * dyi, yi[-1] + 0.5 * dyi, zi[0] - 0.5 * dzi, zi[-1] + 0.5 * dzi])
    plt.plot(centers[:, 1], centers[:, 0], 'rx')
    plt.plot(centers_guess[:, 1], centers_guess[:, 0], 'gx')
    plt.plot(centers_fit[:, 1], centers_fit[:, 0], 'mx')
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")

if 0:
        # ###############################
        # plot interpolated data, using row interpolation
        # ###############################
        plt.figure(figsize=figsize)
        plt.suptitle("interpolated data, row interp")
        ncols = 6
        nrows = 6
        for ii in range(36):
            extent = [xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi,
                      yi[-1] + 0.5 * dyi, yi[0] - 0.5 * dyi]

            ax = plt.subplot(nrows, ncols, ii + 1)
            ax.set_title("z = %0.2fum" % zi[2 * ii])
            ax.imshow(imgs_unskew[2 * ii], vmin=vmin, vmax=vmax, extent=extent)

            if ii == 0:
                plt.xlabel("x")
                plt.ylabel("y")

        # ###############################
        # plot interpolated data, using orthogonal interpolation
        # ###############################
        plt.figure(figsize=figsize)
        plt.suptitle("interpolated data, ortho-interp")
        ncols = 6
        nrows = 6
        for ii in range(36):
            extent = [xi[0] - 0.5 * dxi, xi[-1] + 0.5 * dxi,
                      yi[-1] + 0.5 * dyi, yi[0] - 0.5 * dyi]

            ax = plt.subplot(nrows, ncols, ii + 1)
            ax.set_title("z = %0.2fum" % zi[2 * ii])
            ax.imshow(imgs_unskew2[2 * ii], vmin=vmin, vmax=vmax, extent=extent)

            if ii == 0:
                plt.xlabel("x")
                plt.ylabel("y")

        # ###############################
        # plot coordinates to compare original picture coordinates with interpolation grid
        # ###############################
        plt.figure(figsize=figsize)
        grid = plt.GridSpec(1, 2)
        plt.suptitle("Coordinates, dy=%0.2fum, dx=%0.2fum, dyi=%0.2f, dzi=%0.2f, theta=%0.2fdeg" % (
        dy, dc, dyi, dzi, theta * 180 / np.pi))

        ax = plt.subplot(grid[0, 0])
        ax.set_title("YZ plane")
        yiyi, zizi = np.meshgrid(yi, zi)
        plt.plot(yiyi, zizi, 'bx')
        plt.plot(y.ravel(), np.tile(z, [y.shape[0], 1, 1]).ravel(), 'k.')
        plt.plot(centers[:, 1], centers[:, 0], 'rx')
        plt.xlabel("y")
        plt.ylabel("z")
        plt.axis('equal')

        ax = plt.subplot(grid[0, 1])
        ax.set_title("XY plane")
        yiyi, xixi = np.meshgrid(yi, xi)
        plt.plot(yiyi, xixi, 'bx')
        plt.plot(np.tile(y, [1, 1, x.shape[2]]).ravel(), np.tile(x, [y.shape[0], y.shape[1], 1]).ravel(), 'k.')
        # for ii in range(nx):
        #     for jj in range(nz):
        #         plt.plot(y[jj], x[ii] * np.ones(y[jj].shape), 'k.')
        plt.plot(centers[:, 1], centers[:, 2], 'rx')
        plt.xlabel("y")
        plt.ylabel("x")
        plt.axis("equal")