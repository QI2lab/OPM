import numpy as np
import scipy
from scipy import fft
import scipy.sparse as sp
import skimage.feature
import skimage.filters
import matplotlib.pyplot as plt
import warnings
import localize

if __name__ == "__main__":
    # ###############################
    # setup parametesr
    # ###############################
    figsize = (16, 8)
    na = 1. # numerical aperture
    ni = 1.4 # index of refraction
    wavelength = 0.532 # um
    nx = 300
    ny = 256
    dc = 0.115 # camera pixel size, um
    theta = 30 * np.pi / 180 # light sheet angle to coverslip
    normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel
    #
    dz = dc * np.sin(theta) # distance planes above coverslip
    #dy = 2 * dc * np.cos(theta)  # stage scanning step
    dy = 0.4
    gn = np.arange(0, 30, dy) # stage positions, todo: allow to be unequally spaced
    npos = len(gn)

    # ###############################
    # get coordinates
    # ###############################
    # picture coordinates in coverslip frame
    x, y, z = localize.get_lab_coords(nx, ny, dc, theta, gn)

    # picture coordinates
    xp = dc * np.arange(nx)
    yp = dc * np.arange(ny)

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
    imgs_opm = np.zeros((npos, ny, nx))
    for kk in range(nc):
        params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], sigma_xy, sigma_z, 0]
        imgs_opm += localize.gaussian3d_pixelated_psf(x, y, z, dc, normal, params, sf=3)

    # add shot-noise and gaussian readout noise
    nphotons = 100
    bg = 100
    gain = 2
    noise = 5
    imgs_opm, _, _ = localize.simulated_img(imgs_opm, nphotons, gain, bg, noise, use_otf=False)
    vmin = bg - 2
    vmax = np.percentile(imgs_opm, 99.999)

    # ###############################
    # identify candidate points in opm data
    # ###############################
    centers_guess_inds = localize.find_candidate_beads(imgs_opm, filter_xy_pix=1, filter_z_pix=0.5, max_thresh=150, mode="threshold")
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

    # roi sizes
    xy_size = 3 * sigma_xy
    z_size = 3 * sigma_z
    nxp = int(np.ceil(xy_size / dc))
    nyp = int(np.ceil(z_size / dc / np.cos(theta)))
    nzp = int(np.ceil(xy_size / dc / np.sin(theta)))
    # get rois
    rois = np.array([localize.get_centered_roi(c, [nzp, nyp, nxp]) for c in centers_guess_inds])
    # ensure rois stay within bounds
    for ll in range(3):
        rois[:, 2*ll][rois[:, 2*ll] < 0] = 0
        rois[:, 2*ll + 1][rois[:, 2*ll + 1] >= imgs_opm.shape[ll]] = imgs_opm.shape[ll] - 1

    # remove any rois that are too small
    nmin = 1
    exclude = np.logical_or(rois[:, 1] - rois[:, 0] <= nmin, rois[:, 3] - rois[:, 2] <= nmin)
    exclude = np.logical_or(exclude, rois[:, 5]- rois[:, 4] <= nmin)
    include = np.logical_not(exclude)
    rois = rois[include]
    centers_guess = centers_guess[include]
    centers_guess_inds = centers_guess_inds[include]

    nroi = len(rois)
    centers_fit = np.zeros(centers_guess.shape)
    fit_results = []
    # fit rois
    for ii, roi in enumerate(rois):
        img_roi = imgs_opm[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        x_roi = x[:, :, roi[4]:roi[5]] # only roi on last one because x has only one entry on first two dims
        y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
        z_roi = z[:, roi[2]:roi[3]:, ]

        # gaussian fitting localization
        def model_fn(p): return localize.gaussian3d_pixelated_psf(x_roi, y_roi, z_roi, dc, normal, p, sf=3)
        init_params = [np.max(img_roi), centers_guess[ii, 2], centers_guess[ii, 1], centers_guess[ii, 0], 0.2, 1, np.mean(img_roi)]
        bounds = [[0, x_roi.min(), y_roi.min(), z_roi.min(), 0, 0, 0],
                  [np.inf, x_roi.max(), y_roi.max(), z_roi.max(), np.inf, np.inf, np.inf]]
        results = localize.fit_model(img_roi, model_fn, init_params, bounds=bounds)

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
    xi, yi, zi, imgs_unskew = localize.interp_opm_data(imgs_opm, dc, dy, theta, mode="row-interp")
    _, _, _, imgs_unskew2 = localize.interp_opm_data(imgs_opm, dc, dy, theta, mode="ortho-interp")
    dxi = xi[1] - xi[0]
    dyi = yi[1] - yi[0]
    dzi = zi[1] - zi[0]

    # ###############################
    # get ground truth image in coverslip coordinates
    # ###############################
    imgs_square = np.zeros((len(zi), len(yi), len(xi)))
    for kk in range(nc):
        params = [1, centers[kk, 2], centers[kk, 1], centers[kk, 0], sigma_xy, sigma_z, 0]
        imgs_square += localize.gaussian3d_pixelated_psf(xi[None, None, :], yi[None, :, None], zi[:, None, None], dxi, np.array([0, 0, 1]), params, sf=3)
    # add noise
    imgs_square, _, _ = localize.simulated_img(imgs_square, nphotons, gain, bg, noise, use_otf=False)
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