"""
Localize skewed data with GUI for setting different parameters
"""
import os
import time
import numpy as np
import localize
import localize_skewed
import image_post_processing as ipp
import pycromanager
import napari
from napari.qt.threading import thread_worker
from magicgui import magicgui

root_dir = r"C:\Users\ptbrown2\Desktop"
# root_dir = r"\\10.206.26.21\opm2\20210628\new area"
dir_format = "bDNA_stiff_gel_human_lung_r%04d_y%04d_z%04d_ch%04d_1"

chunk_size_planes = 200
chunk_size_x = 325
chunk_overlap = 5

round = 1
tl = 14
iz = 0
ch = 1
data_fdir = os.path.join(root_dir, dir_format % (round, tl, iz, ch + 1))
dset = pycromanager.Dataset(data_fdir)

md = dset.read_metadata(z=0, channel=0)
frame_time_ms = float(md["OrcaFusionBT-Exposure"])

# load all images
imgs = []
for kk in range(len(dset.axes["z"])):
    imgs.append(dset.read_image(z=kk, channel=0))
imgs = np.flip(np.asarray(imgs), axis=0)

nplanes, nyp, nxp = imgs.shape

# object storing any data we want to access inside/outside of GUI
class LocObj():
    def __init__(self, imgs):
        self.raw_imgs = imgs
        self.dc = None
        self.dstep = None
        self.theta = None

        self.deskewed_data = None

        # filtering params
        self.filter_sigma_small = None
        self.filter_sigma_large = None
        # filtering data
        self.filtered_chunks = None
        self.chunk_coords = None
        self.chunk_rois = None
        self.filtered = None
        self.filtered_deskewed = None

        # localization params
        self.min_spot_sep = None
        self.threshold = None
        self.roi_size = None
        # localization data
        self.init_params = None
        self.fit_params = None
        self.conditions = None

        self.centers_pix = None

        # fit filtering params
        self.fit_dist_max_err = None
        self.sigmas_max = None
        self.sigmas_min = None
        self.fit_threshold = None
        self.dist_boundary_min = None
        # filtered results
        self.to_keep = None
        self.conditions = None
        self.condition_names = None

    def update_deskew(self):
        tstart = time.perf_counter()
        deskewed_data = ipp.deskew(self.raw_imgs, self.theta, self.dstep, self.dc)
        print("Deskewed images in %0.2fs" % (time.perf_counter() - tstart))
        self.deskewed_data = deskewed_data

    def update_filtered(self):
        tstart = time.perf_counter()

        nchunks = int(np.ceil(nplanes / (chunk_size_planes - chunk_overlap)) *
                      np.ceil(nxp / (chunk_size_x - chunk_overlap)))

        filtered_chunks = []
        chunk_coords = []
        chunk_rois = []

        more_chunks = True
        ichunk = 0
        chunk_counter_p = 0
        chunk_counter_x = 0
        while more_chunks:
            print("Chunk %d/%d, x index = %d, step index = %d" % (ichunk + 1, nchunks, chunk_counter_x, chunk_counter_p))
            ix_start = int(np.max([chunk_counter_x * chunk_size_x - chunk_overlap, 0]))
            ix_end = int(np.min([ix_start + chunk_size_x, nxp]))

            ip_start = int(np.max([chunk_counter_p * chunk_size_planes - chunk_overlap, 0]))
            ip_end = int(np.min([ip_start + chunk_size_planes, nplanes]))
            imgs_chunk = imgs[ip_start:ip_end, :, ix_start:ix_end]

            ks = localize_skewed.get_filter_kernel_skewed(self.filter_sigma_small, self.dc,
                                                          self.theta * np.pi / 180,
                                                          self.dstep, sigma_cutoff=2)
            kl = localize_skewed.get_filter_kernel_skewed(self.filter_sigma_large, self.dc,
                                                          self.theta * np.pi / 180,
                                                          self.dstep, sigma_cutoff=2)
            imgs_hp = localize.filter_convolve(imgs_chunk, ks)
            imgs_lp = localize.filter_convolve(imgs_chunk, kl, use_gpu=True)
            filtered = imgs_hp - imgs_lp

            # get image coordinates
            npos, ny, nx = imgs_chunk.shape
            y_offset = ip_start * self.dstep
            x_offset = ix_start * self.dc

            x, y, z = localize_skewed.get_skewed_coords((npos, ny, nx), self.dc, self.dstep, self.theta * np.pi/180)
            x += x_offset
            y += y_offset

            # store information for this chunk
            filtered_chunks.append(filtered)
            chunk_coords.append((z, y, x))
            chunk_rois.append([ip_start, ip_end, 0, filtered.shape[1], ix_start, ix_end])

            # update chunk counters
            if ix_end < nxp:
                chunk_counter_x += 1
                ichunk += 1
            elif ip_end < nplanes:
                chunk_counter_x = 0
                chunk_counter_p += 1
                ichunk += 1
            else:
                more_chunks = False

        print("Filtered images in %0.2fs" % (time.perf_counter() - tstart))

        self.filtered_chunks = filtered_chunks
        self.chunk_coords = chunk_coords
        self.chunk_rois = chunk_rois

        filtered_all = np.zeros(self.raw_imgs.shape)
        for ii in range(len(chunk_rois)):
            roi = chunk_rois[ii]
            filtered_all[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]] = filtered_chunks[ii]
        self.filtered = filtered_all

        self.filtered_deskewed = ipp.deskew(self.filtered, self.theta, self.dstep, self.dc)

    def update_localizations(self):
        # ###################################################
        # identify candidate beads
        # ###################################################
        tstart = time.perf_counter()

        dz_min, dxy_min = self.min_spot_sep

        footprint = localize_skewed.get_skewed_footprint((dz_min, dxy_min, dxy_min), self.dc, self.dstep, self.theta * np.pi/180)

        fit_params_vol = []
        init_params_vol = []
        rois_vol = []
        for ii, (imgs_filtered, coords, grand_roi) in enumerate(zip(self.filtered_chunks, self.chunk_coords, self.chunk_rois)):
            print("localizing chunk %d/%d" % (ii + 1, len(self.filtered_chunks)))
            z, y, x = coords
            imgs_chunk = self.raw_imgs[grand_roi[0]:grand_roi[1], grand_roi[2]:grand_roi[3], grand_roi[4]:grand_roi[5]]

            centers_guess_inds, amps = localize.find_peak_candidates(imgs_filtered, footprint, self.threshold)

            # convert to xyz coordinates
            xc = x[0, 0, centers_guess_inds[:, 2]]
            yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
            zc = z[0, centers_guess_inds[:, 1], 0]  # z-position is determined by the y'-index in OPM image
            centers_guess = np.stack((zc, yc, xc), axis=1)

            print("Found %d points above threshold in %0.2fs" % (
                len(centers_guess), time.perf_counter() - tstart))

            if len(centers_guess) != 0:
                # ###################################################
                # average multiple points too close together. Necessary bc if naive threshold, may identify several points
                # from same spot. Particularly important if spots have very different brightness levels.
                # ###################################################
                tstart = time.perf_counter()

                inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
                weights = imgs_filtered.ravel()[inds]
                centers_guess, inds_comb = localize.filter_nearby_peaks(centers_guess, dxy_min, dz_min, weights=weights,
                                                                        mode="average")

                amps = amps[inds_comb]
                print("Found %d points separated by dxy > %0.5g and dz > %0.5g in %0.1fs" %
                      (len(centers_guess), dxy_min, dz_min, time.perf_counter() - tstart))

                # ###################################################
                # prepare ROIs
                # ###################################################
                tstart = time.perf_counter()

                # cut rois out
                roi_size_skew = localize_skewed.get_skewed_roi_size(self.roi_size, self.theta*np.pi/180, self.dc, self.dstep, ensure_odd=True)
                rois, img_rois, xrois, yrois, zrois = zip(
                    *[localize_skewed.get_skewed_roi(c, imgs_chunk, x, y, z, roi_size_skew) for c in centers_guess])
                rois = np.asarray(rois)

                # exclude some regions of roi
                roi_masks = [localize_skewed.get_roi_mask(c, (np.inf, 0.5 * self.roi_size[1]), (zrois[ii], yrois[ii], xrois[ii]))
                             for ii, c in enumerate(centers_guess)]

                # mask regions
                xrois, yrois, zrois, img_rois = zip(
                    *[(xr[rm][None, :], yr[rm][None, :], zr[rm][None, :], ir[rm][None, :])
                      for xr, yr, zr, ir, rm in zip(xrois, yrois, zrois, img_rois, roi_masks)])

                # extract guess values
                bgs = np.array([np.mean(r) for r in img_rois])
                sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in
                                zip(img_rois, xrois, centers_guess)])
                sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in
                                zip(img_rois, yrois, centers_guess)])
                sxys = np.expand_dims(0.5 * (sxs + sys), axis=1)
                szs = np.expand_dims(np.array(
                    [np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in
                     zip(img_rois, zrois, centers_guess)]), axis=1)

                # get initial parameter guesses
                init_params = np.concatenate((np.expand_dims(amps, axis=1),
                                              centers_guess[:, 2][:, None],
                                              centers_guess[:, 1][:, None],
                                              centers_guess[:, 0][:, None],
                                              sxys, szs,
                                              np.expand_dims(bgs, axis=1)),
                                             axis=1)

                print("Prepared %d rois and estimated initial parameters in %0.2fs" % (
                len(rois), time.perf_counter() - tstart))

                # ###################################################
                # localization
                # ###################################################
                print("starting fitting for %d rois" % centers_guess.shape[0])
                tstart = time.perf_counter()

                fit_params, fit_states, chi_sqrs, niters, fit_t = localize.fit_gauss_rois(img_rois, (zrois, yrois, xrois),
                                                                                          init_params, estimator="LSE",
                                                                                          model="gaussian",
                                                                                          sf=1, dc=self.dc,
                                                                                          angles=(0., self.theta * np.pi/180, 0.))

                tend = time.perf_counter()
                print("Localization took %0.2fs" % (tend - tstart))

                # fitting
                print("Fitting %d rois on GPU" % (len(rois)))
                fit_results = np.concatenate((np.expand_dims(fit_states, axis=1),
                                              np.expand_dims(chi_sqrs, axis=1),
                                              np.expand_dims(niters, axis=1)), axis=1)



                # ###################################################
                # correct ROIs for full volume
                # ###################################################
                rois[:, :2] += grand_roi[0]
                rois[:, 4:] += grand_roi[4]

                # ###################################################
                # store results
                # ###################################################
                fit_params_vol.append(fit_params)
                init_params_vol.append(init_params)
                rois_vol.append(rois)

        self.fit_params = np.concatenate(fit_params_vol, axis=0)
        self.init_params = np.concatenate(init_params_vol, axis=0)
        self.rois = np.concatenate(rois_vol, axis=0)
        self.centers_pix = np.stack((self.fit_params[:, 3], self.fit_params[:, 2], self.fit_params[:, 1]), axis=1) / self.dc

    def update_fit_filters(self):
        # ###################################################
        # preliminary fitting of results
        # ###################################################
        tstart = time.perf_counter()

        x, y, z = localize_skewed.get_skewed_coords(self.raw_imgs.shape, self.dc, self.dstep, self.theta * np.pi/180)

        to_keep, conditions, condition_names, filter_settings = localize_skewed.filter_localizations(
            self.fit_params, self.init_params, (z, y, x),
            self.fit_dist_max_err, self.min_spot_sep,
            (self.sigmas_min,self.sigmas_max),
            self.fit_threshold,
            dist_boundary_min=self.dist_boundary_min)

        print("identified %d/%d localizations in %0.3f" % (
            np.sum(to_keep), to_keep.size, time.perf_counter() - tstart))

        self.to_keep = to_keep
        self.conditions = conditions
        self.condition_names = condition_names

obj = LocObj(imgs)

viewer = napari.Viewer(title=root_dir, ndisplay=3)
# viewer.dims.ndisplay = 3
viewer.camera.angles = (20, -77, -22)

# draw layers on napari
def update_img_layer(layers, names):
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    if not isinstance(names, (list, tuple)):
        names = [names]

    for l, n in zip(layers, names):
        if l is None:
            continue

        try:
            viewer.layers[n].data = l
        except KeyError:
            viewer.add_image(l, name=n)

def update_point_layers(layers, names):
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    if not isinstance(names, (list, tuple)):
        names = [names]

    for l, n in zip(layers, names):
        if l is None:
            continue

        try:
            viewer.layers[n].data = l
        except KeyError:
            viewer.add_points(l, size=2, face_color="red", name=n,
                              opacity=0.75, n_dimensional=True, visible=True)

def update_layers_helper(layers):
    dimg, fimg, fcs = layers
    update_img_layer((dimg, fimg), ("deskewed img", "filtered img"))
    update_point_layers(fcs, "fit centers")

@thread_worker
def loc_thread(dc=None, theta=None, dstep=None, filter_sigma_small=None, filter_sigma_large=None,
               min_spot_sep=None, threshold=None, roi_size=None,
               sigmas_max=None, sigmas_min=None, fit_threshold=None, dist_boundary_min=None,
               fit_dist_max_err=None):

    # based on new parameters decide how much of the processing pipeline must be redone
    update_deskew = dc != obj.dc or theta != obj.theta or dstep != obj.dstep
    update_filter = update_deskew or obj.filter_sigma_small != filter_sigma_small or obj.filter_sigma_large != filter_sigma_large
    update_locs = update_filter or obj.min_spot_sep != min_spot_sep or obj.threshold != threshold or obj.roi_size != roi_size
    update_fit_filters = update_locs or obj.sigmas_max != sigmas_max or obj.sigmas_min != sigmas_min or \
                         obj.fit_threshold != fit_threshold or obj.dist_boundary_min != dist_boundary_min or \
                         obj.fit_dist_max_err != fit_dist_max_err

    if update_deskew:
        obj.dc = dc
        obj.theta = theta
        obj.dstep = dstep
        obj.update_deskew()

    if update_filter:
        obj.filter_sigma_small = filter_sigma_small
        obj.filter_sigma_large = filter_sigma_large
        obj.update_filtered()

    if update_locs:
        obj.min_spot_sep = min_spot_sep
        obj.threshold = threshold
        obj.roi_size = roi_size
        obj.update_localizations()

    if update_fit_filters:
        obj.fit_dist_max_err = fit_dist_max_err
        obj.sigmas_max = sigmas_max
        obj.sigmas_min = sigmas_min
        obj.fit_threshold = fit_threshold
        obj.dist_boundary_min = dist_boundary_min
        obj.update_fit_filters()

    yield (obj.deskewed_data, obj.filtered_deskewed, obj.centers_pix[obj.to_keep])

@magicgui(call_button="update")
def loc(dc=0.115, theta=30., dstep=0.4,
        filter_sz_small=0.18, filter_sxy_small=0.025, filter_sz_large=1.8, filter_sxy_large=0.5,
        min_sep_z=1.8, min_sep_xy=0.4, threshold=100, roi_size_z=1.8, roi_size_xy=1.2,
        sz_max=1., sz_min=0.088, sxy_max=0.38, sxy_min=0.024, fit_threshold=100, dist_boundary_z=0.2, dist_boundary_xy=0.1,
        z_err_fit_max=0.35, xy_fit_err_max=0.3):


    worker_filter = loc_thread(dc=dc, theta=theta, dstep=dstep,
                               filter_sigma_small=(filter_sz_small, filter_sxy_small, filter_sxy_small),
                               filter_sigma_large=(filter_sz_large, filter_sxy_large, filter_sxy_large),
                               min_spot_sep=(min_sep_z, min_sep_xy), threshold=threshold,
                               roi_size=(roi_size_z, roi_size_xy, roi_size_xy),
                               sigmas_max=(sz_max, sxy_max), sigmas_min=(sz_min, sxy_min), fit_threshold=fit_threshold,
                               dist_boundary_min=(dist_boundary_z, dist_boundary_xy), fit_dist_max_err=(z_err_fit_max, xy_fit_err_max))
    worker_filter.yielded.connect(update_layers_helper)
    worker_filter.start()

viewer.window.add_dock_widget(loc)