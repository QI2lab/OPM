"""
Plot localizations using napari and deskewed data
"""

import numpy as np
import napari
import npy2bdv
import h5py
import dask.array as da
import dask.delayed
import pickle
import localize_skewed
import data_io
import tifffile
import os
import pycromanager
import matplotlib.pyplot as plt

# #######################################
# data files
# #######################################
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408n", "glycerol60x_1", "2021_04_20_18;15;17_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408n", "glycerol60x_1", "2021_04_21_10;22;43_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408m", "glycerol50x_1", "2021_04_21_10;17;23_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430i", "glycerol_60_1", "2021_05_02_15;17;29_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430a", "beads_1", "dummy")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430f\glycerol_40_1", "2021_05_20_16;20;04_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210518k", "glycerol40_1", "2021_05_06_17;28;06_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210521s", "glycerol_70_1", "2021_05_31_10;39;05_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210622a", "glycerol90_1", "2021_06_22_23;18;29_localization")
loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210624a", "glycerol80_1", "2021_06_24_16;37;59_localization")

data_dir, _ = os.path.split(loc_data_dir)
root_dir, _ = os.path.split(data_dir)
loc_data_str = "localization_results_vol_%d.pkl"
deskew_fname = os.path.join(root_dir, "full_deskew_only.h5")
md_fname = os.path.join(root_dir, "scan_metadata.csv")
track_fname = os.path.join(loc_data_dir, "tracks.pkl")

plot_centers = True
plot_centers_guess = True
plot_fit_filters = False
plot_tracks = True

# #######################################
# load metadata
# #######################################
md = data_io.read_metadata(md_fname)
pix_sizes = [md["pixel_size"] / 1000] * 3

# #######################################
# lazy loading for reading deskewed image files
# #######################################
# reader for deskewed dataset
reader = npy2bdv.BdvEditor(deskew_fname)
ntimes, nilluminations, nchannels, ntiles, nangle = reader.get_attribute_count()

# create lazy dask array of deskewed data
# with npy2bdv
# imread = dask.delayed(reader.read_view, pure=True)
# lazy_images = [imread(ii, 0, 0, 0, 0) for ii in range(ntimes)]

# with h5py
if not os.path.exists(deskew_fname):
    raise ValueError("path '%s' does not exist" % deskew_fname)
hf = h5py.File(deskew_fname)
imread = dask.delayed(lambda ii: np.asarray(hf["t%05d/s00/0/cells" % ii]), pure=True)
lazy_images = [imread(ii) for ii in range(ntimes)]

# other deskew code
# fpath = r"\\10.206.26.21\opm2\20210408n\deskew_test"
# imread = dask.delayed(lambda ii: tifffile.imread(os.path.join(fpath, "vol=%d.tiff" % ii)), pure=True)
# lazy_images = [imread(ii) for ii in range(3)]

img0 = lazy_images[0].compute()
shape0 = img0.shape

arr = [da.from_delayed(lazy_image, dtype=np.uint16, shape=shape0) for lazy_image in lazy_images]
stack = da.stack(arr, axis=0)

# #######################################
# load localizations
# #######################################
if plot_centers:
    # load all localizations and convert to pixel coordinates
    try:
        centers = []
        centers_guess = []
        to_keep = []
        conditions = []
        rois = []
        fit_params = []
        init_params = []
        for ii in range(ntimes):
            print("loading file %d/%d" % (ii + 1, ntimes))
            data_fname = os.path.join(loc_data_dir, loc_data_str % ii)

            # in case not all images have been analyzed
            if not os.path.exists(data_fname):
                break

            with open(data_fname, "rb") as f:
                dat = pickle.load(f)

            tk = dat["to_keep"]
            cd = dat["conditions"]
            cd_names = dat["condition_names"]
            fp = dat["fit_params"]
            ip = dat["init_params"]
            ri = dat["rois"]
            centers.append(np.concatenate((ii * np.ones((int(np.sum(tk)), 1)),
                                           fp[:, 3][tk][:, None] / pix_sizes[0],
                                           fp[:, 2][tk][:, None] / pix_sizes[1],
                                           fp[:, 1][tk][:, None] / pix_sizes[2]
                                           ), axis=1))
            centers_guess.append(np.concatenate((ii * np.ones((len(ip), 1)),
                                           ip[:, 3][:, None] / pix_sizes[0],
                                           ip[:, 2][:, None] / pix_sizes[1],
                                           ip[:, 1][:, None] / pix_sizes[2]
                                           ), axis=1))
            to_keep.append(tk)
            conditions.append(cd)
            rois.append(ri)
            fit_params.append(fp)
            init_params.append(ip)

        centers = np.concatenate(centers, axis=0)
        centers_guess = np.concatenate(centers_guess, axis=0)
        to_keep = np.concatenate(to_keep, axis=0)
        conditions = np.concatenate(conditions, axis=0)
        rois = np.concatenate(rois, axis=0)
        fit_params = np.concatenate(fit_params, axis=0)
        init_params = np.concatenate(init_params, axis=0)
    except:
        plot_centers = False

# #######################################
# load tracks
# #######################################
if os.path.exists(track_fname) and plot_tracks:
    with open(track_fname, "rb") as f:
        track_data_pd = pickle.load(f)

    pns_unq, inv = np.unique(track_data_pd["particle"], return_inverse=True)
    track_data_pd["particles unique"] = inv
    frames = np.unique(track_data_pd["frame"])

    # track_data = np.zeros((len(pns_unq), len(frames), 3)) * np.nan
    # f = track_data_pd["frame"].values
    # p = track_data_pd["particles unique"].values
    # track_data[p, f, 2] = track_data_pd["xum"].values
    # track_data[p, f, 1] = track_data_pd["yum"].values
    # track_data[p, f, 0] = track_data_pd["zum"].values
    track_data = np.concatenate((np.expand_dims(track_data_pd["particles unique"].values, axis=1),
                                 np.expand_dims(track_data_pd["frame"].values, axis=1),
                                 np.expand_dims(track_data_pd["zum"].values / pix_sizes[0], axis=1),
                                 np.expand_dims(track_data_pd["yum"].values / pix_sizes[1], axis=1),
                                 np.expand_dims(track_data_pd["xum"].values / pix_sizes[2], axis=1)), axis=1)
    track_data = np.round(track_data).astype(np.int)

else:
    plot_tracks = False

# plot roi's with matplotlib
if False:
    # need to load real data for this
    dset_dir = os.path.join(loc_data_dir, "..")
    dset = pycromanager.Dataset(dset_dir)

    imgs = []
    vv = 0
    for kk in range(25):
        imgs.append(dset.read_image(z=kk, t=vv, c=0))
    imgs = np.asarray(imgs)
    imgs = np.flip(imgs, axis=0)

    # find fit nearest to point from napari (in deskewed pixels)
    # inds = [31, 105, 581]
    # dists = ((centers_guess[:, 1] - inds[0])**2 + (centers_guess[:, 2] - inds[1])**2 + (centers_guess[:, 3] - inds[2])**2)
    # dists[centers_guess[:, 0] != 0] = np.nan
    # # dists[np.logical_not(to_keep)] = np.nan
    # ind = np.nanargmin(dists)
    # print("kept = %d" % to_keep[ind])

    # plot
    theta = md["theta"] * np.pi/180
    x, y, z = localize.get_skewed_coords(imgs.shape, md["pixel_size"] / 1000, md["scan_step"] / 1000, theta)

    num_plotted = 0
    ind = 0
    while num_plotted < 73:
        if to_keep[ind]:
            figa, figb = localize.plot_skewed_roi(fit_params[ind], rois[ind], imgs, theta, x, y, z, init_params=init_params[ind],
                                     figsize=(16, 8), same_color_scale=False)
            num_plotted += 1
        ind += 1

# #######################################
# plot with napari
# #######################################
# specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
viewer = napari.view_image(stack, colormap="bone", contrast_limits=[np.percentile(img0, 1), np.percentile(img0, 99.999)],
                           multiscale=False, title=loc_data_dir)
# viewer.scale_bar_visible = True

if plot_centers:
    viewer.add_points(centers, size=2, face_color="red", name="fits", opacity=0.75, n_dimensional=True)

if plot_centers_guess:
    viewer.add_points(centers_guess, size=2, face_color="green", name="guesses", opacity=0.5, n_dimensional=True)

# filters are useful, but slow down rendering considerably even when not visible
if plot_fit_filters:
    # show which fits failed and why
    cds = conditions.transpose()
    colors = ["purple", "blue", "green", "yellow", "orange"] * int(np.ceil(len(cds) / 5))
    for c, cn, col in zip(cds, cd_names, colors):
        ct = centers_guess[np.logical_not(c)]
        viewer.add_points(ct, size=2, face_color=col, opacity=0.5, name="not %s" % cn.replace("_", " "),
                          n_dimensional=True, visible=False)

if plot_tracks:
    viewer.add_tracks(track_data, opacity=1, blending="opaque", tail_length=20, tail_width=2)
