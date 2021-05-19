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
import localize
import data_io
import tifffile
import os

# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408n", "glycerol60x_1", "2021_04_20_18;15;17_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408n", "glycerol60x_1", "2021_04_21_10;22;43_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210408m", "glycerol50x_1", "2021_04_21_10;17;23_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430i", "glycerol_60_1", "2021_05_02_15;17;29_localization")
# loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430a", "beads_1", "dummy")
loc_data_dir = os.path.join(r"\\10.206.26.21\opm2\20210430f", "glycerol_40_1", "2021_05_06_17;28;06_localization")


data_dir, _ = os.path.split(loc_data_dir)
root_dir, _ = os.path.split(data_dir)
loc_data_str = "localization_results_vol_%d.pkl"
deskew_fname = os.path.join(root_dir, "full_deskew_only.h5")
md_fname = os.path.join(root_dir, "scan_metadata.csv")
track_fname = os.path.join(loc_data_dir, "tracks.pkl")
plot_centers = True
plot_tracks = True
plot_centers_guess = True

# reader for deskewed dataset
reader = npy2bdv.BdvEditor(deskew_fname)
ntimes, nilluminations, nchannels, ntiles, nangle = reader.get_attribute_count()

# load metadata
md = data_io.read_metadata(md_fname)
# x, y, z = localize.get_lab_coords(md["x_pixels"], md["y_pixels"], md["pixel_size"] / 1000, md["theta"] * np.pi/180,
#                                   np.arange(md["scan_axis_positions"]) * md["scan_step"] / 1000)
# raw_shape = np.broadcast_arrays(x, y, z)[0].shape

pix_sizes = [md["pixel_size"] / 1000] * 3
# pix_sizes = [md["pixel_size"] * np.sin(md["theta"] * np.pi / 180) / 1000, md["pixel_size"] * np.cos(md["theta"] * np.pi / 180) / 1000, md["pixel_size"] / 1000]

if plot_centers:
    # load all localizations and convert to pixel coordinates
    try:
        centers = []
        centers_guess = []
        for ii in range(ntimes):
            data_fname = os.path.join(loc_data_dir, loc_data_str % ii)

            # in case not all images have been analyzed
            if not os.path.exists(data_fname):
                break

            with open(data_fname, "rb") as f:
                dat = pickle.load(f)
            centers.append(np.concatenate((ii * np.ones((len(dat["centers"]), 1)),
                                           dat["centers"][:, 0][:, None] / pix_sizes[0],
                                           dat["centers"][:, 1][:, None] / pix_sizes[1],
                                           dat["centers"][:, 2][:, None] / pix_sizes[2]), axis=1))
            centers_guess.append(np.concatenate((ii * np.ones((len(dat["centers_guess"]), 1)),
                                           dat["centers_guess"][:, 0][:, None] / pix_sizes[0],
                                           dat["centers_guess"][:, 1][:, None] / pix_sizes[1],
                                           dat["centers_guess"][:, 2][:, None] / pix_sizes[2]), axis=1))
        centers = np.concatenate(centers, axis=0)
        centers_guess = np.concatenate(centers_guess, axis=0)
    except:
        plot_centers = False

# load tracks
if os.path.exists(track_fname):
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


# create lazy dask array of deskewed data
# with npy2bdv
# imread = dask.delayed(reader.read_view, pure=True)
# lazy_images = [imread(ii, 0, 0, 0, 0) for ii in range(ntimes)]

# with h5py
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

# plot with napari
with napari.gui_qt():
    # specify contrast_limits and is_pyramid=False with big data to avoid unnecessary computations
    viewer = napari.view_image(stack, colormap="bone", contrast_limits=[0, 750], multiscale=False, title=loc_data_dir)
    if plot_centers:
        viewer.add_points(centers, size=2, face_color="red", opacity=0.75, n_dimensional=True)
    if plot_centers_guess:
        viewer.add_points(centers_guess, size=2, face_color="green", opacity=0.5, n_dimensional=True)
    if plot_tracks:
        viewer.add_tracks(track_data, opacity=1, blending="opaque", tail_length=20, tail_width=2)