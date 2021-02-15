#!/usr/bin/env python

'''
Stream galvo scan volumes using Dask and Napari

Shepherd 02/21
'''

import napari
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da
from glob import glob
'''
filenames = sorted(glob("Y:/20210203/beads_50glyc_1000dilution_1/decon/*.tif"), key=alphanumeric_key)
# read the first file to get the shape and dtype
# ASSUMES THAT ALL FILES SHARE THE SAME SHAPE/TYPE
sample = imread(filenames[0])

lazy_imread = delayed(imread)  # lazy reader
lazy_arrays = [lazy_imread(fn) for fn in filenames]
dask_arrays = [
    da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
    for delayed_reader in lazy_arrays
]
# Stack into one large dask.array
stack = da.stack(dask_arrays, axis=0)

da.to_zarr(stack,'e:/zarr_decon_50glyc')
'''

stack = da.from_zarr('e:/zarr_decon_50glyc')
with napari.gui_qt():
    # specify contrast_limits and is_pyramid=False with big data
    # to avoid unnecessary computations
    v = napari.view_image(stack, contrast_limits=[100,1000], multiscale=False)
