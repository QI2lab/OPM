#!/usr/bin/env python

from pathlib import Path
import napari
from dask import array as da

path_to_data = Path('E:\\timelapse_zarr.zarr\\opm_data')
image = da.from_zarr(str(path_to_data))

viewer = napari.view_image(image[:,0,:,300:1000,:],name='CFP')
viewer.add_image(image[:,1,:,300:1000,:],name='GFP')
viewer.add_image(image[:,2,:,300:1000,:],name='mCherry')

napari.run()