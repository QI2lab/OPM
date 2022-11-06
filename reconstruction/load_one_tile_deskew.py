from pycromanager import Dataset
import napari
from image_post_processing import deskew
from data_io import return_data_numpy
from pathlib import Path
import numpy as np

file_path = Path(r'D:\20220415_TGEN_probes_Hela_cells\PC_Hella_cells_r0000_y0001_z0000_1')

dataset = Dataset(str(file_path))
da_array = dataset.as_array(['c','z'])
da_array.shape

excess_images = 50
num_images = da_array.shape[1]
y_pixels=da_array.shape[3]
x_pixels=da_array.shape[2]
time_axis = 0

channel_axis = 2
raw_channel_one= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)
channel_axis = 3
raw_channel_two= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)

channel_one = deskew(np.flipud(raw_channel_one),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)
channel_two = deskew(np.flipud(raw_channel_two),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)

viewer = napari.Viewer()
viewer.add_image(channel_one,scale=(.115,.115,.115),name='r0-Atto565',blending='additive')
viewer.add_image(channel_two,scale=(.115,.115,.115),name='r0-Alexa647',blending='additive')

file_path = Path(r'D:\20220415_TGEN_probes_Hela_cells\PC_Hella_cells_r0001_y0001_z0000_1')

dataset = Dataset(str(file_path))
da_array = dataset.as_array(['c','z'])
da_array.shape

excess_images = 50
num_images = da_array.shape[1]
y_pixels=da_array.shape[3]
x_pixels=da_array.shape[2]
time_axis = 0

channel_axis = 2
raw_channel_one= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)
channel_axis = 3
raw_channel_two= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)

channel_one = deskew(np.flipud(raw_channel_one),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)
channel_two = deskew(np.flipud(raw_channel_two),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)

#viewer = napari.Viewer()
viewer.add_image(channel_one,scale=(.115,.115,.115),name='r1-Atto565',blending='additive')
viewer.add_image(channel_two,scale=(.115,.115,.115),name='r1-Alexa647',blending='additive')


file_path = Path(r'D:\20220415_TGEN_probes_Hela_cells\PC_Hella_cells_r0002_y0001_z0000_1')

dataset = Dataset(str(file_path))
da_array = dataset.as_array(['c','z'])
da_array.shape

excess_images = 50
num_images = da_array.shape[1]
y_pixels=da_array.shape[3]
x_pixels=da_array.shape[2]
time_axis = 0

channel_axis = 2
raw_channel_one= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)
channel_axis = 3
raw_channel_two= return_data_numpy(dataset, time_axis=None, channel_axis=channel_axis, num_images=num_images, excess_images=excess_images, y_pixels=y_pixels, x_pixels=x_pixels)

channel_one = deskew(np.flipud(raw_channel_one),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)
channel_two = deskew(np.flipud(raw_channel_two),theta=30,distance=0.4,pixel_size=.115).astype(np.uint16)

#viewer = napari.Viewer()
viewer.add_image(channel_one,scale=(.115,.115,.115),name='r2-Atto565',blending='additive')
viewer.add_image(channel_two,scale=(.115,.115,.115),name='r2-Alexa647',blending='additive')