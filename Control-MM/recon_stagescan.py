#!/usr/bin/env python

'''
Stage scanning OPM post-processing using numpy, numba, skimage, and npy2bdv.
New version to handle altered acquisition code for multi-color large area stage scans. 
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 01/21
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import gc
import sys
import getopt
import re
from skimage.measure import block_reduce
from skimage.util import apply_parallel
import time
from numba import njit, prange

# perform stage scanning reconstruction using orthogonal interpolation
# http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
@njit(parallel=True)
def stage_deskew(data,parameters):

    # unwrap parameters 
    theta = parameters[0]             # (degrees)
    distance = parameters[1]          # (nm)
    pixel_size = parameters[2]        # (nm)
    [num_images,ny,nx]=data.shape     # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end+ny*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = np.int64(np.ceil(ny*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = np.int64(nx)                                           # (pixels)

    # create final image
    output = np.zeros((final_nz, final_ny, final_nx),dtype=np.float32)  # (pixels,pixels,pixels - data is float32)

    # precalculate trig functions for scan angle
    tantheta = np.float32(np.tan(theta * np.pi/180)) # (float32)
    sintheta = np.float32(np.sin(theta * np.pi/180)) # (float32)
    costheta = np.float32(np.cos(theta * np.pi/180)) # (float32)

    # perform orthogonal interpolation

    # loop through output z planes
    # defined as parallel loop in numba
    # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
    for z in prange(0,final_nz):
        # calculate range of output y pixels to populate
        y_range_min=np.minimum(0,np.int64(np.floor(np.float32(z)/tantheta)))
        y_range_max=np.maximum(final_ny,np.int64(np.ceil(scan_end+np.float32(z)/tantheta+1)))

        # loop through final y pixels
        # defined as parallel loop in numba
        # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
        for y in prange(y_range_min,y_range_max):

            # find the virtual tilted plane that intersects the interpolated plane 
            virtual_plane = y - z/tantheta

            # find raw data planes that surround the virtual plane
            plane_before = np.int64(np.floor(virtual_plane/pixel_step))
            plane_after = np.int64(plane_before+1)

            # continue if raw data planes are within the data range
            if ((plane_before>=0) and (plane_after<num_images)):
                
                # find distance of a point on the  interpolated plane to plane_before and plane_after
                l_before = virtual_plane - plane_before * pixel_step
                l_after = pixel_step - l_before
                
                # determine location of a point along the interpolated plane
                za = z/sintheta
                virtual_pos_before = za + l_before*costheta
                virtual_pos_after = za - l_after*costheta

                # determine nearest data points to interpoloated point in raw data
                pos_before = np.int64(np.floor(virtual_pos_before))
                pos_after = np.int64(np.floor(virtual_pos_after))

                # continue if within data bounds
                if ((pos_before>=0) and (pos_after >= 0) and (pos_before<ny-1) and (pos_after<ny-1)):
                    
                    # determine points surrounding interpolated point on the virtual plane 
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    output[z,y,:] = (l_before * dz_after * data[plane_after,pos_after+1,:] + \
                                    l_before * (1-dz_after) * data[plane_after,pos_after,:] + \
                                    l_after * dz_before * data[plane_before,pos_before+1,:] + \
                                    l_after * (1-dz_before) * data[plane_before,pos_before,:]) /pixel_step


    # return output
    return output

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument 
    input_dir_string = ''
    output_dir_string = ''

    try:
        arguments, values = getopt.getopt(argv,"hi:o:n:c:",["help","ipath=","opath="])
    except getopt.GetoptError:
        print('Error. stage_recon.py -i <inputdirectory> -o <outputdirectory>')
        sys.exit(2)
    for current_argument, current_value in arguments:
        if current_argument == '-h':
            print('Usage. stage_recon.py -i <inputdirectory> -o <outputdirectory>')
            sys.exit()
        elif current_argument in ("-i", "--ipath"):
            input_dir_string = current_value
        elif current_argument in ("-o", "--opath"):
            output_dir_string = current_value
        
    if (input_dir_string == ''):
        print('Input parse error.')
        sys.exit(2)

    # Load data
    # this approach assumes data is generated by QI2lab pycromanager control code

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # determine number of directories in root directory. loop over each one.
    tile_dir_path = [f for f in input_dir_path.iterdir() if f.is_dir()]
    num_tiles = len(tile_dir_path)

    # load first tile to get dataset dimensions
    tile_dir_path_to_load = tile_dir_path[1]
    dataset = Dataset(tile_dir_path_to_load)
    dask_array = dataset.as_array()
    num_channels = dask_array.shape[1]
    del dataset
    del dask_array

    # parse all directories to find number of unique y and z positions
    y_values = np.empty([num_tiles])
    z_values = np.empty([num_tiles])

    for tile in range(num_tiles):
        tile_dir_path_to_load = tile_dir_path[tile]
        test_string = tile_dir_path_to_load.parts[-1].split('_')
        for i in range(len(test_string)):
            if 'y0' in test_string[i]:
                y_values[tile] = int(test_string[i].split('y')[1])
            if 'z0' in test_string[i]:
                z_values[tile] = int(test_string[i].split('z')[1])

    num_y=np.unique(y_values).shape[0]
    num_z=np.unique(z_values).shape[0]

    # create parameter array from scan parameters saved by acquisition code
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    df_stage_scan_params = pd.read_pickle(input_dir_path / 'stage_scan_params.pkl')
    stage_scan_params = df_stage_scan_params.to_numpy()
    params=np.array(stage_scan_params,dtype=np.float32)

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    output_path = output_dir_path / 'full.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_channels, ntiles=(num_z*num_y)+1, \
        subsamp=((1,1,1),(2,2,2),(4,4,4),(8,8,8),(16,16,16)),blockdim=((16, 16, 8),))

    # open stage positions file
    df_stage_positions = pd.read_pickle(input_dir_path / 'stage_positions.pkl')

    # calculate pixel sizes of deskewed image in microns
    deskewed_x_pixel = stage_scan_params[1] / 100.
    deskewed_y_pixel = stage_scan_params[1] / 100.
    deskewed_z_pixel = stage_scan_params[1]*np.sin(stage_scan_params[0] * np.pi/180.) / 100.

    # create blank affine transformation to use for stage translation
    unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                    (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                    (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file
    # TO DO: implement directory polling to do this in the background while data is being acquired.
    for tile_id in range(num_tiles):

        # load tile
        tile_dir_path_to_load = tile_dir_path[tile]
        print('Loading directory: '+str(tile_dir_path_to_load))

        # decode directory name to determine which stage position to load
        test_string = tile_dir_path_to_load.parts[-1].split('_')
        for i in range(len(test_string)):
            if 'y0' in test_string[i]:
                y_idx = int(test_string[i].split('y')[1])
            if 'z0' in test_string[i]:
                z_idx = int(test_string[i].split('z')[1])

        # load correct stage position from dataframe
        df_current_stage = df_stage_positions.loc[(df['tile_y'] == y_idx) & df['tile_z'] == z_idx]
        stage_x = df_current_stage['stage_x']
        stage_y = df_current_stage['stage_y']
        stage_z = df_current_stage['stage_z']

        print('y index: '+str(y_idx)+' z index: '+str(z_idx)+' H5 tile id: '+str(tile_id))

        # https://pycro-manager.readthedocs.io/en/latest/read_data.html
        dataset = Dataset(tile_dir_path_to_load)
        dask_array = dataset.as_array()
        num_images = dask_array.shape[0]

        # loop over channels inside tile
        for channel_id in range(num_channels):

            # read images from dataset. Skip first 10 images due to stage coming up to speed.
            sub_stack = dask_array[10:num_images,channel_id,:,:].compute()

            print('Deskew tile.')
            # run deskew
            deskewed = stage_deskew(data=np.flipud(sub_stack),parameters=params)
            del sub_stack
            gc.collect()

            # downsample by 2x in z due to oversampling when going from OPM to coverslip geometry
            deskewed_downsample = block_reduce(deskewed, block_size=(2,1,1), func=np.mean)
            del deskewed
            gc.collect()

            # create affine transformation for stage translation
            affine_matrix = unit_matrix
            affine_matrix[0,3] = stage_x / deskewed_y_pixel # x-translation
            affine_matrix[1,3] = stage_y / deskewed_y_pixel # y-translation
            affine_matrix[2,3] = stage_z / deskewed_z_pixel # y-translation

            print('Write tiles.')
            bdv_writer.append_view(deskewed_downsample, time=0, channel=channel_id, 
                                    tile=tile_id,
                                    voxel_size_xyz=(deskewed_y_pixel, deskewed_y_pixel, 2*deskewed_z_pixel), 
                                    voxel_units='um',
                                    m_affine=affine_matrix,
                                    name_affine = 'tile '+str(tile_id)+' translation')

            # free up memory
            del deskewed_downsample
            #del deskewed_downsample
            gc.collect()

    # write BDV xml file
    # https://github.com/nvladimus/npy2bdv
    bdv_writer.write_xml_file(ntimes=1)
    bdv_writer.close()

    # clean up memory
    gc.collect()


# run
if __name__ == "__main__":
    main(sys.argv[1:])


# The MIT License
#
# Copyright (c) 2020 Douglas Shepherd, Arizona State University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.