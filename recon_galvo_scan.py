#!/usr/bin/env python

'''
Galvo scanning OPM post-processing using numpy, numba, skimage, pyimagej, and npy2bdv.
Orthgonal interpolation method adapted from original description by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 02/21
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv # have to install from source!
import sys
import getopt
import re
from skimage.measure import block_reduce
from numba import njit, prange
import pandas as pd
import flat_field
import imagej
import scyjava
import dask.array as da

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
    # Data must be generated by QI2lab pycromanager control code
    # https://www.github.com/qi2lab/OPM/

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # load first tile to get dataset dimensions
    dataset = Dataset(str(input_dir_path))
    dask_array = dataset.as_array()

    print(dask_array.shape)
    exit()

    num_timepoints = dask_array.shape[0]
    num_scan_axis =  dask_array.shape[1]

    if (len(dask_array.shape) == 4):
        num_channels = 1
        num_pix_y = dask_array.shape[2]
        num_pix_x = dask_array.shape[3]
    else:
        num_channels = dask_array.shape[2]
        num_pix_y = dask_array.shape[3]
        num_pix_x = dask_array.shape[4]
    del dataset
    
    # create parameter array from scan parameters saved by acquisition code
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    df_stage_scan_params = pd.read_pickle(input_dir_path / 'galvo_scan_params.pkl')
    params = np.squeeze(df_stage_scan_params.to_numpy())

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    output_path = output_dir_path / 'full.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_channels, ntiles=1, \
        subsamp=((1,1,1),(2,2,2),(4,4,4),(8,8,8),(16,16,16)),blockdim=((16, 16, 16),))

    # calculate pixel sizes of deskewed image in microns
    # Cannot use a different pixel size in (x,y) in BigStitcher, so calculate here for posterity
    deskewed_x_pixel = np.round(params[2]*np.cos(params[0] * np.pi/180.),0) / 1000.
    deskewed_y_pixel = np.round(params[2],0) / 1000.
    deskewed_z_pixel = np.round(params[2]*np.sin(params[0] * np.pi/180.),1) / 1000.

    # amount of down sampling in z
    z_down_sample = 1

    # create blank affine transformation to use for stage translation
    unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                            (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

    # loop over timepoints
    for timepoint_id in range(num_timepoints):
        # loop over channels inside tile
        for channel_id in range(num_channels):

            # read images from dataset. Skip first 10 images due to stage coming up to speed.
            print('Load raw data for timepoint: ' + str(timepoint_id) + '; channel: '+str(channel_id))
            if num_channels == 1:
                sub_stack = dask_array[timepoint_id,:,:,:]
            else:
                sub_stack = dask_array[timepoint_id,:,channel_id,:,:]

            print(sub_stack.shape)

            # deskew
            print('Deskew channel '+str(channel_id))
            deskewed = stage_deskew(data=np.flipud(sub_stack.compute()),parameters=params)
            #deskewed = stage_deskew(data=sub_stack.compute(),parameters=params)
            del sub_stack

            # downsample by 2x in z due to oversampling when going from OPM to coverslip geometry
            #deskewed_downsample = block_reduce(deskewed, block_size=(z_down_sample,1,1), func=np.mean)
            #del deskewed

            # save tile in BDV H5 with actual stage positions
            print('Write deskewed data for channel '+str(channel_id))
            bdv_writer.append_view(deskewed, time=timepoint_id, channel=channel_id, 
                                    tile=0,
                                    voxel_size_xyz=(deskewed_y_pixel, deskewed_y_pixel, z_down_sample*deskewed_z_pixel), 
                                    voxel_units='um')

            # free up memory
            del deskewed

    # write BDV xml file
    # https://github.com/nvladimus/npy2bdv
    bdv_writer.write_xml_file(ntimes=num_timepoints)
    bdv_writer.close()

    # exit
    sys.exit()

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