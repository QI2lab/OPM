#!/usr/bin/env python

'''
Stage scanning OPM post-processing using numpy, numba, skimage, and npy2bdv.
New version to handle altered acquisition code for multi-color large area stage scans. 
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 03/20
'''

# imports
import numpy as np
from pathlib import Path
from natsort import natsorted, ns
import npy2bdv
import gc
import sys
import getopt
import re
import skimage.io as io
from skimage.measure import block_reduce
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
    [num_images,ny,nx]=data.shape  # (pixels)


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
    # this approach assumes data is generated by QI2lab MM script
    # the strategy is to sequentially activate each channel for each tile position
    # this allows for smooth stage scanning at a fast camera rate without having to synchronize
    # laser changing during stage scan

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # Parse directory for number of channels and strip positions then sort
    sub_dirs = [x for x in input_dir_path.iterdir() if x.is_dir()]
    sub_dirs = natsorted(sub_dirs, alg=ns.PATH)

    # TO DO: automatically determine number of channels and tile positions
    num_channels=3
    num_tiles=22

    # create parameter array
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    params=np.array([30,200,116],dtype=np.float32)

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    # TO DO: modify npy2bdv to support B3D compression, https://git.embl.de/balazs/B3D
    #        this may involve change the underlying hdf5 install that h5py is using
    output_path = output_dir_path / 'deskewed_20200713.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_channels, ntiles=4*num_tiles, \
        subsamp=((1,1,1),(2,2,2),(4,4,4),(8,8,8),),blockdim=((16, 16, 16),))
    
    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file
    # TO DO: implement directory polling to do this in the background while data is being acquired.
    for sub_dir in sub_dirs:

        # determine the channel this directory corresponds to
        m = re.search('ch(\d+)', str(sub_dir), re.IGNORECASE)
        channel_id = int(m.group(1))

        if (channel_id==3):
            channel_id = channel_id - 1

        # determine the experimental tile this directory corresponds to
        m = re.search('y(\d+)', str(sub_dir), re.IGNORECASE)
        tile_id = int(m.group(1))

        # output metadata information to console
        print('Channel ID: '+str(channel_id)+'; Experimental tile ID: '+str(tile_id)+ \
            '; BDV tile IDs: '+str(4*tile_id)+' - '+str(4*tile_id+3))

        # load bright field image for this channel
        bright_field_file = input_dir_path / Path('ch0_flatfield.tif')
        bright_field = np.asarray(io.imread(bright_field_file),dtype=np.float32)
        
        # find all individual tif files in the current channel + tile sub directory and sort 
        files = natsorted(sub_dir.glob('*.tif'), alg=ns.PATH)

        # flip order so that light sheet tilt is along scan direction
        # files.reverse()

        # find middle of tilted plane acquisition
        num_split = 4
        split = len(files)//num_split
        overlap = np.int64(np.floor(split*.05))

        print('Deskew block 1.')
        # read in first block of data with a small overlap for alignment in BigStitcher
        sub_stack = np.asarray([io.imread(file) for file in files[0:split+overlap]],dtype=np.float32)
        sub_stack = sub_stack/bright_field
        

        # run deskew for the first block of data
        deskewed = stage_deskew(data=sub_stack,parameters=params)
        deskewed_downsample = block_reduce(deskewed,block_size=(2,1,1),func=np.mean)
        del deskewed
        del sub_stack
        gc.collect()

        print('Writing deskewed block 1.')
        # write BDV tile
        # https://github.com/nvladimus/npy2bdv 
        bdv_writer.append_view(deskewed_downsample, time=0, channel=channel_id, tile=4*tile_id, voxel_size_xyz=(116, 116, 200), voxel_units='nm')
        # free up memory
        del deskewed_downsample
        gc.collect()

        print('Deskew block 2.')
        # read in second block of data with a small overlap for alignment in BigStitcher
        sub_stack = np.asarray([io.imread(file) for file in files[split-overlap:2*split+overlap]],dtype=np.float32)
        sub_stack = sub_stack/bright_field

        # run deskew for the second block of data
        deskewed = stage_deskew(data=sub_stack,parameters=params)
        deskewed_downsample = block_reduce(deskewed,block_size=(2,1,1),func=np.mean)
        del deskewed
        del sub_stack
        gc.collect()

        print('Writing deskewed block 2.')
        # write BDV tile
        # https://github.com/nvladimus/npy2bdv 
        bdv_writer.append_view(deskewed_downsample, time=0, channel=channel_id, tile=4*tile_id+1, voxel_size_xyz=(116, 116, 200), voxel_units='nm')

        # free up memory
        del deskewed_downsample
        gc.collect()

        print('Deskew block 3.')
        # read in second block of data with a small overlap for alignment in BigStitcher
        sub_stack = np.asarray([io.imread(file) for file in files[2*split-overlap:3*split+overlap]],dtype=np.float32)
        sub_stack = sub_stack/bright_field

        # run deskew for the second block of data
        deskewed = stage_deskew(data=sub_stack,parameters=params)
        deskewed_downsample = block_reduce(deskewed,block_size=(2,1,1),func=np.mean)
        del deskewed
        del sub_stack
        gc.collect()

        print('Writing deskewed block 3.')
        # write BDV tile
        # https://github.com/nvladimus/npy2bdv 
        bdv_writer.append_view(deskewed_downsample, time=0, channel=channel_id, tile=4*tile_id+2, voxel_size_xyz=(116, 116, 200), voxel_units='nm')

        # free up memory
        del deskewed_downsample
        gc.collect()

        print('Deskew block 4.')
        # read in second block of data with a small overlap for alignment in BigStitcher
        sub_stack = np.asarray([io.imread(file) for file in files[3*split-overlap:]],dtype=np.float32)
        sub_stack = sub_stack/bright_field

        # run deskew for the second block of data
        deskewed = stage_deskew(data=sub_stack,parameters=params)
        deskewed_downsample = block_reduce(deskewed,block_size=(2,1,1),func=np.mean)
        del deskewed
        del sub_stack
        gc.collect()

        print('Writing deskewed block 4.')
        # write BDV tile
        # https://github.com/nvladimus/npy2bdv 
        bdv_writer.append_view(deskewed_downsample, time=0, channel=channel_id, tile=4*tile_id+3, voxel_size_xyz=(116, 116, 200), voxel_units='nm')

        # free up memory
        del deskewed_downsample
        del bright_field
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