#!/usr/bin/env python

'''
Stage scanning OPM post-processing using Numpy, Dask, and npy2bdv
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 03/20
'''

# imports
import numpy as np
from dask_image.imread import imread
import dask.array as da
from functools import partial
from pathlib import Path
from natsort import natsorted, ns
import npy2bdv
import gc
import sys
import getopt
import re

# perform stage scanning reconstruction using orthogonal interpolation
def stage_recon(data,parameters):

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
    final_ny = int(np.ceil(scan_end+ny*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = int(np.ceil(ny*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = int(nx)                                           # (pixels)

    # create final image
    output = np.zeros([final_nz, final_ny, final_nx],dtype=np.float32)  # (pixels,pixels,pixels)

    # precalculate trig functions for scan angle
    tantheta = np.tan(theta * np.pi/180).astype('f') # (float)
    sintheta = np.sin(theta * np.pi/180).astype('f') # (float)
    costheta = np.cos(theta * np.pi/180).astype('f') # (float)

    # perform orthogonal interpolation

    # loop through output z planes
    for z in range (0,final_nz):
        # calculate range of output y pixels to populate
        y_range_min=np.min([0,int(np.floor(float(z)/tantheta))])
        y_range_max=np.max([final_ny,int(np.ceil(scan_end+float(z)/tantheta+1))])

        # loop through final y pixels
        for y in range(y_range_min,y_range_max):

            # find the virtual tilted plane that intersects the interpolated plane 
            virtual_plane = y - z/tantheta

            # find raw data planes that surround the virtual plane
            plane_before = int(np.floor(virtual_plane/pixel_step))
            plane_after = int(plane_before+1)

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
                pos_before = int(np.floor(virtual_pos_before))
                pos_after = int(np.floor(virtual_pos_after))

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


    # return interpolated output
    return output

# open data, parse, chunk, perform OPM processing, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument
    # TO DO: parse number of images in strip, number of strips, and number of channels from MM metadata 
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
    # this approach assumes data is generated by our MM script
    # the strategy is to sequentially activate each channel for each tile position ("strip")
    # this allows for smooth stage scanning at a fast camera rate without having to synchronize
    # laser changing during stage scan

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # Parse directory for number of channels and strip positions
    sub_dirs = [x for x in input_dir_path.iterdir() if x.is_dir()]

    num_channels=4
    num_tiles=30

    # create parameter array
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    params=[60,100,116]

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    # TO DO: modify npy2bdv to support B3D compression, https://git.embl.de/balazs/B3D
    #        this may involve change the underlying hdf5 install that h5py is using
    output_path = output_dir_path / 'deskewed.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_channels, ntiles=num_tiles, \
    subsamp=((1, 1, 1), (4, 4, 4), (8, 8, 8), (16,16,16)), blockdim=((256, 256, 32),))

    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file
    for sub_dir in sub_dirs:

        # determine the channel this directory corresponds to
        m = re.search('ch(\d+)', str(sub_dir), re.IGNORECASE)
        channel_id = m.group(1)

        # determine the tile this directory corresponds to
        m = re.search('y(\d+)', str(sub_dir), re.IGNORECASE)
        tile_id = m.group(1)

        # read all strips for each channel. Then place into larger array
        file_pattern = 'input_dir_path/sub_dir/*.tif'
        stack = imread(file_pattern)

        # setup processing
        deskewed = stack.map_blocks(stage_recon, parameters=params)

        # write BDV tile
        bdv_writer.append_view(deskewed, time=0, channel=channel_id, tile=tile_id)


        # free up memory
        del deskewed
        del stack
        gc.collect()

    # https://github.com/nvladimus/npy2bdv
    # write BDV xml file
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