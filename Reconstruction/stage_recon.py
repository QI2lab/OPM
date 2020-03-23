#!/usr/bin/env python

'''
Stage scanning OPM post-processing using Numpy, Dask, and npy2bdv
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 03/20
'''

import numpy as np
from dask_image.imread import imread
import dask.array as da
from functools import partial
from pathlib import Path
import npy2bdv
import gc
import sys
import getopt

# this is just a wrapper because the stage_recon function
# expects ndims==3 but our blocks will have ndim==4
def last3dims(f):

    def func(array):
        return f(array[0])[None, ...]
    return func

# perform stage scanning reconstruction
def stage_recon(data,params):

    # unwrap parameters 
    theta = params[0]                   # (degrees)
    distance = params[1]                # (nm)
    pixel_size = params[2]              # (nm)
    [num_images,ny,nx]=data.shape()     # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = distance/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.ceil(scan_end+ny*np.cos(theta*np.pi/180)).astype(int) # (pixels)
    final_nz = np.ceil(ny*np.sin(theta*np.pi/180)).astype(int)          # (pixels)
    final_nx = int(nx)                                                  # (pixels)

    # create final image
    output = np.zeros([final_nz, final_ny, final_nx])   # (pixels,pixels,pixels)

    # precalculate trig functions for scan angle
    tantheta = np.tan(theta * np.pi/180) # (float)
    sintheta = np.sin(theta * np.pi/180) # (float)
    costheta = np.cos(theta * np.pi/180) # (float)

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
            plane_before = np.floor(virtual_plane/pixel_step)
            plane_after = plane_before+1

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
                pos_before = np.floor(virtual_pos_before)
                pos_after = np.floor(virtual_pos_after)

                # continue if within data bounds
                if ((pos_before>=0) and (pos_after >= 0) and (pos_before<ny-1) and (pos_after<ny-1)):
                    
                    # determine points surrounding interpolated point on the virtual plane 
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    output[z,y,:] = l_before * dz_after * data[plane_after,pos_after+1,:] + \
                                    l_before * (1-dz_after) * data[plane_after,pos_after,:] + \
                                    l_after * dz_before * data[plane_before,pos_before+1,:] + \
                                    l_after * (1-dz_before) * data[plane_before,pos_before,:] 

    # scale output image by pixel size
    output = output/pixel_step

    # return array
    return output

def main(argv):

    # parse directory name from command line argument
    input_dir_string = ''
    output_dir_string = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile=","nimgs="])
    except getopt.GetoptError:
        print('stage_recon.py -i <inputdirectory> -o <outputdirectory> -n <numberimages>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('stage_recon.py -i <inputdirectory> -o <outputdirectory> -n <numberimages>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_dir_string = arg
        elif opt in ("-o", "--ofile"):
            output_dir_string = arg
        elif opt in ("-n", "--nimgs"):
            num_img_per_strip = arg

    # https://docs.python.org/3/library/pathlib.html
    # open create glob-like path to directory with data
    input_dir_path = Path(input_dir_string)
    input_path = input_dir_path / "*.tif"

    # http://image.dask.org/en/latest/dask_image.imread.html
    # read data in using dask-image
    stack_raw = imread(str(input_path),nframes=20000))
    
    # get number of strips from raw data
    num_strips = int(stack_raw.shape[0]/num_imgs_per_strip)

    # get number of processings strips
    # this number should be adjusted to fit images into memory
    split_strip_factor = 4
    overlap = 0.1
    num_strips_splits = split_strip_factor*num_strips
    num_images_per_split = num_imgs_per_strip/split_strip_factor

    # create parameter array
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    params=[30,400,116]

    # https://docs.python.org/3.8/library/functools.html#functools.partial
    # use this to define function that map_blocks will call to figure out how to map data
    function_deskew = partial(stage_recon, params=params)

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV file with sub-sampling for BigStitcher
    output_path = output_dir_path / 'deskewed.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=1, ntiles=number_of_strips, \
    subsamp=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)), blockdim=((4, 256, 256),))

    # loop over each strip in acquistion
    for strip in range (0,num_strips):
        for i in range(0,split_strip_factor):
            if i==0:
                first_image=i*num_images_per_split
                last_image=(i+1)*num_images_per_split
            else:
                first_image=i*num_images_per_split-int(num_images_per_split * overlap)
                last_image=(i+1)*num_images_per_split

            # https://docs.dask.org/en/latest/array-api.html#dask.array.map_blocks
            # use map_blocks to have dask manage what pieces of the array to pull from disk into memory
            # TO DO: write file splitting code to make sure that machine does not run out of memory
            images_to_load = range(first_image,last_image)
            strip_deskew = stack_raw[images_to_load,:].map_blocks(function_deskew,dtype="uint16")

            # https://docs.dask.org/en/latest/api.html#dask.compute
            # evaluate into numpy array held in local memory
            # need to have enough RAM on computer to use this strategy!
            # e.g. 20,000 image strip * [2020,990] * uint16 gives a final image of 
            # ~330 gb for float32 and ~165 gb for uint16.
            # TO DO: write file splitting code to make sure that machine does not run out of memory
            # TO DO: is it possible to use npy2bdv with Dask? Or rewrite the library so that it works?
            strip_deskew.compute()

            # https://github.com/nvladimus/npy2bdv
            # write strip into BDV H5 file
            # TO DO: is it possible to use npy2bdv with Dask? Or rewrite the library so that it works?
            tile_id=(split_strip_factor)*strip+i
            bdv_writer.append_view(strip_deskew, time=0, channel=0, tile=tile_id)

            # make sure we free up memory
            del strip_deskew
            gc.collect()

    # https://github.com/nvladimus/npy2bdv
    # write xml file
    bdv_writer.write_xml_file(ntimes=1)
    bdv_writer.close()

    # TO DO: create text file with stage positions for BigStitcher

    # clean up memory
    del stack_raw
    gc.collect()

if __name__ == "__main__":
    main(sys.argv[1:])