#!/usr/bin/env python

'''
Stage orthogonal interpolation code using Numpy, Dask, and npy2bdv
Orthgonal interpolation method as described by Vincent Maioli (http://doi.org/10.25560/68022)

Shepherd 03/20
'''

import numpy as np
from dask_image.imread import imread
from dask.distributed import LocalCluster, Client
import dask.array as da
from functools import partial
from pathlib import Path
import npy2bdv
import gc

# this is just a wrapper because the stage_recon function
# expects ndims==2 but our blocks will have ndim==3
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
    output = np.zeros([final_nz, final_ny, final_nx])           # (pixels,pixels,pixels)

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

    output = output/pixel_step

    return output

def main(directory):

    # open create glob-like path to directory with data
    # TO DO: should we do this for each strip on it's own?
    #        or can dask-image handle all strips in one big dask array
    directory_path = Path(directory)
    directory_path = directory_path / "*.tif"

    # read data in using dask-image
    stack_raw = imread(directory_path)
    
    # get number of strips from raw data
    number_of_strips = stack_raw.shape[0]

    # create parameter array
    params=[30,400,116]

    # https://docs.python.org/3.8/library/functools.html#functools.partial
    # use this to define function that map_blocks will call to figure out how to map data
    function_deskew = last3dims(partial(stage_recon, params))

    # create BDV file
    output_path = directory / 'deskewed.h5'
    bdv_writer = npy2bdv.BdvWriter(output_path, nchannels=1, ntiles=number_of_strips, \
    subsamp=((1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)), blockdim=((256, 32, 256),))

    # loop over each strip
    for strip in range (0,number_of_strips):

        # use map_blocks to have dask manage what pieces of the array to pull from disk into memory
        strip_deskew = stack_raw[strip,:].map_blocks(function_deskew,dtype="uint16")

        # evaluate into numpy array held in local memory
        # TO DO: is it possible to use npy2bdv with Dask? Or rewrite it so that works?
        strip_deskew.compute()

        # write strip into BDV H5 file
        bdv_writer.append_view(strip_deskew, time=0, channel=0, tile=strip)

        # make sure we free up memory
        del strip_deskew
        gc.collect()

    # write xml file
    bdv_writer.write_xml_file(ntimes=1)
    bdv_writer.close()