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
import warnings

# perform stage scanning reconstruction
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
    final_nz = int(np.ceil(ny*np.sin(theta*np.pi/180)))         # (pixels)
    final_nx = int(nx)                                          # (pixels)

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


    # return array
    return output

def main(argv):

    # parse directory name from command line argument
    input_dir_string = ''
    output_dir_string = ''
    '''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile=","nimgs="])
    except getopt.GetoptError:
        print('Error. stage_recon.py -i <inputdirectory> -o <outputdirectory> -n <numberimages>')
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
    '''

    input_dir_string='D:\whole_brain_trial4\dapi_7'
    output_dir_string = 'D:\whole_brain_trial4' 
    num_imgs_per_strip = 2155

    # https://docs.python.org/3/library/pathlib.html
    # open create glob-like path to directory with data
    #input_dir_path = Path(input_dir_string)
    #input_path = input_dir_path / "*.tif"
    
    input_dir_path=Path(input_dir_string)
    all_files = list(input_dir_path.glob('*.tif'))
    files = [imread(str(all_files[i]),nframes=100) for i in range(len(all_files))]
    stack_raw = da.concatenate(files)

    # http://image.dask.org/en/latest/dask_image.imread.html
    # read data in using dask-image
    #stack_raw = imread(str(input_path),nframes=num_imgs_per_strip)
    
    # get number of strips from raw data
    num_strips = int(stack_raw.shape[0]/num_imgs_per_strip)

    # get number of processings strips
    # this number should be adjusted to fit images into memory
    split_strip_factor = 5
    overlap = 0.1
    num_images_per_split = num_imgs_per_strip/split_strip_factor

    # rechunk dask array to number of images per split strip for faster loading
    # have to do this because data stored in max 4 gig TIFF files created by MM 2.0 gamma 
    # stack_raw = stack_raw.rechunk((10, stack_raw.shape[1],stack_raw.shape[2]))

    # create parameter array
    # [theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    params=[30,116,116]

    # https://docs.python.org/3.8/library/functools.html#functools.partial
    # use this to define function that map_blocks will call to figure out how to map data
    #function_deskew = partial(stage_recon, parameters=params)

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV file with sub-sampling for BigStitcher
    output_path = output_dir_path / 'deskewed.h5'
    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=1, ntiles=num_strips*split_strip_factor, \
    subsamp=((1, 1, 1), (4, 2, 4), (16, 4, 16)), blockdim=((256, 32, 256),))

    # loop over each strip in acquistion
    for strip in range (0,num_strips):
        for i in range(0,split_strip_factor):
            if i==0:
                first_image=(strip*num_imgs_per_strip)+i*num_images_per_split
                last_image=(strip*num_imgs_per_strip)+(i+1)*num_images_per_split
            else:
                first_image=((strip*num_imgs_per_strip))+i*num_images_per_split-int(num_images_per_split * overlap)
                last_image=((strip*num_imgs_per_strip))+(i+1)*num_images_per_split

            tile_id=(split_strip_factor)*strip+i
            print('Computing tile ' +str(tile_id+1)+' out of '+str(num_strips*split_strip_factor))

            # https://docs.dask.org/en/latest/api.html#dask.compute
            # evaluate into numpy array held in local memory
            # need to have enough RAM on computer to use this strategy!
            # TO DO: write file splitting code to make sure that machine does not run out of memory
            # TO DO: is it possible to use Dask map_blocks with deskew function?
            stack_to_process = stack_raw[first_image:last_image,:].compute()
            strip_deskew = stage_recon(stack_to_process,params)
           
            # look into https://github.com/dask/dask-image/issues/110
            # this might give a way to write the TIFFs out and then read into HDF5

            # https://github.com/nvladimus/npy2bdv
            # write strip into BDV H5 file
            # TO DO: is it possible to use npy2bdv with Dask? Or rewrite the library so that it works?
            print('Writing tile '+str(tile_id+1))
            bdv_writer.append_view(strip_deskew, time=0, channel=0, tile=tile_id)

            # keep track of location
            pos[tile_id,:]=[strip*]

            # make sure we free up memory
            del strip_deskew
            del stack_to_process
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