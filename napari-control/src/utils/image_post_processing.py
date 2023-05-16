'''
QI2lab OPM suite
Reconstruction tools

Image processing tools for OPM reconstruction

Last updated: Shepherd 01/22 - changes to include map_blocks based dexp deconvolution and recent other changes.
'''

#!/usr/bin/env python
import sys
import numpy as np
from pathlib import Path
#from tifffile import tifffile
from numba import njit, prange
#from flat_field import calc_flatfield
from functools import partial
import dask.array as da
from dask.diagnostics import ProgressBar
import gc

try:
    import cupy as cp
    CP_AVAILABLE = True
except:
    CP_AVAILABLE = False

if CP_AVAILABLE:
    try:
        import microvolution_py as mv_decon
        DECON_LIBRARY = 'mv'
    except:
        try:
            from clij2fft.richardson_lucy import richardson_lucy_nc
            from clij2fft.richardson_lucy import getlib
            DECON_LIBRARY = 'clij'
        except:
            DECON_LIBRARY = None
else:
    DECON_LIBRARY = None

# http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
@njit(parallel=True)
def deskew(data,theta,distance,pixel_size):
    """
    Perform parallelized orthogonal interpolation into a uniform pixel size grid.
    
    :param data: ndarray
        image stack of uniformly spaced OPM planes
    :param theta: float 
        angle relative to coverslip
    :param distance: float 
        step between image planes along coverslip
    :param pizel_size: float 
        in-plane camera pixel size in OPM coordinates

    :return output: ndarray
        image stack of deskewed OPM planes on uniform grid
    """

    # unwrap parameters 
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
    output = np.zeros((final_nz, final_ny, final_nx),dtype=np.float32)  # (time, pixels,pixels,pixels - data is float32)

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
                    output[z,y,:] = (l_before * dz_after * data[plane_after,pos_after+1,:] +
                                    l_before * (1-dz_after) * data[plane_after,pos_after,:] +
                                    l_after * dz_before * data[plane_before,pos_before+1,:] +
                                    l_after * (1-dz_before) * data[plane_before,pos_before,:]) /pixel_step


    # return output
    return output


def manage_flat_field_py(stack):
    """
    Manage performing flat and dark-field using python adapation of BaSiC algorithm.

    Returns flat- and darkfield corrected image
    
    :param stack: ndarray
        matrix of OPM planes

    :return corrected_stack: ndarray of deskewed OPM planes on uniform grid
    """
  
    num_images = 500

    if stack.shape[0] > num_images:
        stack_for_flat_field = stack[np.random.choice(stack.shape[0], num_images, replace=False)]
    else:
        stack_for_flat_field = stack

    flat_field, dark_field = calc_flatfield(images=stack_for_flat_field)
    corrected_stack = perform_flat_field(flat_field,dark_field,stack)

    return corrected_stack, flat_field, dark_field

def perform_flat_field(flat_field,dark_field,stack):
    """
    Calculate flat- and darkfield corrected image. Returns corrected image.
    
    :param flat_field: ndarray
        flatfield correction
    :param dark_field: ndarray
        darkfield correction
    :param stack: dask.array
        matrix of OPM planes

    :return corrected_stack: ndarray
        corrected OPM image planes 
    """

    #dark_field[dark_field>50]=50
    #corrected_stack = stack.astype(np.float32) - dark_field
    stack[stack<0] = 0 
    corrected_stack = stack/flat_field

    return corrected_stack

def lr_deconvolution(image,psf,iterations=5):
    """
    Tiled Lucy-Richardson deconvolution using DECON_LIBRARY

    :param image: ndarray
        raw data
    :param psf: ndarray
        theoretical PSF
    :param iterations: int
        number of iterations to run 
    :return deconvolved: ndarray
        deconvolved image
    """

    # create dask array and apodization window
    scan_chunk_size = 64
    if image.shape[0]<scan_chunk_size:
        dask_raw = da.from_array(image,chunks=(image.shape[0],image.shape[1],image.shape[2]))
        overlap_depth = (0,0,0)
    else:
        dask_raw = da.from_array(image,chunks=(scan_chunk_size,image.shape[1],image.shape[2]))
        overlap_depth = (psf.shape[0],0,0)
    del image
    gc.collect()

    if DECON_LIBRARY=='mv':
        lr_dask = partial(mv_lr_decon,psf=psf,num_iterations=iterations)
    elif DECON_LIBRARY=='clij':
        lib = getlib()
        lr_dask = partial(clij_lr,psf=psf,num_iters=iterations,tau=0.0001,lib=lib)

    # create dask plan for overlapped blocks
    dask_decon = da.map_overlap(lr_dask,dask_raw,depth=overlap_depth,boundary='reflect',trim=True,meta=np.array((), dtype=np.uint16))

    # perform LR deconvolution in blocks
    with ProgressBar():
        decon_data = dask_decon.compute(scheduler='single-threaded')

    # clean up memory
    del dask_decon, lib
    gc.collect()

    cp.clear_memo()
    cp._default_memory_pool.free_all_blocks()

    return decon_data.astype(np.uint16)

def mv_lr_decon(image,psf,num_iterations):
    '''
    Lucy-Richardson deconvolution using commerical Microvolution library. 

    :param image: ndarray
        raw image
    :param ch_idx: int
        wavelength index
    

    :return image: ndarray
        deconvolved image 
    '''

    params = mv_decon.DeconParameters()
    params.generatePsf = False
    params.nx = image.shape[2]
    params.ny = image.shape[1]
    params.nz = image.shape[0]
    params.blind = False
    params.psfNx = psf.shape[2]
    params.psfNy = psf.shape[1]
    params.psfNz = psf.shape[0]
    params.dr = 115.0
    params.dz = 400.0
    params.psfDr = 115.0
    params.psfDz = 400.0
    params.iterations = num_iterations
    params.background = 100
    params.regularizationType=mv_decon.RegularizationType_TV
    params.scaling = mv_decon.Scaling_U16

    try:
        launcher = mv_decon.DeconvolutionLauncher()
        image = image.astype(np.float16)

        launcher.SetParameters(params)
        for z in range(params.nz):
            launcher.SetImageSlice(z, image[z,:])

        psf_image = psf.astype(np.float16)
        for z in range(params.psfNz):
            launcher.SetPsfSlice(z,psf_image[z,:])

        new_image = np.zeros(image.shape,dtype=np.uint16)
        del image

        launcher.Run()

        for z in range(params.nz):
            launcher.RetrieveImageSlice(z, new_image[z,:])

    except:
        err = sys.exc_info()
        print("Unexpected error:", err[0])
        print(err[1])
        print(err[2])
        new_image = np.zeros(image.shape,dtype=np.uint16)

    return new_image.astype(np.uint16)

def clij_lr(image,psf,num_iters,tau,lib):

    result = richardson_lucy_nc(image.astype(np.float32),psf.astype(np.float32),num_iters,tau,lib)

    return result.astype(np.uint16)