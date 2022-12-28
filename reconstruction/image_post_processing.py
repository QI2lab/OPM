'''
QI2lab OPM suite
Reconstruction tools

Image processing tools for OPM reconstruction

Last updated: Shepherd 12/22 - Remove dexp, use cucim L-R, add CuPy verison of periodic-smooth decomposition before L-R decon

To install cucim on windows:
pip install -e "git+https://github.com/rapidsai/cucim.git@v22.12.00#egg=cucim&subdirectory=python/cucim"

Check version number (...@v22.12.00...) for most recent RapidsAI API to install 
'''

#!/usr/bin/env python
import sys
import numpy as np
from pathlib import Path
from tifffile import tifffile
from numba import njit, prange
from flat_field import calc_flatfield
from functools import partial
import dask.array as da
from dask.diagnostics import ProgressBar
import gc

try:
    import cupy as cp
    GPU_AVAILABLE=True
except:
    GPU_AVAILABLE=False

if GPU_AVAILABLE:
    try:
        import microvolution_py as mv_decon
        DECON_LIBRARY = 'mv'
    except:
        try:
            from clij2fft.richardson_lucy import richardson_lucy_nc
            import clij2fft
            DECON_LIBRARY = 'clij'
        except:
            try:
                from cupyx.scipy.signal import fftconvolve
                from cucim.skimage.restoration import denoise_tv_chambolle
                DECON_LIBRARY = 'custom'
            except:
                try:
                    from cucim.skimage.restoration import richardson_lucy as cucim_decon
                    DECON_LIBRARY = 'cucim'
                except:
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
    scan_chunk_size = 256
    if image.shape[0]<scan_chunk_size:
        dask_raw = da.from_array(image,chunks=(image.shape[0],image.shape[1],image.shape[2]//2))
        overlap_depth = (psf.shape[0],psf.shape[1],psf.shape[2])
    else:
        dask_raw = da.from_array(image,chunks=(scan_chunk_size,image.shape[1],image.shape[2]//2))
        overlap_depth = (psf.shape[0],psf.shape[1],psf.shape[2])
    del image
    gc.collect()

    if DECON_LIBRARY=='mv':
        lr_dask = partial(mv_lr_decon,psf=psf,num_iterations=iterations)
    elif DECON_LIBRARY=='cucim':
        lr_dask = partial(cucim_lr_decon,psf=psf,num_iter=iterations,clip=False,filter_epsilon=1)
    elif DECON_LIBRARY=='custom':
        lr_dask = partial(custom_lr,psf=psf,num_iters=iterations)
    elif DECON_LIBRARY=='clij':
        lr_dask = partial(clij_lr,psf=psf,num_iters=iterations,tau=.0002)

    # create dask plan for overlapped blocks
    dask_decon = da.map_overlap(lr_dask,dask_raw,depth=overlap_depth,boundary='reflect',trim=True,meta=np.array((), dtype=np.uint16))

    # perform LR deconvolution in blocks
    with ProgressBar():
        decon_data = dask_decon.compute(scheduler='single-threaded')

    # clean up memory
    del dask_decon
    gc.collect()

    cp.clear_memo()
    cp._default_memory_pool.free_all_blocks()

    return decon_data.astype(np.uint16)

def cucim_lr_decon(image,psf,num_iter,clip=False,filter_epsilon=1,camera_bkd=100):
    """
    Lucy-Richardson deconvolution using RapidsAI cuCIM library

    :param image: ndarray
        data tile generated by dask
    :param psf: ndarray
        skewed PSF
    :param num_iters: int
        number of iterations
    :param clip: bool
        clip above np.abs(1). Default False
    :param filter_epsilon:
        set values below this to zero
    :param camera_bkd:
        camera background to substract

    :return result: ndarray
        deconvolved data tile
    """

    # substract camera offset and enforce positivity
    image_cp = cp.asarray(image.astype(np.float32))
    psf_cp = cp.asarray(psf.astype(np.float32))

    result_cp = cucim_decon(image_cp,psf_cp,num_iter=num_iter,clip=clip,filter_epsilon=filter_epsilon)
    result_cp[result_cp<0]=0
    result = cp.asnumpy(result_cp)
    del image_cp, psf_cp, result_cp

    gc.collect()

    
    cp.clear_memo()
    cp._default_memory_pool.free_all_blocks()

    return result.astype(np.uint16)

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

def clij_lr(image,psf,num_iters,tau):

    result = richardson_lucy_nc(image.astype(np.float32),psf.astype(np.float32),num_iters,tau)

    return result.astype(np.uint16)

def custom_lr(image,psf,num_iters):

    cache = cp.fft.config.get_plan_cache()

    image_cp = cp.asarray(image.astype(np.float32),dtype=cp.float32)
    psf_cp = cp.asarray(psf.astype(np.float32),dtype=cp.float32)

    nz, ny, nx = image_cp.shape

    crop_z = psf_cp.shape[0]//2
    crop_y = psf_cp.shape[1]//2
    crop_x = psf_cp.shape[2]//2

    def H(density):

        blurred_glow = fftconvolve(density, psf_cp, mode='same')
        blurred_glow[blurred_glow <= 1e-12] = 1e-12 # Avoid true zeros
        blurred_glow_crop = blurred_glow[crop_z:density.shape[0]-crop_z,
                                        crop_y:density.shape[1]-crop_y,
                                        crop_x:density.shape[2]-crop_x]

        del blurred_glow
        gc.collect()

        return blurred_glow_crop
            
    # Define H_t, the transpose of the forward measurement operator
    def H_t(ratio):
        correction_factor = cp.zeros((nz, ny, nx),dtype=cp.float32)
        padded_ratio = cp.pad(ratio,
                             ((crop_z, crop_z),
                              (crop_y, crop_y),
                              (crop_x, crop_x)),
                             mode='constant')
        correction_factor = fftconvolve(padded_ratio, psf_cp, mode='same')

        del padded_ratio
        gc.collect()

        return correction_factor

    H_t_norm = H_t(cp.ones_like(image_cp)) # Normalization factor

    estimate = H_t(cp.ones_like(image_cp)) # Naive initial belief

    for ii in range(num_iters):
        estimate *= H_t(image_cp / H(estimate)) / H_t_norm
        estimate = denoise_tv_chambolle(estimate,weight=1e-6,channel_axis=0)

    result = cp.asnumpy(estimate[crop_z:estimate.shape[0]-crop_z,
                                 crop_y:estimate.shape[1]-crop_y,
                                 crop_x:estimate.shape[2]-crop_x])

    del image_cp, psf_cp, H_t_norm, estimate
    gc.collect()

    cache.clear()

    cp.clear_memo()
    cp._default_memory_pool.free_all_blocks()

    return result.astype(np.uint16)