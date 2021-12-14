'''
Paired down image post processing for napari-OPM control. Contains only orthogonal deskew.
D. Shepherd - 12/2021
'''

#!/usr/bin/env python
import sys
import numpy as np
from numba import njit, prange
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from functools import partial
import dask.array as da

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

def _c(array):
    """
    :param array: dexp
        array on dexp backend
    :return array: ndarray
        array converted to numpy
    """
    return Backend.to_numpy(array)

def tiled_lr_decon(image,skewed_psf,num_iterations,padding,internal_dtype):
    """
    Lucy-Richardson deconvolution using dexp library
    :param image: ndarray
        raw OPM data tile generated by dask
    :param skewed_psf: ndarray
        theoretical PSF skewed into OPM coordinates
    :param padding: int
        internal padding for dexp
    :param internal_dtype: dtype
        data type to use
    :return result: ndarray
        deconvolved tile
    """

    result = lucy_richardson_deconvolution(
        image=image,
        psf=skewed_psf,
        num_iterations=num_iterations,
        padding=padding,
        internal_dtype=internal_dtype
    )

    return _c(result)

def lr_deconvolution_cupy(image,psf,iterations=50):
    """
    Tiled Lucy-Richardson deconvolution using dask and dexp library
    :param image: ndarray
        raw OPM data
    :param skewed_psf: ndarray
        theoretical PSF skewed into OPM coordinates
    :param iterations: int
        number of iterations to run 
    :return deconvolved: ndarray
        deconvolved image
    """

    nz,ny,nx = image.shape
    lr_dask = partial(tiled_lr_decon,psf=psf,num_iterations=iterations,padding=16,internal_dtype=np.float16)
    dask_raw = da.from_array(image.astype(np.float16),chunks=(nz,2048,nx))
    dask_decon = da.map_overlap(lr_dask,dask_raw,depth=100,boundary=None,trim=True,dtype=np.float16)

    with CupyBackend():
        decon = dask_decon.compute(scheduler='single-threaded')
        
    return decon.astype(np.uint16)


def manage_flat_field_py(stack):
    """
    Manage performing flat and dark-field using python adapation of BaSiC algorithm.

    Returns flat- and darkfield corrected image
    
    :param stack: ndarray
        matrix of OPM planes

    :return corrected_stack: ndarray of deskewed OPM planes on uniform grid
    """

    print('Calculating flat-field correction using python BaSiC adaption.')
    if stack.shape[0] > 1000:
        stack_for_flat_field = stack[np.random.choice(stack.shape[0], 1000, replace=False)]
    else:
        stack_for_flat_field = stack

    flat_field, dark_field = calc_flatfield(images=stack_for_flat_field)

    print('Performing flat-field correction.')
    corrected_stack = perform_flat_field(flat_field,dark_field,stack)

    return corrected_stack

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

    #corrected_stack = stack.astype(np.float32) - dark_field
    corrected_stack = stack.astype(np.float32)
    corrected_stack[corrected_stack<0] = 0 
    corrected_stack = corrected_stack/flat_field

    return corrected_stack

def mv_decon(image,ch_idx,dr,dz):
    '''
    Perform deconvolution using Microvolution API. To do: implement reading known PSF from disk. Return deconvolved image.

    :param image: ndarray
        raw image
    :param ch_idx: int
        wavelength index
    :param dr: float
        xy pixel size
    :param dz: float
        z pixel size

    :return image: ndarray
        deconvolved image 
    '''

    import microvolution_py as mv

    wavelengths = [460.,520.,605.,670.,780.]
    wavelength=wavelengths[ch_idx]

    params = mv.LightSheetParameters()
    params.nx = image.shape[2]
    params.ny = image.shape[1]
    params.nz = image.shape[0]
    params.generatePsf = True
    params.lightSheetNA = 0.16
    params.blind=False
    params.NA = 1.2
    params.RI = 1.4
    params.ns = 1.4
    params.psfModel = mv.PSFModel_Vectorial
    params.psfType = mv.PSFType_LightSheet
    params.wavelength = wavelength
    params.dr = dr
    params.dz = dz
    params.iterations = 20
    params.background = 0
    params.regularizationType=mv.RegularizationType_TV
    params.scaling = mv.Scaling_U16

    try:
        launcher = mv.DeconvolutionLauncher()
        image = image.astype(np.float32)

        launcher.SetParameters(params)
        for z in range(params.nz):
            launcher.SetImageSlice(z, image[z,:])

        launcher.Run()

        for z in range(params.nz):
            launcher.RetrieveImageSlice(z, image[z,:])

    except:
        err = sys.exc_info()
        print("Unexpected error:", err[0])
        print(err[1])
        print(err[2])

    return image