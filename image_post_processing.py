#!/usr/bin/env python
import sys
import numpy as np
from numba import njit, prange

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

def manage_flat_field(stack,ij):
    """
    Manage performing flat and dark-field using BaSiC algorithm via pyimagej. 
    The use of pyimagej can likely be improved, but this works for now.
    https://doi.org/10.1038/ncomms14836

    Returns flat- and darkfield corrected image
    
    :param stack: ndarray
        matrix of OPM planes
    :param ij: imagej
        imagej object

    :return corrected_stack: ndarray of deskewed OPM planes on uniform grid
    """

    print('Calculating flat-field correction using ImageJ and BaSiC plugin.')
    flat_field, dark_field = calculate_flat_field(stack,ij)

    print('Performing flat-field correction.')
    corrected_stack = perform_flat_field(flat_field,dark_field,stack)

    return corrected_stack

def calculate_flat_field(stack,ij):
    """
    Calculate flat- and darkfield using BaSiC algorithm via pyimagej. Returns flat- and darkfield images.
    
    :param stack: ndarray
        matrix of OPM planes
    :param ij: imagej
        imagej object

    :return flat_field: ndarray
        retrospective flat field 
    :return dark_field: ndarray
        retrospective dark field 
    """

    # convert dataset from numpy -> java
    if stack.shape[0] >= 200:
        stack_for_flat_field = stack[np.random.choice(stack.shape[0], 50, replace=False)]
    else:
        stack_for_flat_field = stack
    stack_iterable = ij.op().transform().flatIterableView(ij.py.to_java(stack_for_flat_field))

    # show image in imagej since BaSiC plugin cannot be run headless
    ij.ui().show(stack_iterable)
    WindowManager = jimport('ij.WindowManager')
    current_image = WindowManager.getCurrentImage()

    # convert virtual stack to real stack and reorder for BaSiC
    macro = """
    rename("active")
    run("Duplicate...", "duplicate")
    selectWindow("active")
    run("Close")
    selectWindow("active-1")
    run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
    """
    ij.py.run_macro(macro)

    # run BaSiC plugin
    plugin = 'BaSiC '
    args = {
        'processing_stack': 'active-1',
        'flat-field': 'None',
        'dark-field': 'None',
        'shading_estimation': '[Estimate shading profiles]',
        'shading_model': '[Estimate both flat-field and dark-field]',
        'setting_regularisationparametes': 'Automatic',
        'temporal_drift': '[Ignore]',
        'correction_options': '[Compute shading only]',
        'lambda_flat': 0.5,
        'lambda_dark': 0.5
    }
    ij.py.run_plugin(plugin, args)

    # grab flat-field image, convert from java->numpy
    macro2 = """
    selectWindow("active-1")
    run("Close")
    selectWindow("Flat-field:active-1")
    """
    ij.py.run_macro(macro2)
    current_image = WindowManager.getCurrentImage()
    flat_field_ij = ij.py.from_java(current_image)
    flat_field = flat_field_ij.data

    # close flat-field, grab dark-field image, convert from java->numpy
    macro3 = """
    selectWindow("Flat-field:active-1")
    run("Close")
    selectWindow("Dark-field:active-1")
    """
    ij.py.run_macro(macro3)

    current_image = WindowManager.getCurrentImage()
    dark_field_ij = ij.py.from_java(current_image)
    dark_field = dark_field_ij.data

    # close dark-field image
    macro4 = """
    selectWindow("Dark-field:active-1")
    run("Close")
    run("Collect Garbage")
    """
    ij.py.run_macro(macro4)

    return flat_field, dark_field

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

    corrected_stack = stack.astype(np.float32) - dark_field
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
    params.blind=True
    params.NA = 1.35
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