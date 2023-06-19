'''
QI2lab OPM suite
Reconstruction tools v3

Image processing tools for OPM reconstruction

Last updated: Shepherd 06/23 - Refactor to match MERFISH image processing code, include new optical flow registration

To install cucim on windows:
pip install -e "git+https://github.com/rapidsai/cucim.git@v22.12.00#egg=cucim&subdirectory=python/cucim"

Check version number (...@v22.12.00...) for most recent RapidsAI API to install 
'''

#!/usr/bin/env python
import sys
import numpy as np
from pathlib import Path
from basicpy import BaSiC
from numba import njit, prange
from functools import partial
import dask.array as da
import dask.array.core
import zarr
from dask.diagnostics import ProgressBar
import gc
from typing import Union, Optional
import warnings
import subprocess
from itertools import product
import deeds
import _dataio as data_io
from localize_psf import affine
from skimage.transform import downscale_local_mean, warp

cupy_available = True
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = np
    cupy_available = False
else:
    from cupy.fft.config import get_plan_cache

# Deconvolution imports
opencl_avilable = True
try:
    from clij2fft.richardson_lucy import richardson_lucy_nc as clij2_decon # type: ignore
    from clij2fft.libs import getlib # type: ignore
    DECON_LIBRARY = 'clij2'
except ImportError:
    opencl_avilable = False
    DECON_LIBRARY = 'none'

@njit(parallel=True)
def deskew(data: np.ndarray,
           pixel_size: float,
           scan_step: float,
           theta: float) -> np.ndarray:
    """
    Perform parallelized orthogonal interpolation into a uniform pixel size grid.
    
    Parameters
    ----------
    data : np.ndarray
        Image stack of uniformly spaced oblique image planes
    pizel_size: float 
        Effective camera pixel size
    scan_step: float 
        Spacing between oblique planes
    theta : float 
        Oblique angle in degrees
    
    Returns
    -------
    deskewed_image : np.ndarray
        Image stack of deskewed oblique planes on uniform grid
    """

    # unwrap parameters 
    [num_images,ny,nx]=data.shape     # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = scan_step/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end+ny*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = np.int64(np.ceil(ny*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = np.int64(nx)                                           # (pixels)

    # create final image
    # (time, pixels,pixels,pixels - data is float32)
    deskewed_image = np.zeros((final_nz, final_ny, final_nx),dtype=np.float32)  

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
                
                # find distance of a point on the  interpolated plane to plane_before 
                # and plane_after
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
                if ((pos_before>=0) and (pos_after >= 0) and\
                    (pos_before<ny-1) and (pos_after<ny-1)):
                    
                    # determine points surrounding interpolated point on virtual plane 
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    deskewed_image[z,y,:] = (l_before * dz_after *\
                                             data[plane_after,pos_after+1,:] +\
                                             l_before * (1-dz_after) *\
                                             data[plane_after,pos_after,:] +\
                                             l_after * dz_before *\
                                             data[plane_before,pos_before+1,:] +\
                                             l_after * (1-dz_before) *\
                                             data[plane_before,pos_before,:])/pixel_step

    # return output
    return deskewed_image.astype(np.uint16)

def clij_lr(image: np.ndarray,
            psf: np.ndarray,
            iterations: int = 100,
            tv_tau: float = .001,
            lib = None) -> np.ndarray:
    """
    Lucy-Richardson non-circulant with total variation deconvolvution 
    using clij2-fft (OpenCL)

    Parameters
    ----------
    image : np.ndarray
        Image to be deconvolved
    psf : np.ndarray
        Point spread function
    iterations : int
        Number of iterations
    tau : float
        Total variation parameter
    lib: Optional
        pre-initialized libfft

    Returns
    -------
    result: np.ndarray
        Deconvolved image
    """

    result = clij2_decon(image.astype(np.float32),
                         psf.astype(np.float32),
                         iterations,
                         tv_tau,
                         lib=lib)

    return result.astype(np.uint16)

def deconvolve(image: Union[np.ndarray, dask.array.core.Array],
               psf: Union[np.ndarray, dask.array.core.Array],
               decon_params: dict,
               overlap_depth: list) -> np.ndarray:
    """
    Deconvolve current image in blocks using Dask and GPU. 
    Will not run if clij2-fft is not installed.

    Parameters
    ----------
    image : np.ndarray
        Image data
    psf : np.ndarray
        Point spread function
    decon_params : dict
        Deconvolution parameters
    overlap_depth : list
        Overlap padding

    Returns
    -------
    deconvolved_image : np.ndarray
        Deconolved image
    """

    decon_setup = False
    if DECON_LIBRARY=='clij2':
        iterations = decon_params['iterations']
        tv_tau = decon_params['tv_tau']
        lib = getlib()
        lr_dask_func = partial(clij_lr,psf=psf,iterations=iterations,tv_tau=tv_tau,lib=lib)
        decon_setup = True
        
    if decon_setup:
        dask_decon = da.map_overlap(lr_dask_func,
                                    image,
                                    depth=overlap_depth,
                                    boundary='reflect',
                                    trim=True,
                                    meta=np.array((), dtype=np.uint16))

        with ProgressBar():
            deconvolved_image = dask_decon.compute(scheduler='single-threaded')

        # clean up RAM
        del dask_decon, lr_dask_func, lib
        gc.collect()
    else:
        warnings.warn('GPU libraries not loaded, deconvolution not run.')

    return deconvolved_image.astype(np.uint16)

def generate_flatfield(data: np.ndarray) -> np.ndarray:
    """
    Optimize parameters and generate shading correction via BaSiCpy.

    Parameters
    ----------
    data : np.ndarray
        image data to be used. commonly random sub-sampled of OPM stage scan data
     
    Returns
    -------
    flatfield : np.ndarray
        flatfield correction
    """
    
    basic = BaSiC(get_darkfield=False,
                  smoothness_flatfield=1)
    basic.autotune(data, early_stop=True, n_iter=100)
    basic.fit(data)
    flatfield = basic.flatfield

    return flatfield.astype(np.float32)

@njit(nopython=True)
def apply_flatfield(data: np.nadrray,
                    flatfield: np.ndarray,
                    camera_offset: Optional[np.ndarray] = 110.0) -> np.ndarray:
    """
    Apply flatfield shading correction and remove camera offset

    Parameters
    ----------
    data : np.ndarray
        data to be corrected
    flatfield : np.ndarray
        flatfield image
    camera_offset : float
        scalar camera offset. Default is 110.0 for our FusionBT.

    Returns
    -------
    corrected_data : np.ndarray
        shading corrected data
    
    """
    
    corrected_data = (data.astype(np.float32) - camera_offset.astype(np.float32)) / flatfield
    corrected_data[corrected_data<0] = 0

    return corrected_data.astype(np.uint16)

def run_bigstitcher(data_path: Path,
                    fiji_path: Path,
                    macro_path: Path) -> None:
    """
    Run bigstitcher stitching using ImageJ headless

    Parameters
    ----------
    data_path : Path
        path to datasets
    fiji_path : Path
        path to Fiji

    Returns
    -------
    None    
    """
    
    # construct bigstitcher command
    xml_path = data_path / Path('processed') / Path('deskewed_bdv') / Path('dataset.xml')
    n5data_path = data_path / Path('processed') / Path('fused') / Path('fused4x.n5')
    n5xml_path = data_path / Path('processed') / Path('fused') / Path('fused4x.xml')
    command_to_run = str(fiji_path)+r' --headless -macro'+\
                     str(macro_path) + ' \"' + str(xml_path) + r' ' +\
                     str(n5data_path) + r' ' + str(n5xml_path) +'\"'

    # call bigstitcher and block until completion (long process)
    subprocess.run(command_to_run,capture_output=False)
    

    return None

def make_composed_transformation(path_affine: Path,
                                 r_idx: int,
                                 x_idx: int,
                                 y_idx: int,
                                 z_idx: int,
                                 pixel_size: float):
    """
    Compute a composed affine transformation for a given tile position.

    Parameters
    ----------
    path_affine: Path
        path to BDV XML file
    r_idx: int
        imaging round. saved as "time" in BDV XML.
    x_idx: int
        x tile index
    y_idx: int
        y tile index
    z_idx: int
        z tile index
    pixel_size: float
        pixel size of isotropic OPM grid

    Returns
    -------
    xform: np.ndarray
        composed transformation for all BigStitcher transformations
    """
    
    xforms = data_io.return_affine_xform(path_affine,
                                         r_idx=r_idx,
                                         x_idx=x_idx,
                                         y_idx=y_idx,
                                         z_idx=z_idx,
                                         ch_affine=0, 
                                         tile_offset=0,
                                         verbose=0)

    # convert to homogeneous coordinate matrix
    xforms = [np.concatenate((a, np.array([[0, 0, 0, 1]])), axis=0) for a in xforms]
    # get xform which acts on deskewed pixel coordinates
    xform_pix = np.copy(xforms[-1])
    for mat in xforms[-2::-1]:
        # apply dot product starting from the end
        xform_pix = mat.dot(xform_pix)
    # put this in um's instead of dcs
    coordinate_scale_xform = np.array([[pixel_size, 0, 0, 0], [0, pixel_size, 0, 0], [0, 0, pixel_size, 0], [0, 0, 0, 1]])
    coordinate_scale_xform_inv = np.linalg.inv(coordinate_scale_xform)
    # invert x-axis again, so not reflected compared with initial data
    x_invert_xform = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    xform = x_invert_xform.dot(coordinate_scale_xform.dot(xform_pix.dot(coordinate_scale_xform_inv)))
    
    return xform

def make_warper_from_field(field):
    """
    Make a warper array from the results of a optical flow method for
    scikit-image's `warp` function.
    
    Parameters
    ----------
    field: ndarray or list(ndarray)
        Result from scikit-image or cucim ILK or TLV1 methods, or from DEEDS.
    
    Returns
    -------
    warper: ndarray
        A (3 x N0 x N1 x N2) array indicating origin and final position of voxels.
    """
    
    nz, ny, nx = field[0].shape
    z_coords, y_coords, x_coords = np.meshgrid(np.arange(nz), 
                                               np.arange(ny), 
                                               np.arange(nx),
                                               indexing='ij')
        
    warper = np.array([z_coords + field[0], 
                       y_coords + field[1],
                       x_coords + field[2]])
        
    return warper

def make_warper_from_affine(shape, xform):
    """
    Make a warper array from an affine transformation matrix for 
    scikit-image's `warp` function.
    
    Parameters
    ----------
    shape: ndarray or list
        Shape of the image the wraper is used for.
    xform: array
        Affine transformation. 
    
    Returns
    -------
    warper: ndarray
        A (3 x N0 x N1 x N2) array indicating origin and final position of voxels.
    """
    
    nz, ny, nx = shape
    z_coords, y_coords, x_coords = np.meshgrid(np.arange(nz), 
                                               np.arange(ny), 
                                               np.arange(nx),
                                               indexing='ij')
                                            
    coords = np.array([z_coords, y_coords, x_coords])
    coords = coords.T.reshape((-1, 3))
    coords_warped = affine.xform_points(coords, xform)
    coords_warped = coords_warped.reshape((nx, ny, nz, 3)).T
    
    return coords_warped


def apply_relative_affine_transform(data_target: np.ndarray,
                                    transform_reference: np.ndarray,
                                    transform_target: np.ndarray,
                                    pixel_size: float) -> np.ndarray:
    """
    Warp a target image to a reference image given their affine transformations.
    
    Parameters
    ----------
    data_target : np.ndarray
        Target image.
    transform_reference : np.ndarray
        Affine transformation to transform the reference data into real world space,
        in the xyz convention.
    transform_target : np.ndarray
        Affine transformation to transform the target data into real world space,
        in the xyz convention.
    pixel_size: float
        Pixel size on isotropic grid

    Returns
    -------
    data_warped : np. ndarray
        Warped target image
    """
    
    transfo_ref_inv = np.linalg.inv(transform_reference)
    transfo_ref_inv_trg = transfo_ref_inv.dot(transform_target)
    # swap zyx / xyz conventions
    transfo_ref_inv_trg[:3, -1][:3] = transfo_ref_inv_trg[:, -1][:3][::-1]
    
    # Make warper from affine transformation, because `warp` doesn't work directly with affine matrices for 3D data
    transfo_ref_inv_trg_pix = np.copy(transfo_ref_inv_trg)
    # need to rescale to pixel size
    transfo_ref_inv_trg_pix[:-1, -1] /= pixel_size
    # need to use the inverse of the transformation
    transfo_ref_inv_trg_pix = np.linalg.inv(transfo_ref_inv_trg_pix)
    # compute the warper
    warper = make_warper_from_affine(data_target.shape, transfo_ref_inv_trg_pix)
    # compute the warped image

    data_warped = warp(data_target, warper, mode='edge')

    return data_warped

def compute_optical_flow(data_reference: np.ndarray, 
                         data_target: np.ndarray) -> np.ndarray:
    """
    Compute the optical flow to warp a target image to a reference image.

    Parameters
    ----------
    data_reference : np.ndarray
        reference data
    data_target : np.ndarray
        target data
    scale_factors : List[float,float,float]
        zyx list of down-scampling factors

    Returns
    -------
    field: np.ndarray
        optical flow field calculated on downscaled data
    """
    
    field = deeds.registration_fields(fixed=data_reference, 
                                      moving=data_target, 
                                      alpha=1.6, 
                                      levels=5, 
                                      verbose=False)
    
    field = np.array(field)

    return field

def perform_local_registration(BDV_N5_path: Path,
                               BDV_XML_path: Path,
                               zarr_path: Path,
                               num_r: int,
                               num_x: int,
                               num_y: int,
                               num_z: int,
                               pixel_size: float,
                               compressor) -> None:
    """
    Perform local registration using 'deeds' optical flow on bigstitcher N5 file. 
    Store composed bigstitcher transforms and optical flow deformation in raw data zarr.

    Parameters
    ----------
    BDV_N5_path : Path
        path to BDV N5
    BDV_XML_path : Path
        path to BDV XML
    zarr_path : Path
        path containing raw Zarr data
    num_r : int
        number of rounds
    num_x : int
        number of x positions
    num_y : int
        number of y positions 
    num_z : int
        number of z positions 
    pixel_size : float
        isotropic deskewed pixel size

    Returns
    -------
    None    
    """

    n5_data  = zarr.open(BDV_N5_path)

    # loop over all tiles
    for (x_idx,y_idx,z_idx) in product(range(num_x),range(num_y),range(num_z)):
        # create composed transformation for r_idx = 0
        transform_reference = make_composed_transformation(path_affine=BDV_XML_path,
                                                           r_idx=0,
                                                           x_idx=x_idx,
                                                           y_idx=y_idx,
                                                           z_idx=z_idx,
                                                           pixel_size=pixel_size)

        # save composed transformation to zarr
        tile_name = 'x'+str(x_idx).zfill(3)+'_y'+str(y_idx).zfill(3)+'_z'+str(z_idx).zfill(3)
        channel_id = 'ch488'
        current_channel = zarr.open_group(zarr_path,mode='a',
                                          path='r000'+'/'+tile_name+'/'+channel_id)
        current_BDV_xform = current_channel.zeros('bdv_xform',
                                                  shape=(4,4),
                                                  compressor=compressor,
                                                  dtype=float)
        current_BDV_xform[:] = transform_reference
        current_BDV_tile_idx = current_channel['current_BDV_tile_idx']

        # load reference tile at r_idx = 0
        group_path = 'setup'+str(current_BDV_tile_idx)+'/timepoint0/s0'
        data_reference = da.from_zarr(n5_data[group_path])

        # downsample reference tile at r_idx =0
        data_reference_downsampled = downscale_local_mean(data=data_reference.compute(),
                                                          factors=(4,4,4),
                                                          cval=0)
        del data_reference, current_channel
        gc.collect()

        # loop over all rounds in BDV N5 file
        for r_idx in range(num_r):
            # create composed transformation for this r_idx
            transform_target = make_composed_transformation(path_affine=BDV_XML_path,
                                                            r_idx=r_idx,
                                                            x_idx=x_idx,
                                                            y_idx=y_idx,
                                                            z_idx=z_idx,
                                                            pixel_size=pixel_size)

            # save composed transformation to zarr
            round_name = 'r'+str(r_idx).zfill(3)
            tile_name = 'x'+str(x_idx).zfill(3)+'_y'+str(y_idx).zfill(3)+'_z'+str(z_idx).zfill(3)
            channel_id = 'ch488'
            current_channel = zarr.open_group(zarr_path,
                                              mode='a',
                                              path=round_name+'/'+tile_name+'/'+channel_id)
            current_BDV_xform = current_channel.zeros('bdv_xform',
                                                      shape=(4,4),
                                                      compressor=compressor,
                                                      dtype=float)
            current_BDV_xform[:] = transform_target

            # load data
            group_path = 'setup'+str(current_BDV_tile_idx)+'/timepoint'+str(r_idx)+'/s0'
            data_target = da.from_zarr(n5_data[group_path])

            # warp this tile to match r_idx = 0 tile
            data_warped = apply_relative_affine_transform(data_target,
                                                          transform_reference,
                                                          transform_target,
                                                          pixel_size)
            del data_target
            
            # downsample tile for this r_idx
            data_warped_downsampled = downscale_local_mean(data=data_warped,
                                                           factors=(4,4,4),
                                                           cval=0)
            del data_warped
            gc.collect()

            # run optical flow on this tile
            optical_flow = compute_optical_flow(data_reference_downsampled,
                                                data_warped_downsampled)

            # save optical flow transformation to zarr
            current_OF_xform = current_channel.zeros('of_xform',
                                                     shape=optical_flow.shape,
                                                     compressor=compressor,
                                                     dtype=float)
            current_OF_xform[:] = optical_flow

            del current_channel, current_BDV_xform, current_OF_xform
            del data_warped_downsampled
            gc.collect()