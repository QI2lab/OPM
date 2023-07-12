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
from skimage.transform import warp
from tqdm import tqdm
from tqdm.contrib.itertools import product as tqdm_product

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

def generate_flatfield(data: np.ndarray,) -> np.ndarray:
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
    
    basic = BaSiC(
        get_darkfield=True, 
        working_size=[128,425],
        smoothness_flatfield=2,
        smoothness_darkfield=10
    ) 
    basic.fit(data)
    flatfield = basic.flatfield
    darkfield = basic.darkfield

    return flatfield.astype(np.float32), darkfield.astype(np.float32)

def apply_flatfield(data: np.ndarray,
                    flatfield: np.ndarray,
                    darkfield: np.ndarray,
                    camera_offset: Optional[np.float32] = 100.0) -> np.ndarray:
    """
    Apply flatfield shading correction and remove camera offset

    Parameters
    ----------
    data : np.ndarray
        data to be corrected
    flatfield : np.ndarray
        flatfield image
    darkfield : np.ndarray
        darkfield image
    camera_offset : float
        scalar camera offset. Default is 110.0 for our FusionBT.

    Returns
    -------
    corrected_data : np.ndarray
        shading corrected data
    
    """
    
    corrected_data = (data.astype(np.float32) - darkfield.astype(np.float32)) / flatfield.astype(np.float32)
    corrected_data[corrected_data<0.0]=0.0

    return corrected_data.astype(np.uint16)

def run_bigstitcher(data_path: Path,
                    fiji_path: Path,
                    macro_path: Path,
                    bdv_xml_path: Path) -> None:
    """
    Run bigstitcher stitching using ImageJ headless

    Parameters
    ----------
    data_path : Path
        path to datasets
    fiji_path : Path
        path to Fiji on local system
    macro_pathh : Path
        path to Fiji stitching macro on local system
    fiji_path : Path
        path to BDV xml to use

    Returns
    -------
    None    
    """
    
    # construct bigstitcher command
    n5data_path = data_path / Path('fused') / Path('fused4x.n5')
    n5xml_path = data_path / Path('fused') / Path('fused4x.xml')
    command_to_run = str(fiji_path)+r' --headless -macro '+\
                     str(macro_path) + ' \"' + str(bdv_xml_path) + r' ' +\
                     str(n5data_path) + r' ' + str(n5xml_path) +'\"'

    # call bigstitcher and block until completion (long process)
    subprocess.run(command_to_run,capture_output=False)
    
    return None

def make_composed_transformation(path_affine: Path,
                                 tile_idx: int,
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
    
    xforms = data_io.return_affine_xform(path_to_xml=path_affine,
                                         tile_idx=tile_idx,
                                         r_idx=r_idx,
                                         x_idx=x_idx,
                                         y_idx=y_idx,
                                         z_idx=z_idx,
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
    
    # generate scaled transforms for 4x isotropic downsampled data
    scale_factor = 4
    scaled_xform_pix = np.copy(xform_pix)
    scaled_xform_pix[:-1,-1] /= scale_factor
    scaled_coordinate_scale_xform = np.copy(coordinate_scale_xform)
    scaled_coordinate_scale_xform *= scale_factor
    scaled_coordinate_scale_xform_inv = np.linalg.inv(scaled_coordinate_scale_xform)
    scaled_xform = x_invert_xform.dot(scaled_coordinate_scale_xform.dot(scaled_xform_pix.dot(scaled_coordinate_scale_xform_inv)))

    return xform, scaled_xform

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
    data_warped = warp(data_target, warper, mode='edge',preserve_range=True)

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

    Returns
    -------
    field: np.ndarray
        optical flow field
    """
    
    field = deeds.registration_fields(fixed=data_reference, 
                                      moving=data_target, 
                                      alpha=1.6, 
                                      levels=5, 
                                      verbose=False)
    to_return = np.array(field)
    del field
    gc.collect()

    return to_return

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
    tracking_tile_idx = -1
    for (x_idx,y_idx,z_idx) in tqdm_product(range(num_x),range(num_y),range(num_z),
                                            desc='Tiles'):
        tracking_tile_idx = tracking_tile_idx + 1

        # code to skip already processed tiles. Restarting due to page file memory issues.
        if tracking_tile_idx >= 132:
      
            # create composed transformation for r_idx = 0
            round_name = 'r'+str(0).zfill(3)
            tile_name = 'x'+str(x_idx).zfill(3)+'_y'+str(y_idx).zfill(3)+'_z'+str(z_idx).zfill(3)
            channel_id = 'ch488'
            current_channel = zarr.open_group(zarr_path,mode='a',path=round_name+'/'+tile_name+'/'+channel_id)
            try:
                tile_idx = int(da.from_zarr(current_channel['BDV_tile_idx']).compute())
            except:
                BDV_tile_idx = current_channel.zeros('BDV_tile_idx',
                                                    shape=(1),
                                                    compressor=compressor,
                                                    dtype=int)
                BDV_tile_idx[:] = tracking_tile_idx
                tile_idx = tracking_tile_idx
            xform_ref, scaled_xform_ref = make_composed_transformation(path_affine=BDV_XML_path,
                                                                        tile_idx=tile_idx,
                                                                        r_idx=0,
                                                                        x_idx=x_idx,
                                                                        y_idx=y_idx,
                                                                        z_idx=z_idx,
                                                                        pixel_size=pixel_size)

            # save composed transformations to zarr
            try:
                current_BDV_xform = da.from_zarr(current_channel['BDV_xform']).compute().astype(np.float32)
            except:
                current_BDV_xform = current_channel.zeros('BDV_xform',
                                                        shape=(4,4),
                                                        compressor=compressor,
                                                        dtype=float)
                current_BDV_xform[:] = xform_ref
            else:
                da.to_zarr(da.from_array(xform_ref),
                        current_channel['BDV_xform'],
                        overwrite=True,
                        compressor=compressor)

            try:
                current_scaled_xform = da.from_zarr(current_channel['scaled_xform']).compute().astype(np.float32)
            except:
                current_scaled_xform = current_channel.zeros('scaled_xform',
                                                        shape=(4,4),
                                                        compressor=compressor,
                                                        dtype=float)
                current_scaled_xform[:] = scaled_xform_ref
            else:
                da.to_zarr(da.from_array(scaled_xform_ref),
                        current_channel['scaled_xform'],
                        overwrite=True,
                        compressor=compressor)

            # load reference tile at r_idx = 0
            group_path = 'setup'+str(tile_idx)+'/timepoint0/s1' 
            scaled_data_reference = da.from_zarr(n5_data[group_path]).compute()
            # voxel_size = da.from_zarr(current_channel['voxel_size']).compute().astype(np.float32)
            # deskew_pixel_size = voxel_size[1]
            # scan_step = voxel_size[0]
            # theta = float(da.from_zarr(current_channel['theta']).compute())

            # data_reference_raw = da.from_zarr(current_channel['raw_data']).compute().astype(np.uint16)
            # data_reference_deskew = deskew(data_reference_raw,
            #                                deskew_pixel_size,
            #                                scan_step,
            #                                theta)
            # del data_reference_raw
            # gc.collect()
            # scaled_data_reference = downscale_local_mean(data_reference_deskew,factors=(4,4,4),cval=0).astype(np.float32)
            # del data_reference_deskew
            # gc.collect()

            # loop over all rounds in BDV N5 file
            for r_idx in tqdm(range(1,num_r),desc='Rounds',leave=False):
                # create composed transformation for this r_idx
                xform_target, scaled_xform_target = make_composed_transformation(path_affine=BDV_XML_path,
                                                                                tile_idx=tile_idx,
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
                try:
                    current_BDV_xform = da.from_zarr(current_channel['BDV_xform']).compute().astype(np.float32)
                except:
                    current_BDV_xform = current_channel.zeros('BDV_xform',
                                                            shape=(4,4),
                                                            compressor=compressor,
                                                            dtype=float)
                    current_BDV_xform[:] = xform_target
                else:
                    da.to_zarr(da.from_array(xform_target),
                            current_channel['BDV_xform'],
                            overwrite=True,
                            compressor=compressor)
                
                try:
                    current_scaled_xform = da.from_zarr(current_channel['scaled_xform']).compute().astype(np.float32)
                except:
                    current_scaled_xform = current_channel.zeros('scaled_xform',
                                                            shape=(4,4),
                                                            compressor=compressor,
                                                            dtype=float)
                    current_scaled_xform[:] = scaled_xform_target
                else:
                    da.to_zarr(da.from_array(scaled_xform_target),
                            current_channel['scaled_xform'],
                            overwrite=True,
                            compressor=compressor)

                # load 4x downsampled tile for current r_idx
                group_path = 'setup'+str(tile_idx)+'/timepoint'+str(r_idx)+'/s1'
                scaled_data_target = da.from_zarr(n5_data[group_path]).compute()
                # data_target_raw = da.from_zarr(current_channel['raw_data']).compute().astype(np.uint16)
                # data_target_deskew = deskew(data_target_raw,
                #                             deskew_pixel_size,
                #                             scan_step,
                #                             theta)
                # del data_target_raw
                # gc.collect()
                
                # scaled_data_target = downscale_local_mean(data_target_deskew,factors=(4,4,4),cval=0).astype(np.float32)
                # del data_target_deskew
                # gc.collect()

                # warp this r_idx tile to match r_idx = 0 tile
                scaled_data_warped = apply_relative_affine_transform(scaled_data_target,
                                                                    scaled_xform_ref,
                                                                    scaled_xform_target,
                                                                    pixel_size).astype(np.float32)
                del scaled_data_target
                gc.collect()
                
                # run optical flow on this tile
                optical_flow = compute_optical_flow(scaled_data_reference.astype(np.float32),
                                                    scaled_data_warped.astype(np.float32))

                # save optical flow transformation to zarr
                try:
                    current_OF_xform = da.from_zarr(current_channel['of_xform']).compute().astype(np.float32)
                except:
                    current_OF_xform = current_channel.zeros('of_xform',
                                                            shape=optical_flow.shape,
                                                            compressor=compressor,
                                                            dtype=float)
                    current_OF_xform[:] = optical_flow
                else:
                    da.to_zarr(da.from_array(optical_flow),
                            current_channel['of_xform'],
                            overwrite=True,
                            compressor=compressor)

                del current_channel, current_BDV_xform, current_OF_xform
                del scaled_data_warped, optical_flow
                gc.collect()
            
            del scaled_data_reference
            gc.collect()