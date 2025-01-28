"""Sensorless adaptive optics.

TO DO: 
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""
from src.hardware.AOMirror import AOMirror


import numpy as np
import json
from scipy.fftpack import dct
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from numpy.typing import ArrayLike
from typing import Optional
from pathlib import Path
from pymmcore_plus import CMMCorePlus
# import matplotlib.pyplot as plt
from tifffile import imwrite
from datetime import datetime
import time

def get_image_center(image: ArrayLike,
                     threshold: float):
    """Calculate the image center using a thresholded binary mask
    """
    try:
        binary_image = image > threshold
        center = center_of_mass(binary_image)
        center = tuple(map(int, center))
    except Exception:
        center = (image.shape[1]//2, image.shape[0]//2)
    return center

def get_cropped_image(image, crop_size: int, center):
    """
    """    
    # x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[0])
    # y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[1])
    if len(image.shape)==3:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[1])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[2])
        cropped_image = image[:, x_min:x_max, y_min:y_max]
    else:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[0])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[1])
        cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image

def snap_image(mmc, obis_on: bool = False):
    """Snap an image using MM core
    """
    if obis_on:
        mmc.setConfig("Channels","488nm")
    mmc.snapImage()
    image = mmc.getImage()
    if obis_on:
        mmc.setConfig("Channels","OFF")
    return image

def gauss2d(
    coords_xy: ArrayLike, 
    amplitude: float, 
    center_x: float, 
    center_y: float, 
    sigma_x: float, 
    sigma_y: float, 
    offset: float
):
    """Generate 2D guassian given parameters.

    Parameters
    ----------
    coords_xy: ArrayLike
        2D coordinate grid
    amplitude: float
        2D gaussian amplitude
    center_x: float
        2D gaussian x coordinate center
    center_y: float
        2D gaussian y coordinate center
    sigma_x: float
        2D gaussian x coordinate sigma
    sigma_y: float
        2D gaussian y coordinate sigma
    offset: float
        2D gaussian offset

    Returns
    -------
    raveled_gauss2d: ArrayLike
        raveled 2D gaussian for fitting
    """
    x, y = coords_xy
    raveled_gauss2d = (
        offset +
        amplitude * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) + ((y - center_y)**2 / (2 * sigma_y**2)))
        )
    ).ravel()

    return raveled_gauss2d

def otf_radius(img: ArrayLike, psf_radius_px: float):
    """Maximum number of spatial frequencies in the image.
    
    Parameters
    ----------
    img : ArrayLike
        2D image.
    psf_radius_px : float
        theoretical psf radius in pixels.

    Returns
    ----------
    cutoff: int
        OTF cutoff in pixels.
    """

    w = min(img.shape)
    psf_radius_px = np.ceil(psf_radius_px)  # clip all PSF radii below 1 px to 1.
    cutoff = np.ceil(w / (2 * psf_radius_px)).astype(int)

    return cutoff

def normL2(x: ArrayLike):
    """L2 norm of n-dimensional array.
    
    Parameters
    ----------
    x: ArrayLike
        nD array.

    Returns
    -------
    l2norm: float
        L2 norm of array.
    """

    l2norm = np.sqrt(np.sum(x.flatten() ** 2))

    return l2norm

def abslog2(x: ArrayLike):
    """Absolute value, log2 of array
    
    Parameters
    ----------
    x: ArrayLike
        nD array.

    Returns
    -------
    result : ArrayLike
        log2(abs(x)).
    """

    x_abs = abs(x)
    result = np.zeros(x_abs.shape)
    result[x_abs > 0] = np.log2(x_abs[x_abs > 0].astype(np.float32))

    return result

def shannon(spectrum_2d: ArrayLike, otf_radius: int =100):
    """Normalized shannon entropy of an image spectrum, bound by OTF support radius.
    
    Parameters
    ----------
    spectrum_2d : ArrayLike
        spectrum (DCT, FFT, etc..) of image.
    otf_radius: int
        OTF radius in pixels.

    Returns
    -------
    entropy: float
        entropy of spectrum_2d.
    """

    h, w = spectrum_2d.shape
    y, x = np.ogrid[:h, :w]
    support = (x + y < otf_radius)
    norm = normL2(spectrum_2d[support])
    if norm != 0:
        terms = spectrum_2d[support].flatten() / norm
        entropy = -2 / otf_radius**2 * np.sum(abs(terms) * abslog2(terms))
    else:
        entropy = 0
    
    return entropy

def dct_2d(
    image: ArrayLike, 
    cutoff: int = 100):
    """DCT 2D of image, subject to cutoff.
    
    Parameters
    ----------
    image: ArrayLike
        2D image.
    cutoff: int
        OTF radius in pixels.

    Returns
    -------
    dct_2d: ArrayLike
        2D discrete cosine transform
    """
    
    dct_2d = dct(dct(image.astype(np.float32).T, norm='ortho', n=cutoff).T, norm='ortho', n=cutoff)

    return dct_2d

def quadratic(x,a,b,c):
    """Quadratic function evaluation at x.
    
    Parameters
    ----------
    x: float
        point to evaluate
    a: float
        x**2 coeff.
    b: float
        x coeff
    c: float
        offset
        
    Returns
    -------
    value: float
        a*x**2 + b*x + c
    """
    
    return a*x**2 + b*x + c

def quadratic_fit(x, y):
    """Quadratic function for curve fitting.
    
    Parameters
    ----------
    x: ArrayLike
        1D x-axis data.
    y: ArrayLike
        1D y-axis data.
        
    Returns
    -------
    coeffs: Sequence[float]
        fitting parameters
    """
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        
    return coeffs

def metric_brightness(
    image: ArrayLike,
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool]= False
    ):
    """Compute weighted metric for 2D gaussian.

    Parameters
    ----------
    image: ArrayLike
        2D image.
    threshold: float
        initial threshold to find spot (default 100)
    crop_size_px: int
        crop size in pixels, one side (default 20)

    Returns
    -------
    weighted_metric: float
        weighted metric
    """
    # Optionally crop the image
    if crop_size:    
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        # crop image
        image = get_cropped_image(image, crop_size, center)
    if len(image.shape)==3:
        # max project over first axis assuming it is a zstack.
        image = np.max(image, axis=0)

    if return_image:
        return np.max(image,axis=(0,1))+1e-12, image
    else:
        return np.max(image,axis=(0,1))+1e-12

def metric_gauss2d(
    image: ArrayLike,
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool]= False
    ):
    """Compute weighted metric for 2D gaussian.

    Parameters
    ----------
    image: ArrayLike
        2D image.
    threshold: float
        initial threshold to find spot (default 100)
    crop_size_px: int
        crop size in pixels, one side (default 20)

    Returns
    -------
    weighted_metric: float
        weighted metric
    """
    # Optionally crop the image
    if crop_size:    
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        # crop image
        image = get_cropped_image(image, crop_size, center)
        
    # normalize image 0-1
    image = image / np.max(image)
    image = image.astype(np.float32)
    
    # create coord. grid for fitting 
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    
    # TODO: Use localize_psf to locate spots and generate average 2-d gauss
    
    # fitting assumes a single bead in FOV....
    initial_guess = (image.max(), image.shape[1] // 2, 
                     image.shape[0] // 2, 5, 5, image.min())
    fit_bounds = [[0,0,0,1.0,1.0,0],
                  [1.5,image.shape[1],image.shape[0],100,100,5000]]
    try:
        popt, pcov = curve_fit(gauss2d, (x, y), image.ravel(), 
                               p0=initial_guess,
                               bounds=fit_bounds,
                               maxfev=1000)
        
        amplitude, center_x, center_y, sigma_x, sigma_y, offset = popt
        weighted_metric = ((1 - np.abs((sigma_x-sigma_y) / (sigma_x+sigma_y))) 
                           + (1 / (sigma_x+sigma_y)) 
                           + np.exp(-1 * (sigma_x+sigma_y-4)**2))
        
        if (weighted_metric < 0) or (weighted_metric > 100):
            weighted_metric = 1e-12 
    except Exception:
        weighted_metric = 1e-12
        
        
    if return_image:
        return weighted_metric, image
    else:
        return weighted_metric

"""
Need to work on this:
1. is there a conflict installing any of the localize_psf packages?
"""
# localize_psf imports
# from localize_psf.localize import (localize_beads_generic, 
#                                    get_param_filter,
#                                    get_coords, 
#                                    plot_fit_roi)
# from localize_psf.fit import fit_model
# import localize_psf.fit_psf as psf

# def metric_gauss3d(
#     image: ArrayLike,
#     metric_value: str = "mean",
#     crop_size: Optional[int] = None,
#     threshold: Optional[float] = 100,
#     image_center: Optional[int] = None,
#     verbose: Optional[bool] = False,
#     return_image: Optional[bool] = False
#     ):
#     """Compute wieghted metric for 3D Gaussian using LocalizePSF
    
#     Parameters
#     ----------
#     metric_value: str
#         "mean"-use the mean fit values, "average"-generate the average PSF and use that fit.
#     """
#     # Optionally crop the image
#     if crop_size:    
#         if image_center is None:
#             center = get_image_center(image, threshold)
#         else:
#             center = image_center
#         # crop image
#         image = get_cropped_image(image, crop_size, center)
        
#     image = image / np.max(image)
#     image = image.astype(np.float32)
        
#     # Define coordinates to pass to localization, use pixel units
#     # Using pixel units, but assumes we are using 0.270 z-steps
#     dxy = 1 # 0.115
#     dz  = 0.250 / 0.115 # 0.250 
#     coords_3d = get_coords(image.shape, (dz, dxy, dxy))
#     # coords_2d = get_coords(cropped_image.shape[1:], (dxy, dxy))
    
#     # Prepare filter for localization
#     sigma_bounds = ((0.1, 0.1),(100, 100)) # [xy min, xy max, z min, z max]
#     amp_bounds = (0.1, 2.0) # [min / max]
#     param_filter = get_param_filter(coords_3d,
#                                     fit_dist_max_err=(5, 5),
#                                     min_spot_sep=(10, 6),
#                                     amp_bounds=amp_bounds,
#                                     dist_boundary_min=[3, 3],
#                                     sigma_bounds=sigma_bounds
#                                 )
#     filter = param_filter  
     
#     # define roi sizes used in fitting, assumes a minimum 3um z-stack, dz=0.27um
#     fit_roi_size = [9, 7, 7]
    
#     # Run localization function
#     model = psf.gaussian3d_psf_model()
#     _, r, img_filtered = localize_beads_generic(
#         image,
#         drs=(dz, dxy, dxy),
#         threshold=0.5,
#         roi_size=fit_roi_size,
#         filter_sigma_small=None,
#         filter_sigma_large=None,
#         min_spot_sep=(10,10),
#         model=model,
#         filter=filter,
#         max_nfit_iterations=100,
#         use_gpu_fit=False,
#         use_gpu_filter=False,
#         return_filtered_images=True,
#         fit_filtered_images=False,
#         verbose=True
#         )
    
#     if r is None:
#         print("no beads found!")
#         return 0, image[image.shape[0]//2]
#     else:
#         to_keep = r["to_keep"]
#         fit_params = r["fit_params"]
#         sz = fit_params[to_keep, 5]
#         sxy = fit_params[to_keep, 4]
#         amp = fit_params[to_keep, 0]
#         # Use averages over fit results
#         if metric_value=="mean":
#             sz = np.mean(fit_params[to_keep, 5])
#             sxy = np.mean(fit_params[to_keep, 4])
#             amp = np.mean(fit_params[to_keep, 0])
#         elif metric_value=="median":
#             sz = np.median(fit_params[to_keep, 5])
#             sxy = np.median(fit_params[to_keep, 4])
#             amp = np.median(fit_params[to_keep, 0])
#         elif metric_value=="average":
#             # Generate average PSF
#             fit_roi_size_pix = np.round(np.array(fit_roi_size) / np.array([dz, dxy, dxy])).astype(int)
#             fit_roi_size_pix += (1 - np.mod(fit_roi_size_pix, 2))

#             psfs_real = np.zeros((1) + tuple(fit_roi_size_pix))
#             # otfs_real = np.zeros(psfs_real.shape, dtype=complex)
#             fit_params_average = np.zeros((1, model.nparams))
#             psf_coords = None

#             # only use a percent of bead results based on the sxy
#             percentile = 50
#             if percentile:
#                 sigma_max = np.percentile(fit_params[:, 4][to_keep], percentile)
#                 to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

#                 # get centers
#                 centers = np.stack((fit_params[:, 3][to_use],
#                                     fit_params[:, 2][to_use],
#                                     fit_params[:, 1][to_use]), axis=1)

#             # find average experimental psf/otf
#             psfs_real, psf_coords = psf.average_exp_psfs(r["data"],
#                                                          coords_3d,
#                                                          centers,
#                                                          fit_roi_size_pix,
#                                                          backgrounds=fit_params[:, 5][to_use],
#                                                          return_psf_coords=True)

#             # fit average experimental psf
#             def fn(p): return model.model(psf_coords, p)
#             init_params = model.estimate_parameters(psfs_real, psf_coords)

#             results = fit_model(psfs_real, fn, init_params, jac='3-point', x_scale='jac')
#             fit_params_average = results["fit_params"]     

#             sz = fit_params_average[5]
#             sxy = fit_params_average[4]
#             amp = fit_params_average[0]    
        
#         # TODO: Refine weighted metric if needed
#         weight_amp = 1 # scales amplitude to a value between 0-65
#         # SJS: Normalize image and remove brightness
#         weight_xy = 2 
#         weight_z = 2
#         weighted_metric = weight_amp * amp + weight_xy / sxy + weight_z / sz + np.exp(-1*(sxy+sz-6)**2)

#         if return_image:
#             return weighted_metric, image
#         else:
#             return weighted_metric

def metric_shannon_dct(
    image: ArrayLike, 
    psf_radius_px: float = 3,
    crop_size: Optional[int] = 501,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool]= False):
    """Shannon entropy of discrete cosine transform, for 2D images.
    
    Parameters
    ----------
    img: ArrayLike
        2D image.
    psf_radius_px: float
        radius of PSF in pixels
    
    Returns
    -------
    shannon_dct: float
        shannon entropy of 2D DCT of image.
    """
    # Optionally crop the image
    if crop_size:    
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        # crop image
        image = get_cropped_image(image, crop_size, center)
    
        
    cutoff = otf_radius(image, psf_radius_px)
    shannon_dct = shannon(dct_2d(image, cutoff), cutoff)

    if return_image:
        return shannon_dct, image
    else:
        return shannon_dct

def load_initial_state(file_path: Path):
    """Load initial wavefront corrector zernike mode amplitudes from a JSON file.
    
    Parameters
    ----------
    file_path: Path
        path to json storing mirror zernike mode amplitudes amplitudes
        
        
    Returns
    -------
    mirror_states: Sequence[float]
        mirror actuator amplitudes
    """

    with open(file_path, 'r') as file:
        data = json.load(file)

    mirror_states = np.asarray(data.get("initial_coeffs", []),dtype=np.float32)
    
    return mirror_states

def save_final_state(file_path, mirror_states):
    """Save the final optimized zernike mode amplitudes to a JSON file.
    
    Parameters
    ----------
    file_path: Path
        path to json for storing zernike mode amplitudes
    mirror_states: ArrayLike
        mirror zernike mode amplitudes
    """

    with open(file_path, 'w') as file:
        json.dump({"optimized_coeffs": mirror_states.tolist()}, file, indent=4)

def measure_metric(
        mmc: CMMCorePlus, 
        crop_size: int,
        threshold: float,
        psf_radius_px: Optional[float] = 3,
        obis_on: bool = False, 
        metric_type: str = "brightness2d",
        image_center: Optional[int] = None,
        verbose: bool = False):
    """Snap image and compute quality metric.

    Parameters
    ----------
    mmc: CMMCorePlus
        pymmcore-plus core
    metric_type: str
        image metric. Either "gauss2d" or "dct2d".
    psf_radius_px: float
        psf radius in pixels.

    Returns
    -------
    metric: float
        image quality metric
    cropped_image: ArrayLike
        cropped image for display or saving
    """
    if metric_type=="gauss3d":
        # Take z-stack with 03 and return 3d image
        # grab position and name of current MM focus stage
        # exp_zstage_pos = np.round(mmc.getPosition(),2)
        # exp_zstage_name = mmc.getFocusDevice()
        # if verbose: 
        #     print(f'Current z-stage: {exp_zstage_name} with position {exp_zstage_pos}')

        # set MM focus stage to O3 piezo stage
        O3_stage_name = "MCL NanoDrive Z Stage"
        z_start = 48.7
        mmc.setFocusDevice(O3_stage_name)
        mmc.waitForDevice(O3_stage_name)
        mmc.setPosition(z_start)
        mmc.waitForDevice(O3_stage_name)

        # grab O3 focus stage position
        z_stage_start = np.round(mmc.getPosition(),2)
        mmc.waitForDevice(O3_stage_name)
        if verbose:
            print(f'    O3 z-stage: {O3_stage_name} with position {z_stage_start}')

        # generate arrays
        num_z_steps= 31
        z_step_um = .25
        z_positions = np.round(np.linspace(z_start-z_step_um*(num_z_steps//2),
                                         z_start+z_step_um*(num_z_steps//2),
                                         num_z_steps),
                               2).astype(np.float64)
        chip_shape = snap_image(mmc, False).shape
        image = np.zeros((num_z_steps, chip_shape[0], chip_shape[1]))
        
        if verbose:
            print('    Start z-stack acquisition')

        for ii, z_pos in enumerate(z_positions):
            mmc.setPosition(z_pos)
            mmc.waitForDevice(O3_stage_name)
            image[ii, :, :] = snap_image(mmc, obis_on=obis_on)
            
            if verbose: 
                print(f'      Current position: {mmc.getPosition():.2f}')

        mmc.setPosition(z_start)
        mmc.waitForDevice(O3_stage_name)
        if verbose:
            print(f'    Stage moved to initial position: {mmc.getPosition():.2f}')
    else:
        image = snap_image(mmc, obis_on)
    
    if metric_type == "dct2d":
        metric, cropped_image = metric_shannon_dct(image,
                                            psf_radius_px=psf_radius_px,
                                            crop_size=crop_size)
    elif metric_type == "gauss2d":
        metric, cropped_image = metric_gauss2d(image, 
                                               crop_size=crop_size, 
                                               threshold=threshold,
                                               image_center=image_center)
    elif metric_type == "brightness2d":
        metric, cropped_image = metric_brightness(image,
                                                  crop_size=crop_size,
                                                  threshold=threshold,
                                                  image_center=image_center)
    # elif metric_type == "gauss3d":
    #     metric, cropped_image = metric_gauss3d(image,
    #                                            threshold,
    #                                            crop_size,
    #                                            image_center,
    #                                            metric_value="mean",
    #                                            verbose=verbose)
    return metric, cropped_image

def optimize_modes(
        mmc: CMMCorePlus,
        ao_mirror,
        metric_type: str,
        obis_on: bool = False,
        modes_to_optimize = [2,7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
        n_iter: int = 5, 
        n_steps: int = 5, 
        init_range: float = 0.25, 
        alpha: float =0.95,
        crop_size: int = 51, 
        threshold: float = 2000.0,
        display_data: bool = False,
        save_dir_path: Path = None,
        verbose: bool = False
):
    """
    Optimize Zernike modes to maximize image metric.

    Based on 3N measurement.
    
    Parameters
    ----------
    mmc: CMMCorePlus
        pymmcore-plus instance
    wavefront_corrector
        deformable mirror instance
    only_focus: bool
        include focus optimization. Default = False
    n_iter: int
        Number of overall iterations.
    n_steps: int 
        Steps per mode sweep (3, 5, or 7).
    init_range: float
        Initial sweep range for coefficient perturbations.
    alpha: float
        Range reduction factor (0 < alpha <= 1).
    display_data: bool
        display cropped images
    save_data: bool
        save cropped images
    
    Returns
    -------
    mode_coeff: ArrayLike 
        Optimized Zernike coefficients.
    """
    # Initialization
    initial_coeffs = ao_mirror.current_coeffs.copy() # coeff before optmization
    test_coeffs = initial_coeffs.copy() # modified coeffs to be applied to mirror
    optimized_coeffs = initial_coeffs.copy() # final coeffs after running iterations
    delta_range = init_range
    
    if metric_type=="gauss3d":
        # Turn off live display
        display_data=False
        
    # setup live data display and image saving
    if display_data:
        pass
            
    if save_dir_path:
        # Create a unique dir. in the given save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"optimization_{timestamp}"
        save_data_directory_path = save_dir_path / Path(directory_name)
        save_data_directory_path.mkdir(parents=True,exist_ok=True)
        
        # empty list for holding images / metrics / coeffs
        images = []
        opt_images = []
        opt_metrics = []
        opt_coeffs = []
        
    n_zernike_modes = max(modes_to_optimize)+1
    for k in range(n_iter):
        print(f"\nIteration {k+1}/{n_iter}:")
        
        # Get the ROI for this given iteration
        if (metric_type=="gauss2d") or (metric_type=="brightness2d") or (metric_type=="gauss3d"):
            image = snap_image(mmc,obis_on)
            image_center = get_image_center(image, threshold=threshold)
        else:
            image_center = None            
        
        # Calculate the starting metric, future pertubations must improve from here.
        zero_metric, start_image = measure_metric(mmc,
                                                    obis_on=obis_on,
                                                    metric_type=metric_type,
                                                    image_center=image_center,
                                                    threshold=threshold,
                                                    crop_size=crop_size,
                                                    verbose=verbose)
        if k==0:
            opt_metric = zero_metric
        if verbose:
            print(f"\n   This iterations initial metric is {zero_metric:.4f}")
            
        if save_dir_path:
            save_data_path = save_data_directory_path / Path(f"starting_image_{k}.tif")
            if metric_type=="gauss3d":
                imwrite(save_data_path,
                        start_image,
                        imagej=True,
                        resolution=(1.0 / .115, 1.0 / .115),
                        metadata={'axes': 'ZYX', 'spacing':0.250, 'unit':'um'})
            else:
                # save the initial image
                imwrite(save_data_path,
                        start_image,
                        imagej=True,
                        resolution=(1.0 / .115, 1.0 / .115),
                        metadata={'axes': 'YX', 'unit':'um'})
                
        for mode in modes_to_optimize:           
            print(f"  Optimizing mode {mode+1}/{n_zernike_modes}...")
            # Grab the current starting modes for this iteration
            current_mode_coeffs = ao_mirror.current_coeffs.copy()
            deltas = np.linspace(-delta_range, delta_range, n_steps)
            metrics = []
            for delta in deltas:
                test_coeffs = current_mode_coeffs.copy()
                test_coeffs[mode] += delta
                success = ao_mirror.set_modal_coefficients(test_coeffs)
                if not(success):
                    print("Setting mirror coefficients failed!")
                    metric = 0
                    cropped_image = np.zeros_like(images[0])
                else:
                    metric, cropped_image = measure_metric(mmc,
                                                           obis_on=obis_on,
                                                           threshold=threshold,
                                                           crop_size=crop_size,
                                                           metric_type=metric_type,
                                                           image_center=image_center,
                                                    verbose=verbose)
                    if metric==np.nan:
                        print("Metric is NAN, setting to 0")
                        metric = float(np.nan_to_num(metric))
                metrics.append(metric)
                
                if display_data:
                    pass
                        
                if save_dir_path:
                    images.append(cropped_image)
                if verbose:
                    print(f"    Delta={delta:.4f}, Metric={metric:.4f}")
            
            # Quadratic fit to determine optimal delta
            try:
                popt = quadratic_fit(deltas, metrics)
                a, b, c = popt
                if a >=0 or np.abs(a) < 10.0:
                    raise Exception
                optimal_delta = -b / (2 * a)
                if verbose:
                    print(f"    Fitted optimal delta: {optimal_delta:.4f}")
                    
                # if the delta is outside of the sample range do not update
                if (optimal_delta>delta_range) or (optimal_delta<-delta_range):
                    optimal_delta = 0
                    if verbose:
                        print(f"      Optimal delta is outside of delta_range: {-b / (2 * a):.3f}")
                        
            except Exception:
                optimal_delta = 0
                if verbose:
                    print(f"    Exception in fit occurred: {optimal_delta:.4f}")
                    print(f"    a value: {a:.3f}")
                    print(f"    b value: {b:.3f}")
                    
                    
            
            coeff_opt = current_mode_coeffs[mode] + optimal_delta
            
            # test the new coeff to make sure it improves the overall metric.
            test_coeffs[mode] = coeff_opt
            
            success = ao_mirror.set_modal_coefficients(test_coeffs)
            if not(success):
                print("Setting mirror coefficients failed, reverting to last metric!")
                coeff_to_keep = current_mode_coeffs[mode]
            else:
                # calculate the new metric to compare to zero-metric
                metric, cropped_image = measure_metric(mmc,
                                                        obis_on=obis_on,
                                                        crop_size=crop_size,
                                                        threshold=threshold,
                                                        metric_type=metric_type,
                                                        image_center=image_center,
                                                verbose=verbose)
                opt_image = cropped_image
                
                if metric>opt_metric:
                    # if the new metric is better, keep it
                    coeff_to_keep = coeff_opt
                    opt_metric = metric
                    if verbose:
                        print(f"      Keeping new coeff: {coeff_to_keep:.4f} with metric: {metric:.4f}")
                else:
                    # if not keep the original coeff
                    coeff_to_keep = current_mode_coeffs[mode]
                           
            # update mirror
            test_coeffs[mode] = coeff_to_keep
            _ = ao_mirror.set_modal_coefficients(test_coeffs)

        # Update the current_mode_coeffs
        current_mode_coeffs = ao_mirror.current_coeffs.copy()
        if verbose:
            print(f"current_mode_coeffs at end of iteration:\n{current_mode_coeffs}")
            
        if save_dir_path:
            opt_metrics.append(metric)
            opt_coeffs.append(current_mode_coeffs)
            opt_images.append(opt_image)
        
        # Reduce the sweep range for finer sampling around new optimal coefficient amplitude
        delta_range *= alpha
        if verbose:
            print(f" Reduced sweep range to {delta_range:.4f}",
                  f"\n Current metric: {metric:.4f}")
    
    optimized_coeffs = ao_mirror.current_coeffs.copy()

    if verbose:
        print(f"Starting coefficients:\n{initial_coeffs}",
              f"\nFinal optimized coefficients:\n{optimized_coeffs}")
        
    # apply new coefficeints to the mirror
    _ = ao_mirror.set_modal_coefficients(optimized_coeffs)
    
    if save_dir_path:
        if metric_type=="gauss3d":
            opt_images = np.asarray(opt_images,dtype=np.uint16)
            save_data_path = save_data_directory_path / Path("opt_images.tif")
            imwrite(
                save_data_path,
                opt_images,
                imagej=True,
                resolution=(1.0 / .115, 1.0 / .115),
                metadata={'axes': 'TZYX', 'spacing':0.250, 'unit':'um'}
            )
            
            _, final_image = measure_metric(mmc,
                                            metric_type=metric_type,
                                            image_center=image_center,
                                            obis_on=obis_on,
                                            threshold=threshold,
                                            crop_size=crop_size)
            
            save_data_path = save_data_directory_path / Path("final.tif")
            imwrite(
                save_data_path,
                final_image,
                imagej=True,
                resolution=(1.0 / .115, 1.0 / .115),
                metadata={'axes': 'ZYX', 'spacing':0.250, 'unit':'um'}
            )
        else:
            images = np.asarray(images,dtype=np.uint16)
            save_data_path = save_data_directory_path / Path("images.tif")
            imwrite(
                save_data_path,
                images,
                imagej=True,
                resolution=(1.0 / .115, 1.0 / .115),
                metadata={'axes': 'TYX'}
            )
            
            opt_images = np.asarray(opt_images,dtype=np.uint16)
            save_data_path = save_data_directory_path / Path("opt_images.tif")
            imwrite(
                save_data_path,
                opt_images,
                imagej=True,
                resolution=(1.0 / .115, 1.0 / .115),
                metadata={'axes': 'TYX'}
            )
            
            _, final_image = measure_metric(mmc,
                                            metric_type=metric_type,
                                            image_center=image_center,
                                            obis_on=obis_on,
                                            threshold=threshold,
                                            crop_size=crop_size)
            
            save_data_path = save_data_directory_path / Path("final.tif")
            imwrite(
                save_data_path,
                final_image,
                imagej=True,
                resolution=(1.0 / .115, 1.0 / .115),
                metadata={'axes': 'YX'}
            )
        
        # save the progression of metrics and coefficients
        save_data_path = save_data_directory_path / Path("metric_tracking.txt")
        np.savetxt(save_data_path, np.asarray(opt_metrics))
        save_data_path = save_data_directory_path / Path("coeff_tracking.txt")
        np.savetxt(save_data_path, np.asarray(opt_coeffs))
        
    if display_data:
        pass
    
    return optimized_coeffs

if __name__ == "__main__":
    tstart = time.time()
    # Create a unique directory to save this versions optimization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"O1_beads_optimization_dps_{timestamp}"
    save_dir_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\optimization_data") / Path(dir_name)
    mm_configPath = Path(r"C:\Program Files\Micro-Manager-2.0\wip_1217.cfg")
    wfc_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")
    wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc")
    haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat")
    wfc_flat_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\flat_actuator_positions.wcs")
    save_wfc_laser_save_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\a20250122_tilted_laser_actuator_positions.wcs")
    save1_wfc_laser_save_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250122_tilted_gauss2d_laser_actuator_positions.wcs")
    wfc_beads_save_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20240121_dps_beads_actuator_positions.wcs")
    wfc_argosim_save_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\argosim_actuator_positions.wcs")
    wfc_laser_save_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\laser_actuator_positions.wcs")
    
    # Load ao_mirror controller
    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                         haso_config_file_path = haso_config_file_path,
                         interaction_matrix_file_path = wfc_correction_file_path,
                         flat_positions_file_path = save1_wfc_laser_save_path,
                         coeff_file_path = None,
                         n_modes = 32,
                         modes_to_ignore = [])
    
    
    # Define imaging parameters:
    # For the laser spot, use exp=20ms, power=Off
    exposure = 10
    laser_power = 10
    threshold = 1000
    crop_size = 51
    obis_on = False
    metric_type = "gauss2d"
    verbose = True
    show_plot = True
    
    # Load pymmcore-plus and connect to hardware
    mmc = CMMCorePlus.instance()
    mmc.loadSystemConfiguration(mm_configPath)
    mmc.setConfig("Channels", "OFF")
    mmc.setExposure(exposure)
    mmc.setProperty(r"Coherent-Scientific Remote",r"Laser 488-150C - PowerSetpoint (%)",laser_power)
    
    optimized_coeffs = optimize_modes(
        mmc=mmc,
        ao_mirror=ao_mirror, 
        metric_type=metric_type, 
        modes_to_optimize=[7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31],
        n_iter=1,
        n_steps=3,
        init_range=.4,
        alpha=.8,
        threshold=threshold,
        crop_size=crop_size,
        obis_on=obis_on,
        verbose=verbose,
        display_data=show_plot,
        save_dir_path=save_dir_path
    )
    
    print(f"\nFinal difference in mirror positions: \n{ao_mirror.flat_positions - ao_mirror.current_positions}")
    # plotDM(ao_mirror.current_positions, "Final mirror positions")
    
    # save optimized positions
    ao_mirror.save_mirror_state(save1_wfc_laser_save_path)
    
    ao_mirror = None



