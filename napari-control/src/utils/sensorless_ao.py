"""Sensorless adaptive optics.

TO DO:
- Load interaction matrix from disk
- Set and get Zernike mode amplitudes from mirror
- Might need HASO functions to do this, since Zernike need to be composed given the pupil

2024/12 DPS initial work
"""
from src.hardware.AOMirror import AOMirror

import numpy as np
import h5py
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Sequence, List
from scipy.fftpack import dct
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from pathlib import Path
# localize_psf imports
# from localize_psf.localize import (localize_beads_generic, 
#                                    get_param_filter,
#                                    get_coords, 
#                                    plot_fit_roi)
# from localize_psf.fit import fit_model
# import localize_psf.fit_psf as psf

#-------------------------------------------------#
# Plotting functions
#-------------------------------------------------#

def plot_zernike_coeffs(optimal_coefficients: ArrayLike,
                        zernike_mode_names: ArrayLike,
                        save_path: Optional[Path] = None,
                        show_fig: Optional[bool] = False):
    """_summary_

    Parameters
    ----------
    optimal_coefficients : ArrayLike
        _description_
    save_path : Path
        _description_
    showfig : bool
        _description_
    """
    import matplotlib.pyplot as plt
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']  
    markers = ['x', 'o', '^', 's', '*']  

    # populate plots
    for i in range(len(zernike_mode_names)):
        for j in range(optimal_coefficients.shape[0]):
            marker_style = markers[j % len(markers)]
            ax.scatter(optimal_coefficients[j, i], i, 
                       color=colors[j % len(colors)], s=125, marker=marker_style)  
        ax.axhline(y=i, linestyle="--", linewidth=1, color='k')
        
    # Plot a vertical line at 0 for reference
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    # Customize the plot
    ax.set_yticks(np.arange(len(zernike_mode_names)))
    ax.set_yticklabels(zernike_mode_names)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Zernike mode coefficients at each iteration")
    ax.set_xlim(-0.15, 0.15)

    # Add a legend for time points
    ax.legend([f'Iteration: {i+1}' for i in range(optimal_coefficients.shape[0])], loc='upper right')

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_path:
        fig.savefig(save_path)


def plot_metric_progress(optimal_metrics: ArrayLike,
                         modes_to_optimize: List[int],
                         zernike_mode_names: List[str],
                         save_path: Optional[Path] = None,
                         show_fig: Optional[bool] = False):
    """_summary_

    Parameters
    ----------
    metrics : ArrayLike
        _description_
    modes_to_optmize : List[int]
        _description_
    zernike_mode_names : List[str]
        _description_
    save_path : Optional[Path], optional
        _description_, by default None
    show_fig : Optional[bool], optional
        _description_, by default False
    """
    import matplotlib.pyplot as plt
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each iteration
    colors = ['b', 'g', 'r', 'c', 'm']
    markers = ['x', 'o', '^', 's', '*']

    # Loop over iterations and plot each series
    for ii, series in enumerate(optimal_metrics):
        ax.plot(series, color=colors[ii], label=f"iteration {ii}", marker=markers[ii], linestyle="--", linewidth=1)

    # Set the x-axis to correspond to the modes_to_optimize
    mode_labels = [zernike_mode_names[i] for i in modes_to_optimize]
    ax.set_xticks(np.arange(len(mode_labels))) 
    ax.set_xticklabels(mode_labels, rotation=60, ha="right", fontsize=16) 

    # Customize the plot
    ax.set_ylabel("Metric", fontsize=16)
    ax.set_title("Optimal Metric Progress per Iteration", fontsize=18)

    ax.legend(fontsize=15)
    
    plt.tight_layout()
    
    if show_fig:
        plt.show()
    if save_path:
        fig.savefig(save_path)


#-------------------------------------------------#
# Helper functions for saving optmization results
#-------------------------------------------------#

def save_optimization_results(iteration_images: ArrayLike,
                              mode_delta_images: ArrayLike,
                              optimal_coefficients: ArrayLike,
                              optimal_metrics: ArrayLike,
                              modes_to_optimize: List[int],
                              results_save_path: Path):
    """_summary_

    Parameters
    ----------
    optimal_coefficients : ArrayLike
        _description_
    optimal_metrics : ArrayLike
        _description_
    modes_to_optimize : List[int]
        _description_
    results_save_path : Path
        _description_
    """
    with h5py.File(str(results_save_path), "w") as f:
                f.create_dataset("optimal_images", data=iteration_images)
                f.create_dataset("mode_delta_images", data=mode_delta_images)
                f.create_dataset("optimal_coefficients", data=optimal_coefficients)
                f.create_dataset("optimal_metrics", data=optimal_metrics)
                f.create_dataset("modes_to_optimize", data=modes_to_optimize)
                f.create_dataset("zernike_mode_names", data=np.array(AOMirror.mode_names, dtype="S"))
    
    
def load_optimization_results(results_path: Path):
    """_summary_

    Parameters
    ----------
    results_path : Path
        _description_
    """
    # Load the mixed dictionary from HDF5
    with h5py.File(str(results_path), "r") as f:
        optimal_images = f["optimal_images"][:]
        mode_delta_images = f["mode_delta_images"][:]
        optimal_coefficients = f["optimal_coefficients"][:]
        optimal_metrics = f["optimal_metrics"][:]
        modes_to_optimize = f["modes_to_optimize"][:]
        zernike_mode_names = [name.decode("utf-8") for name in f["zernike_mode_names"][:]]

    return optimal_images, mode_delta_images, optimal_coefficients, optimal_metrics, modes_to_optimize, zernike_mode_names


#-------------------------------------------------#
# Functions for preparing data
#-------------------------------------------------#

def get_image_center(image: ArrayLike, threshold: float) -> Tuple[int, int]:
    """
    Calculate the center of an image using a thresholded binary mask.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    threshold : float
        Intensity threshold for binarization.

    Returns
    -------
    center : Tuple[int, int]
        Estimated center coordinates (x, y).
    """
    try:
        binary_image = image > threshold
        center = center_of_mass(binary_image)
        center = tuple(map(int, center))
    except Exception:
        center = (image.shape[1]//2, image.shape[0]//2)
    return center


def get_cropped_image(image: ArrayLike, crop_size: int, center: Tuple[int, int]) -> ArrayLike:
    """
    Extract a square region from an image centered at a given point.

    Parameters
    ----------
    image : ArrayLike
        Input 2D or 3D image.
    crop_size : int
        Half-width of the cropping region.
    center : Tuple[int, int]
        Center coordinates (x, y) of the crop.

    Returns
    -------
    cropped_image : ArrayLike
        Cropped region from the input image.
    """
    if len(image.shape) == 3:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[1])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[2])
        cropped_image = image[:, x_min:x_max, y_min:y_max]
    else:
        x_min, x_max = max(center[0] - crop_size, 0), min(center[0] + crop_size, image.shape[0])
        y_min, y_max = max(center[1] - crop_size, 0), min(center[1] + crop_size, image.shape[1])
        cropped_image = image[x_min:x_max, y_min:y_max]
    return cropped_image


#-------------------------------------------------#
# Functions for fitting and calculations
#-------------------------------------------------#

def gauss2d(coords_xy: ArrayLike, amplitude: float, center_x: float, center_y: float,
            sigma_x: float, sigma_y: float, offset: float) -> ArrayLike:
    """
    Generates a 2D Gaussian function for curve fitting.

    Parameters
    ----------
    coords_xy : ArrayLike
        Meshgrid coordinates (x, y).
    amplitude : float
        Peak intensity of the Gaussian.
    center_x : float
        X-coordinate of the Gaussian center.
    center_y : float
        Y-coordinate of the Gaussian center.
    sigma_x : float
        Standard deviation along the x-axis.
    sigma_y : float
        Standard deviation along the y-axis.
    offset : float
        Background offset intensity.

    Returns
    -------
    raveled_gauss2d : ArrayLike
        Flattened 2D Gaussian function values.
    """
    x, y = coords_xy
    raveled_gauss2d = (
        offset +
        amplitude * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) + ((y - center_y)**2 / (2 * sigma_y**2)))
        )
    ).ravel()

    return raveled_gauss2d


def otf_radius(img: ArrayLike, psf_radius_px: float) -> int:
    """
    Computes the optical transfer function (OTF) cutoff frequency.

    Parameters
    ----------
    img : ArrayLike
        2D image.
    psf_radius_px : float
        Estimated point spread function (PSF) radius in pixels.

    Returns
    -------
    cutoff : int
        OTF cutoff frequency in pixels.
    """
    w = min(img.shape)
    psf_radius_px = max(1, np.ceil(psf_radius_px))  # clip all PSF radii below 1 px to 1.
    cutoff = np.ceil(w / (2 * psf_radius_px)).astype(int)

    return cutoff


def normL2(x: ArrayLike) -> float:
    """
    Computes the L2 norm of an n-dimensional array.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    l2norm : float
        L2 norm of the array.
    """
    l2norm = np.sqrt(np.sum(x.flatten() ** 2))

    return l2norm


def shannon(spectrum_2d: ArrayLike, otf_radius: int = 100) -> float:
    """
    Computes the Shannon entropy of an image spectrum within a given OTF radius.

    Parameters
    ----------
    spectrum_2d : ArrayLike
        2D spectrum of an image (e.g., from DCT or FFT).
    otf_radius : int, optional
        OTF support radius in pixels (default is 100).

    Returns
    -------
    entropy : float
        Shannon entropy of the spectrum.
    """
    h, w = spectrum_2d.shape
    y, x = np.ogrid[:h, :w]

    # Circular mask centered at (0,0) for DCT
    support = (x**2 + y**2) < otf_radius**2

    spectrum_values = np.abs(spectrum_2d[support])
    total_energy = np.sum(spectrum_values)

    if total_energy == 0:
        return 0  # Avoid division by zero

    probabilities = spectrum_values / total_energy
    entropy = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))

    return entropy


def dct_2d(image: ArrayLike, cutoff: int = 100) -> ArrayLike:
    """
    Computes the 2D discrete cosine transform (DCT) of an image with a cutoff.

    Parameters
    ----------
    image : ArrayLike
        2D image array.
    cutoff : int, optional
        OTF radius cutoff in pixels (default is 100).

    Returns
    -------
    dct_2d : ArrayLike
        Transformed image using DCT.
    """
    dct_2d = dct(dct(image.astype(np.float32), axis=0, norm='ortho'), axis=1, norm='ortho')

    return dct_2d


def quadratic(x: float, a: float, b: float, c: float) -> ArrayLike:
    """
    Quadratic function evaluation at x.

    Parameters
    ----------
    x : float
        Point to evaluate.
    a : float
        x^2 coefficient.
    b : float
        x coefficient.
    c : float
        Offset.

    Returns
    -------
    value : float
        a * x^2 + b * x + c
    """
    return a * x**2 + b * x + c


def quadratic_fit(x: ArrayLike, y: ArrayLike) -> Sequence[float]:
    """
    Quadratic function for curve fitting.

    Parameters
    ----------
    x : ArrayLike
        1D x-axis data.
    y : ArrayLike
        1D y-axis data.

    Returns
    -------
    coeffs : Sequence[float]
        Fitting parameters.
    """
    A = np.vstack([x**2, x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

    return coeffs


#-------------------------------------------------#
# Functions to calculate image metrics
#-------------------------------------------------#

def metric_brightness(image: ArrayLike,
                      crop_size: Optional[int] = None,
                      threshold: Optional[float] = 100,
                      image_center: Optional[int] = None,
                      return_image: Optional[bool] = False
                      ) -> float:
    """
    Compute weighted metric for 2D Gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)
        else:
            center = image_center
        image = get_cropped_image(image, crop_size, center)

    if len(image.shape) == 3:
        image = np.max(image, axis=0)

    image_perc = np.percentile(image, 90)
    max_pixels = image[image >= image_perc]

    if return_image:
        return np.mean(max_pixels), image
    else:
        return np.mean(max_pixels)


def metric_gauss2d(image: ArrayLike,
                   crop_size: Optional[int] = None,
                   threshold: Optional[float] = 100,
                   image_center: Optional[int] = None,
                   return_image: Optional[bool]= False
                   ) -> float:
    """Compute weighted metric for 2D gaussian.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
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
        
        if (weighted_metric <= 0) or (weighted_metric > 100):
            weighted_metric = 1e-12 
    except Exception:
        weighted_metric = 1e-12
        
        
    if return_image:
        return weighted_metric, image
    else:
        return weighted_metric


def metric_gauss3d(
    image: ArrayLike,
    metric_value: str = "mean",
    crop_size: Optional[int] = None,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    verbose: Optional[bool] = False,
    return_image: Optional[bool] = False
    ):
    """Compute weighted metric for 3D Gaussian using LocalizePSF
    
    Parameters
    ----------
    image : ArrayLike
        2D image.
    metric_value: str
        Whether to average fit values or generate an average PSF and fit the result.
    threshold : float, optional
        Initial threshold to find spot (default is 100).
    crop_size_px : int, optional
        Crop size in pixels, one side (default is 20).
    image_center : Optional[int], optional
        Center of the image to crop (default is None).
    return_image : Optional[bool], optional
        Whether to return the cropped image (default is False).

    Returns
    -------
    weighted_metric : float
        Weighted metric value.
    """
    pass
    """
    Need to work on this:
    1. is there a conflict installing any of the localize_psf packages?
    """
    # # Optionally crop the image
    # if crop_size:    
    #     if image_center is None:
    #         center = get_image_center(image, threshold)
    #     else:
    #         center = image_center
    #     # crop image
    #     image = get_cropped_image(image, crop_size, center)
        
    # image = image / np.max(image)
    # image = image.astype(np.float32)
        
    # # Define coordinates to pass to localization, use pixel units
    # # Using pixel units, but assumes we are using 0.270 z-steps
    # dxy = 1 # 0.115
    # dz  = 0.250 / 0.115 # 0.250 
    # coords_3d = get_coords(image.shape, (dz, dxy, dxy))
    # # coords_2d = get_coords(cropped_image.shape[1:], (dxy, dxy))
    
    # # Prepare filter for localization
    # sigma_bounds = ((0.1, 0.1),(100, 100)) # [xy min, xy max, z min, z max]
    # amp_bounds = (0.1, 2.0) # [min / max]
    # param_filter = get_param_filter(coords_3d,
    #                                 fit_dist_max_err=(5, 5),
    #                                 min_spot_sep=(10, 6),
    #                                 amp_bounds=amp_bounds,
    #                                 dist_boundary_min=[3, 3],
    #                                 sigma_bounds=sigma_bounds
    #                             )
    # filter = param_filter  
     
    # # define roi sizes used in fitting, assumes a minimum 3um z-stack, dz=0.27um
    # fit_roi_size = [9, 7, 7]
    
    # # Run localization function
    # model = psf.gaussian3d_psf_model()
    # _, r, img_filtered = localize_beads_generic(
    #     image,
    #     drs=(dz, dxy, dxy),
    #     threshold=0.5,
    #     roi_size=fit_roi_size,
    #     filter_sigma_small=None,
    #     filter_sigma_large=None,
    #     min_spot_sep=(10,10),
    #     model=model,
    #     filter=filter,
    #     max_nfit_iterations=100,
    #     use_gpu_fit=False,
    #     use_gpu_filter=False,
    #     return_filtered_images=True,
    #     fit_filtered_images=False,
    #     verbose=True
    #     )
    
    # if r is None:
    #     print("no beads found!")
    #     return 0, image[image.shape[0]//2]
    # else:
    #     to_keep = r["to_keep"]
    #     fit_params = r["fit_params"]
    #     sz = fit_params[to_keep, 5]
    #     sxy = fit_params[to_keep, 4]
    #     amp = fit_params[to_keep, 0]
    #     # Use averages over fit results
    #     if metric_value=="mean":
    #         sz = np.mean(fit_params[to_keep, 5])
    #         sxy = np.mean(fit_params[to_keep, 4])
    #         amp = np.mean(fit_params[to_keep, 0])
    #     elif metric_value=="median":
    #         sz = np.median(fit_params[to_keep, 5])
    #         sxy = np.median(fit_params[to_keep, 4])
    #         amp = np.median(fit_params[to_keep, 0])
    #     elif metric_value=="average":
    #         # Generate average PSF
    #         fit_roi_size_pix = np.round(np.array(fit_roi_size) / np.array([dz, dxy, dxy])).astype(int)
    #         fit_roi_size_pix += (1 - np.mod(fit_roi_size_pix, 2))

    #         psfs_real = np.zeros((1) + tuple(fit_roi_size_pix))
    #         # otfs_real = np.zeros(psfs_real.shape, dtype=complex)
    #         fit_params_average = np.zeros((1, model.nparams))
    #         psf_coords = None

    #         # only use a percent of bead results based on the sxy
    #         percentile = 50
    #         if percentile:
    #             sigma_max = np.percentile(fit_params[:, 4][to_keep], percentile)
    #             to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

    #             # get centers
    #             centers = np.stack((fit_params[:, 3][to_use],
    #                                 fit_params[:, 2][to_use],
    #                                 fit_params[:, 1][to_use]), axis=1)

    #         # find average experimental psf/otf
    #         psfs_real, psf_coords = psf.average_exp_psfs(r["data"],
    #                                                      coords_3d,
    #                                                      centers,
    #                                                      fit_roi_size_pix,
    #                                                      backgrounds=fit_params[:, 5][to_use],
    #                                                      return_psf_coords=True)

    #         # fit average experimental psf
    #         def fn(p): return model.model(psf_coords, p)
    #         init_params = model.estimate_parameters(psfs_real, psf_coords)

    #         results = fit_model(psfs_real, fn, init_params, jac='3-point', x_scale='jac')
    #         fit_params_average = results["fit_params"]     

    #         sz = fit_params_average[5]
    #         sxy = fit_params_average[4]
    #         amp = fit_params_average[0]    
        
    #     # TODO: Refine weighted metric if needed
    #     weight_amp = 1 # scales amplitude to a value between 0-65
    #     # SJS: Normalize image and remove brightness
    #     weight_xy = 2 
    #     weight_z = 2
    #     weighted_metric = weight_amp * amp + weight_xy / sxy + weight_z / sz + np.exp(-1*(sxy+sz-6)**2)

    #     if return_image:
    #         return weighted_metric, image
    #     else:
    #         return weighted_metric


def metric_shannon_dct(
    image: ArrayLike, 
    psf_radius_px: float = 3,
    crop_size: Optional[int] = 501,
    threshold: Optional[float] = 100,
    image_center: Optional[int] = None,
    return_image: Optional[bool] = False
    ) -> float:
    """Compute the Shannon entropy metric using DCT.

    Parameters
    ----------
    image : ArrayLike
        2D image.
    psf_radius_px : float, optional
        Estimated point spread function (PSF) radius in pixels (default: 3).
    crop_size : Optional[int], optional
        Crop size for image (default: 501).
    threshold : Optional[float], optional
        Intensity threshold to find the center (default: 100).
    image_center : Optional[int], optional
        Custom image center (default: None).
    return_image : Optional[bool], optional
        Whether to return the image along with the metric (default: False).
    
    Returns
    -------
    entropy_metric : float
        Shannon entropy metric.
    """
    # Crop image if necessary
    if crop_size:
        if image_center is None:
            center = get_image_center(image, threshold)  # Ensure this function is defined
        else:
            center = image_center

        # Crop image (ensure get_cropped_image is correctly implemented)
        image = get_cropped_image(image, crop_size, center)

    # Compute the cutoff frequency based on OTF radius
    cutoff = otf_radius(image, psf_radius_px)

    # Compute DCT
    dct_result = dct_2d(image)

    # Compute Shannon entropy within the cutoff radius
    shannon_dct = shannon(dct_result, cutoff)

    if return_image:
        return shannon_dct, image
    else:
        return shannon_dct


#-------------------------------------------------#
# Run as script 'keeps mirror flat'
#-------------------------------------------------#
if __name__ == "__main__":
    """Keeps the mirror in it's flat position
    """
    wfc_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")
    wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc")
    haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat")
    wfc_flat_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\flat_actuator_positions.wcs")
    wfc_calibrated_flat_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250122_tilted_gauss2d_laser_actuator_positions.wcs")
    
    # Load ao_mirror controller
    # ao_mirror puts the mirror in the flat_position state to start.
    ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                         haso_config_file_path = haso_config_file_path,
                         interaction_matrix_file_path = wfc_correction_file_path,
                         flat_positions_file_path = wfc_calibrated_flat_path)
    
    input("Press enter to exit . . . ")
    ao_mirror = None



