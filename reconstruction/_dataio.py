#!/usr/bin/env python
'''
QI2lab OPM suite
Reconstruction tools

Read and write metadata; read raw data; read affine transforms; read pre-generated OPM psfs
'''

import re
from npy2bdv.npy2bdv import BdvEditor
import pandas as pd
import numpy as np
from pathlib import Path
from tifffile import tifffile
from datetime import datetime
from pycromanager import Dataset
import zarr
from typing import Optional

def read_metadata(fname: str) -> dict:
    """
    Read data from csv file consisting of one line giving titles, 
    and the other giving values. Return as dictionary

    Parameters
    ----------
    fname: str
        filename
    
    Returns
    -------
    metadata: dict
        metadata dictionary
    """
    scan_data_raw_lines = []

    with open(fname, "r") as f:
        for line in f:
            scan_data_raw_lines.append(line.replace("\n", ""))

    titles = scan_data_raw_lines[0].split(",")

    # convert values to appropriate datatypes
    vals = scan_data_raw_lines[1].split(",")
    for ii in range(len(vals)):
        if re.fullmatch("\d+", vals[ii]):
            vals[ii] = int(vals[ii])
        elif re.fullmatch("\d*.\d+", vals[ii]):
            vals[ii] = float(vals[ii])
        elif vals[ii].lower() == "False".lower():
            vals[ii] = False
        elif vals[ii].lower() == "True".lower():
            vals[ii] = True
        else:
            # otherwise, leave as string
            pass

    # convert to dictionary
    metadata = {}
    for t, v in zip(titles, vals):
        metadata[t] = v

    return metadata

def write_metadata(data_dict: dict,
                   save_path: Path) -> None:
    """
    Write dictionary as CSV file using Pandas.

    Parameters
    ----------
    data_dict: dict
        metadata dictionary
    save_path: Path
        path for file
    
    Returns
    -------
    None
    """
    
    pd.DataFrame([data_dict]).to_csv(save_path)

def return_data_dask(dataset: Dataset,
                     excess_images: int,
                     channel_id: str) -> np.ndarray:
    """
    Return OPM data as numpy array using NDTIFF Dask API.

    Parameters
    ----------
    dataset: dataset
        pycromanager dataset object
    excess_images: int
        number of excess images for ASI stage scan warmup
    channel_axis: str
        channel axis name

    Returns
    -------
    data: np.ndarray
        data as Dask Array
    """

    data = dataset.as_array(e=excess_images+1,channel=channel_id,axes=['s'])
    data = data.compute(num_workers=4)

    return np.squeeze(data)

def return_affine_xform(path_to_xml: Path,
                        tile_idx: int,
                        r_idx: int,
                        x_idx: int,
                        y_idx: int,
                        z_idx: int,
                        verbose: int = 0) -> np.ndarray:
    """
    Return affine transformation for a given tile from BDV XML.
    ONLY works for v3 OPM acquistions!

    Parameters
    ----------
    zarr_output_path: Path
        path to zarr output
    path_to_xml: Path
        path to BDV XML
    r_idx : int
        round index
    x_idx: int
        x tile index
    y_idx: int 
        y tile index
    z_idx: int 
        z tile index
    total_x_pos: int
        total number of x tiles in data
    total_y_pos: int
        total number of y tiles in data
    total_z_pos: int
        total number of z tiles in data
    verbose : int
        verbose output for debugging

    Returns
    -------
    data_numpy: np.ndarray
        4D numpy array of all affine transforms
    """ 

    # construct tile name
    if verbose > 0:
        print('r_idx:', r_idx, 'x_idx:', x_idx, 'y_idx:', y_idx, 'z_idx:', z_idx, 'tile_idx:', tile_idx)

    # open BDV XML
    bdv_editor = BdvEditor(str(path_to_xml),skip_h5=True)

    affine_xforms = []
    read_affine_success = True
    affine_idx = 0
    while read_affine_success:
        try:
            affine_xform = bdv_editor.read_affine(time=r_idx,
                                                  illumination=0,
                                                  channel=0,
                                                  tile=tile_idx,
                                                  angle=0,
                                                  index=affine_idx, 
                                                  tile_offset=0,
                                                  verbose=verbose)
        except:
            read_affine_success = False
        else:
            affine_xforms.append(affine_xform)
            affine_idx = affine_idx + 1
            read_affine_success = True

    return affine_xforms

def return_opm_psf(wavelength_um: float,
                   z_idx: Optional[float] = None) -> np.ndarray:
    """
    Load pre-generated OPM psf

    Parameters
    ----------
    wavelength: float
        wavelength in um
    z_idx: int
        index of z slice. Assume 15 um steps above coverslip for now
        
    Returns
    psf: np.ndarray
        pre-generated skewed PSF
    """ 

    wavelength_nm = int(wavelength_um*100)

    psf_path = Path('psfs') / Path('opm_psf_w'+str(wavelength_nm).rstrip("0")+'_p0.tiff')
    opm_psf = tifffile.imread(psf_path)

    return np.flipud(opm_psf)

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")