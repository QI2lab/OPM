
#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
OPM I/O functions
----------------------------------------------------------------------------------------
Peter Brown
Douglas Shepherd
12/11/2021
douglas.shepherd@asu.edu
----------------------------------------------------------------------------------------
'''

import re
import pandas as pd
import numpy as np
from pathlib import Path
import tifffile
from datetime import datetime


def read_metadata(fname):
    """
    Read data from csv file consisting of one line giving titles, and the other giving values. Return as dictionary
    :param fname:
    :return metadata:
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

def read_config_file(config_path):
    """
    Read data from csv file consisting of one line giving titles, and the other giving values. Return as dictionary
    :param config_path: Path
        Location of configuration file
    :return dict_from_csv: dict
        instrument configuration metadata
    """

    dict_from_csv = pd.read_csv(config_path, header=None, index_col=0, squeeze=True).to_dict()

    return dict_from_csv

def read_fluidics_program(program_path):
    """
    Read fluidics program from CSV file as pandas dataframe
    :param program_path: Path
        location of fluidics program
    :return df_program: Dataframe
        dataframe containing fluidics program 
    """

    df_program = pd.read_csv(program_path)
    return df_program

def write_metadata(data_dict, save_path):
    """
    Write metadata file as csv

    :param data_dict: dict
        dictionary of metadata entries
    :param save_path: Path
        path for file
    :return None:
    """

    pd.DataFrame([data_dict]).to_csv(save_path)

def return_data_from_zarr_to_numpy(dataset, time_idx, channel_idx, num_images, y_pixels,x_pixels):
    """
    :param dataset: zarr dataset object
    :param time_idx: integer time_axis
    :param channel_idx: integer channel index
    :param num_images: integer for number of images from sweep to return 
    :param y_pixels: integer for y pixel size
    :param x_pixels: integer for x pixel size
    :return data_numpy: 3D numpy array of requested data
    """

    data_numpy = np.empty([num_images,y_pixels,x_pixels]).astype(np.uint16)
    data_numpy = dataset[time_idx,channel_idx,0:num_images,:]

    return data_numpy


def return_opm_psf(ch_idx):
    """
    Load pre-generated OPM psf

    TO DO: write checks and generate PSF if it does not exist on disk

    :param z_idx: int
        index of z slice. Assume 15 steps above coverslip for now
        
    :return psf: ndarray
        pre-generated skewed PSF
    """ 

    root_path = Path(__file__).parent.resolve()


    if ch_idx == 0:
        psf_name = Path('psfs') / Path('opm_psf_420_nm.tif')
    elif ch_idx == 1:
        psf_name = Path('psfs') / Path('opm_psf_520_nm.tif')
    elif ch_idx == 2:
        psf_name = Path('psfs') / Path('opm_psf_580_nm.tif')
    elif ch_idx == 3:
        psf_name = Path('psfs') / Path('opm_psf_670_nm.tif')
    elif ch_idx == 4:
        psf_name = Path('psfs') / Path('opm_psf_780_nm.tif')


    psf_path = root_path / psf_name
    opm_psf = tifffile.imread(psf_path)

    return opm_psf

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")