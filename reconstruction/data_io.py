#!/usr/bin/env python
'''
QI2lab OPM suite
Reconstruction tools

Read and write metadata; read raw data; read pre-generated OPM psfs
'''

import re
from npy2bdv.npy2bdv import BdvEditor
import pandas as pd
import numpy as np
from pathlib import Path
from tifffile import tifffile
from datetime import datetime

def read_metadata(fname):
    """
    Read data from csv file consisting of one line giving titles, and the other giving values. Return as dictionary

    :param fname: str
        filename
    :return metadata: dict
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

def write_metadata(data_dict, save_path):
    """
    Write dictionary as CSV file

    :param data_dict: dict
        metadata dictionary
    :param save_path: Path
        path for file
    
    :return: None
    """
    
    pd.DataFrame([data_dict]).to_csv(save_path)


def return_data_numpy(dataset, time_axis, channel_axis, num_images, excess_images, y_pixels,x_pixels):
    """
    :param dataset: dataset
        pycromanager dataset object
    :param channel_axis: int
        channel axis index
    :param time_axis: int
        time axis index
    :param num_images: int
        number of images in scan direction (TO DO: change to tuple to load range)
    :param y_pixels: int
        y pixels
    :param x_pixels: int
        x pixels

    :return data_numpy: ndarray
        3D numpy array of OPM data. First axis is scan sxis
    """

    data_numpy = np.empty([(num_images-excess_images),y_pixels,x_pixels]).astype(np.uint16)
    j = 0
    for i in range(excess_images,num_images):
        if (time_axis is None):
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i,channel=0)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, c=channel_axis,channel=0)
        else:
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis,channel=0)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis, c=channel_axis,channel=0)
        j = j + 1

    return data_numpy

def return_data_dask(dataset,excess_images,channel_id):
    """
    :param dataset: dataset
        pycromanager dataset object
    :param excess_images: int
        number of excess images for stage warmup
    :param channel_axis: str
        channel axis name

    :return data: np.ndarray
        load disk
    """

    data = dataset.as_array(e=excess_images+1,channel=channel_id,axes=['s'])
    data = data.compute(num_workers=4)

    return np.squeeze(data)

def return_data_numpy_widefield(dataset, channel_axis, ch_BDV_idx, num_z, y_pixels,x_pixels):
    """
    :param dataset: dataset 
        pycromanager dataset object
    :param channel_axis: int 
        channel axis index
    :param time_axis: int
        time axis index
    :param num_images: int 
        number of images in z stack 
    :param y_pixels: int
        y pixels
    :param x_pixels: int
        x pixels

    :return data_numpy: ndarray 
        3D numpy array of requested data
    """

    data_numpy = np.empty([num_z,y_pixels,x_pixels]).astype(np.uint16)

    for i in range(num_z):
        if (channel_axis is None):
            data_numpy[i,:,:] = dataset.read_image(z=i)
        else:
            data_numpy[i,:,:] = dataset.read_image(z=i, c=channel_axis, channel=ch_BDV_idx)

    return data_numpy

def stitch_data(path_to_xml,iterative_flag):
    """
    Call BigStitcher via Python to calculate rigid stitching transformations across tiles and rounds

    :param path_to_xml: Path
        path to BDV XML. BDV H5 must be present for loading
    :param iterative_flag: Bool
        flag if multiple rounds need to be aligned
    """

    # TO DO: 1. write either pyimagej bridge + macro OR call FIJI/BigStitcher in headless mode.
    #        2. fix flipped x-axis between Python and FIJI. Easier to flip data in Python than deal with
    #           annoying affine that flips data.



def return_affine_xform(path_to_xml,r_idx,y_idx,z_idx,total_z_pos):
    """
    Return affine transformation for a given tile from BDV XML

    :param path_to_xml: Path
        path to BDV XML. BDV H5 must be present for loading
    :param r_idx: integer
        round index
    :param t_idx: integer
        time index
    :param y_idx: integer 
        y tile index
    :param z_idx: integer 
        z tile index
    :param total_z_pos: integer
        total number of z tiles in data
    :return data_numpy: ndarray
        4D numpy array of all affine transforms
    """ 

    bdv_editor = BdvEditor(str(path_to_xml))
    tile_idx = (y_idx+z_idx)+(y_idx*(total_z_pos-1))

    affine_xforms = []
    read_affine_success = True
    affine_idx = 0
    while read_affine_success:
        try:
            affine_xform = bdv_editor.read_affine(time=r_idx,illumination=0,channel=0,tile=tile_idx,angle=0,index=affine_idx)
        except:
            read_affine_success = False
        else:
            affine_xforms.append(affine_xform)
            affine_idx = affine_idx + 1
            read_affine_success = True

    return affine_xforms

def return_opm_psf(wavelength_um,z_idx):
    """
    Load pre-generated OPM psf

    TO DO: write checks and generate PSF if it does not exist on disk

    :param wavelength: float
        wavelength in um

    :param z_idx: int
        index of z slice. Assume 15 steps above coverslip for now
        
    :return psf: ndarray
        pre-generated skewed PSF
    """ 
    if z_idx == 0 or z_idx == 1:
        psf_idx = 0
    elif z_idx == 2 or z_idx == 3:
        psf_idx = 1
    elif z_idx == 4 or z_idx == 5:
        psf_idx = 2
    elif z_idx == 6 or z_idx == 7:
        psf_idx = 3
    elif z_idx == 8 or z_idx == 9:
        psf_idx = 4
    elif z_idx == 10 or z_idx == 11:
        psf_idx = 5
    elif z_idx == 12 or z_idx == 13:
        psf_idx = 6
    elif z_idx == 14 or z_idx == 15:
        psf_idx = 7

    wavelength_nm = int(wavelength_um*100)
    pz = int((psf_idx) * 15)

    psf_path = Path('psfs') / Path('opm_psf_w'+str(wavelength_nm)+'_p'+str(pz)+'.tiff')
    opm_psf = tifffile.imread(psf_path)

    return np.flipud(opm_psf)

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")