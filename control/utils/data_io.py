#!/usr/bin/env python

import re
from npy2bdv import BdvEditor
import pandas as pd
import numpy as np
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

    :return df_fluidics: Dataframe
        dataframe containing fluidics program 
    """

    try:                
        df_fluidics = pd.read_csv(program_path)            
        df_fluidics = df_fluidics[["round", "source", "time", "pump"]]
        df_fluidics.dropna(axis=0, how='any', inplace=True)
        df_fluidics["round"] = df_fluidics["round"].astype(int)
        df_fluidics["pump"] = df_fluidics["pump"].astype(int)

        print("Fluidics program loaded")
    except Exception as e:
        raise Exception("Error in loading fluidics file:\n", e)

    return df_fluidics

def write_metadata(data_dict, save_path):
    """

    :param data_dict: dictionary of metadata entries
    :param save_path:
    :return:
    """
    pd.DataFrame([data_dict]).to_csv(save_path)

def return_data_numpy(dataset, time_axis, channel_axis, num_images, excess_images, y_pixels,x_pixels):
    """
    :param dataset: pycromanager dataset object
    :param channel_axis: integer channel index
    :param time_axis: integer time_axis
    :param num_images: integer for number of images to return 
    :param y_pixels: integer for y pixel size
    :param x_pixels: integer for x pixel size
    :return data_numpy: 3D numpy array of requested data
    """

    data_numpy = np.empty([(num_images-excess_images),y_pixels,x_pixels]).astype(np.uint16)
    j = 0
    for i in range(excess_images,num_images):
        if (time_axis is None):
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, c=channel_axis)
        else:
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis, c=channel_axis)
        j = j + 1

    return data_numpy

def return_affine_xform(path_to_xml,r_idx,y_idx,z_idx,total_z_pos):

    """
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
    :return data_numpy: NDarray
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

def time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")