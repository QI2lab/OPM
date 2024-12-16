#!/usr/bin/env python

import re
from npy2bdv import BdvEditor
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pathlib
from pycromanager import Dataset
import sys

# Define exception for missing keys in fluidics program
class MissingMuxKey(Exception):
    """Custom exception for invalid MUX values."""
    def __init__(self, value, mux_name):
        super().__init__(f"Fluidics program source '{value}' is not a valid key in {mux_name} configuration.")
        self.value = value
        self.mux_name = mux_name


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

def read_fluidics_program(program_path: str = None,
                          flow_controller=None,
                          verbose: bool = False):
    """
    Load fluidics program saved as work book (.xlsx)
    Each 'sheet' or 'tab' corresponds to a round.

    Args:
        program_path (str): Path to the fluidics .xlsx program file. Defaults to None.
        flow_controller (Class):
        verbose (bool, optional): Print progress statements. Defaults to False
    """
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(program_path)
    except Exception as e:
        print(f"Error loading program workbook, path:{program_path}\n", f"Exception: {e}")
        sys.exit()

    # Function to check if key is in either mux1 or mux2
    def check_mux_key(key, mux1_config, mux2_config):
        if key is not None and key not in mux1_config and key not in mux2_config:
            raise MissingMuxKey(key, 'mux1 or mux2')

    # Create a list to store each sheet as a DataFrame
    program_list = []

    # Iterate through each sheet name and read it into a DataFrame, then append to the list
    for sheet_name in excel_data.sheet_names:
        df = pd.read_excel(program_path, sheet_name=sheet_name, engine='openpyxl')

        df = df[["round", "type", "pause", "source", "prime_buffer", "volume", "rate"]]
        df.dropna(axis=0, how='any', inplace=True)
        df["round"] = df["round"].astype(int)
        df["type"] = df["type"].astype(str)
        df["source"] = df["source"].astype(str).replace('none', None)
        df["prime_buffer"] = df["prime_buffer"].astype(str).replace('none', None)
        df["volume"] = df["volume"].astype(float)
        df["rate"] = df["rate"].astype(float)
        df["pause"] = df["pause"].astype(float)

        # Verify program source and prime_buffer exist in the ElveFlow MUX config keys
        try:
            # Check 'source' column
            df["source"].apply(lambda x: check_mux_key(x, flow_controller.config["mux1"], flow_controller.config["mux2"]))

            # Check 'prime_buffer' column
            df["prime_buffer"].apply(lambda x: check_mux_key(x, flow_controller.config["mux1"], flow_controller.config["mux2"]))

        except MissingMuxKey as e:
            print(f"{e}\n", f"Error occurred loading round {df['round'].iloc[0]}, go back and set up ElveFlow Configuration")
            sys.exit()

        program_list.append(df)

    if verbose:
        # Now program_list contains each sheet as a DataFrame with validated MUX values
        for i, sheet in enumerate(program_list):
            print(f"{excel_data.sheet_names[i]}")
            print(sheet.head())  # Print the first few rows of each sheet

    return program_list

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

def return_data_dask(data_path, axes_order):
    """
    :param dataset: pycromanager dataset object
    :param axes_order: order to load axes in

    :return data_dask: ND dask array of requested data
    """

    return Dataset(data_path).as_array(axes=axes_order)

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

def append_index_filepath(filepath):
    """
    Append a number to a file path if the file already exists,
    the number increases as long as there is a file that exists.
    """
    if isinstance(filepath, (pathlib.WindowsPath, pathlib.PosixPath)):
        to_pathlib = True
        filepath = str(filepath.as_posix())
    else:
        to_pathlib = False

    i = 1
    while os.path.exists(filepath):
        filepath = "".join(filepath.split('.')[:-1]) + f"-{i}." + filepath.split('.')[-1]
        i += 1
    if to_pathlib:
        filepath = pathlib.Path(filepath)
    return filepath