import re
import pandas as pd
import numpy as np

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

def write_metadata(data_dict, save_path):
    """
    Write metadata dictionary as CSV file

    :param data_dict: dict
        metadata entries
    :param save_path: Path
        where to save file
    :return: None
    """
    pd.DataFrame([data_dict]).to_csv(save_path)

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

def return_data_numpy(dataset, channel_axis,scan_positions,excess_stage_positions,y_pixels,x_pixels):
    """
    Return requested data from pycromanager Dataset as numpy array

    :param dataset: Dataset
        pycromanager dataset object
    :param channel_axis: int
        channel index
    :param scan_positions: int
        number of scan positions
    :param excess_stage_positions: int
        number of excess position captured to allow stage to come up to speed
    :param y_pixels: int
        y pixel size
    :param x_pixels: int
        x pixel size
    :return data_numpy: ndarray 
        3D numpy array of requested data
    """

    data_numpy = np.empty([(scan_positions-excess_stage_positions),y_pixels,x_pixels]).astype(np.uint16)

    for i in range(excess_stage_positions,scan_positions):
        if (channel_axis is None):
            data_numpy[i,:,:] = dataset.read_image(z=i)
        else:
            data_numpy[i,:,:] = dataset.read_image(z=i, c=channel_axis)

    return data_numpy