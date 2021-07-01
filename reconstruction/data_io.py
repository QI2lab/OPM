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
        elif vals[ii] == "False":
            vals[ii] = False
        elif vals[ii] == "True":
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
    :param data_dict: dictionary of metadata entries
    :param save_path:
    :return:
    """
    pd.DataFrame([data_dict]).to_csv(save_path)

def return_data_numpy(dataset, channel_axis, num_images,y_pixels,x_pixels):
    """
    :param dataset: pycromanager dataset object
    :param channel_axis: integer channel index
    :param axis_range: integer array for images to return from dataset
    :param y_pixels: integer for y pixel size
    :param x_pixels: integer for x pixel size
    :return data_numpy: 3D numpy array of requested data
    """

    data_numpy = np.empty([num_images,y_pixels,x_pixels]).astype(np.uint16)

    for i in range(num_images):
        if (channel_axis is None):
            data_numpy[i,:,:] = dataset.read_image(z=i)
        else:
            data_numpy[i,:,:] = dataset.read_image(z=i, c=channel_axis)

    return data_numpy