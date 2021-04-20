"""
Load image data and/or metadata
"""
import os
import re
import numpy as np
import tifffile
import pycromanager

def read_metadata(fname):
    """
    Read data from CSV file consisting of one line giving titles, and the other giving values
    :param fname:
    :return:
    """
    scan_data_raw_lines = []

    with open(fname, "r") as f:
        for line in f:
            scan_data_raw_lines.append(line.replace("\n", ""))

    titles = scan_data_raw_lines[0].split(",")
    vals = scan_data_raw_lines[1].split(",")
    for ii in range(len(vals)):
        if re.fullmatch("\d+", vals[ii]):
            vals[ii] = int(vals[ii])
        elif re.fullmatch("\d+.\d+", vals[ii]):
            vals[ii] = float(vals[ii])
        elif vals[ii] == "False":
            vals[ii] = False
        elif vals[ii] == "True":
            vals[ii] = True
        else:
            # otherwise, leave as string
            pass

    metadata = {}
    for t, v in zip(titles, vals):
        metadata[t] = v

    return metadata

def load_volume(fnames, vol_index, imgs_per_vol, chunk_index=0, imgs_per_chunk=100, n_chunk_overlap=3, mode="hcimage"):
    """

    :param fnames: if mode is "hcimage" will be a list of all image file names, each storing an individual image.
    :param size_xy:
    :param vol_index:
    :param imgs_per_vol:
    :param chunk_index:
    :param imgs_per_chunk:
    :param n_chunk_overlap:
    :param mode:
    :return:
    """
    if mode == "hcimage":
        # indices of planes in volume
        img_start = int(np.max([chunk_index * imgs_per_chunk - n_chunk_overlap, 0]))
        img_end = int(np.min([img_start + imgs_per_chunk, imgs_per_vol]))
        if img_start >= img_end - n_chunk_overlap:
            img_start = img_end

        inds = (img_start, img_end)

        # load images
        imgs = []
        for kk in range(img_start, img_end):
            imgs.append(tifffile.imread(fnames[vol_index * imgs_per_vol + kk]))
        imgs = np.asarray(imgs)

    elif mode == "ndtiff":
        # clearly don't want to load dataset everytime...but for the moment to allow me to switch modes easily...
        # dir_one_up, _ = os.path.split(fnames[0])
        # dir_two_up, _ = os.path.split(dir_one_up)
        ds = pycromanager.Dataset(fnames)

        # indices of planes in volume
        img_start = int(np.max([chunk_index * imgs_per_chunk - n_chunk_overlap, 0]))
        img_end = int(np.min([img_start + imgs_per_chunk, imgs_per_vol]))
        if img_start >= img_end - n_chunk_overlap:
            img_start = img_end

        inds = (img_start, img_end)

        # load images
        imgs = []
        for kk in range(img_start, img_end):
            imgs.append(ds.read_image(z=kk, t=vol_index, c=0))
        imgs = np.asarray(imgs)

    else:
        raise ValueError("mode must be 'hcimage' or 'ndtiff'")


    return imgs, inds
