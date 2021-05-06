"""
Load image data and/or metadata

todo: get rid of this file
"""
import numpy as np
import tifffile
import pycromanager

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
