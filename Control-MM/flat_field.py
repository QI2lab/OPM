#!/usr/bin/env python

'''
Calculate flat-field using BaSiC algorithm via pyimagej
https://doi.org/10.1038/ncomms14836

The use of pyimagej can likely be improved, but this works for now.

Shepherd 01/21
'''

# imports
import numpy as np
from pathlib import Path
import os
import imagej
import scyjava
from scyjava import jimport
from skimage.io import imread, imsave

def manage_flat_field(output_dir_path,channel_id,z,sub_stack,ij):

    # check if flat-field exists on disk
    channel_flat_field_name = 'flatfield_z'+str(z)+'_c'+str(channel_id)+'.tif'
    flat_field_name = output_dir_path / channel_flat_field_name

    channel_dark_field_name = 'darkfield_z'+str(z)+'_c'+str(channel_id)+'.tif'
    dark_field_name = output_dir_path / channel_dark_field_name

    if flat_field_name.exists ():
        flat_field = imread(flat_field_name)
        dark_field = imread(dark_field_name)
    else:
        print('Calculating flat-field correction using ImageJ and BaSiC plugin.')
        flat_field, dark_field = calculate_flat_field(sub_stack,ij)
        imsave(flat_field_name,flat_field)
        imsave(dark_field_name,dark_field)
    
    print('Performing flat-field correction.')
    corrected_sub_stack = perform_flat_field(flat_field,dark_field,sub_stack)

    return corrected_sub_stack

def calculate_flat_field(sub_stack,ij):
    # convert dataset from numpy -> java
    sub_stack_for_flat_field = sub_stack[np.random.choice(sub_stack.shape[0], 500, replace=False)]
    sub_stack_iterable = ij.op().transform().flatIterableView(ij.py.to_java(sub_stack_for_flat_field.compute()))

    # show image in imagej since BaSiC plugin cannot be run headless
    ij.ui().show(sub_stack_iterable)
    WindowManager = jimport('ij.WindowManager')
    current_image = WindowManager.getCurrentImage()

    # convert virtual stack to real stack and reorder for BaSiC
    macro = """
    rename("active")
    run("Duplicate...", "duplicate")
    selectWindow("active")
    run("Close")
    selectWindow("active-1")
    run("Re-order Hyperstack ...", "channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)]")
    """
    ij.py.run_macro(macro)

    # run BaSiC plugin
    plugin = 'BaSiC '
    args = {
        'processing_stack': 'active-1',
        'flat-field': 'None',
        'dark-field': 'None',
        'shading_estimation': '[Estimate shading profiles]',
        'shading_model': '[Estimate both flat-field and dark-field]',
        'setting_regularisationparametes': 'Automatic',
        'temporal_drift': '[Ignore]',
        'correction_options': '[Compute shading only]',
        'lambda_flat': 0.5,
        'lambda_dark': 0.5
    }
    ij.py.run_plugin(plugin, args)

    # grab flat-field image, convert from java->numpy
    macro2 = """
    selectWindow("active-1")
    run("Close")
    selectWindow("Flat-field:active-1")
    """
    ij.py.run_macro(macro2)
    current_image = WindowManager.getCurrentImage()
    flat_field_ij = ij.py.from_java(current_image)
    flat_field = flat_field_ij.data

    # close flat-field, grab dark-field image, convert from java->numpy
    macro3 = """
    selectWindow("Flat-field:active-1")
    run("Close")
    selectWindow("Dark-field:active-1")
    """
    ij.py.run_macro(macro3)

    current_image = WindowManager.getCurrentImage()
    dark_field_ij = ij.py.from_java(current_image)
    dark_field = dark_field_ij.data

    # close dark-field image
    macro4 = """
    selectWindow("Dark-field:active-1")
    run("Close")
    run("Collect Garbage")
    """
    ij.py.run_macro(macro4)

    del sub_stack_iterable
    del sub_stack

    return flat_field, dark_field

def perform_flat_field(flat_field,dark_field,sub_stack):

    corrected_sub_stack = sub_stack.astype(np.float32) - dark_field
    corrected_sub_stack[corrected_sub_stack<0] = 0 
    corrected_sub_stack = corrected_sub_stack/flat_field

    return corrected_sub_stack.compute()