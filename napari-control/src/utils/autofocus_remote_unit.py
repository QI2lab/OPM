#!/usr/bin/env python
'''
Optimize O2-O3 coupling by capturing images of collimated 532 alignment laser injected into system 
using the back of pentaband dichroic with O3 at different positions along the (tilted) optical axis.

Shepherd 11/2022
'''

import numpy as np
from scipy import ndimage
from pymmcore_plus import CMMCorePlus

def calculate_focus_metric(image: np.ndarray):
    """
    calculate focus metric

    :param image: ndarray
        image to test

    :return focus_metric: float
        focus metric
    """

    # calculate focus metric
    image[image>2**16-10]=0
    image[image<100]=0
    kernel = [[0,1,0],[1,1,1],[0,1,0]]
    focus_metric = np.max(ndimage.minimum_filter(image,footprint=kernel))

    # return focus metric
    return focus_metric
 
def find_best_O3_focus_metric(mmc: CMMCorePlus,shutter_controller,O3_stage_name,verbose=False):
    """
    optimize position of O3 with respect to O2 using TTL control of a Thorlabs K101 controller, Thorlabs PIA25 piezo motor, and Thorlabs 1" translation stage.

    :param mmc: mmc
        PyMMCorePlus mmc object
    :param shutter_controller: PicardShutter
        Picard shutter controller
    :param O3_piezo_stage: str
        name of O3 piezo stage in MM config
    :param verbose: bool
        print information on autofocus

    :return found_focus_metric: float
        automatically determined focus metric
    """
    
    # grab position and name of current MM focus stage
    exp_zstage_pos = np.round(mmc.getPosition(),2)
    exp_zstage_name = mmc.getFocusDevice()
    if verbose: print(f'Current z-stage: {exp_zstage_name} with position {exp_zstage_pos}')

    # set MM focus stage to O3 piezo stage
    mmc.setFocusDevice(O3_stage_name)
    mmc.waitForDevice(O3_stage_name)

    # grab O3 focus stage position
    O3_stage_pos_start = np.round(mmc.getPosition(),2)
    mmc.waitForDevice(O3_stage_name)
    if verbose: print(f'O3 z-stage: {O3_stage_name} with position {O3_stage_pos_start}')

    # generate arrays
    n_O3_stage_steps=20.
    O3_stage_step_size = .25
    O3_stage_positions = np.round(np.arange(O3_stage_pos_start-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_pos_start+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float)
    focus_metrics = np.zeros(O3_stage_positions.shape[0])
    if verbose: print('Starting rough alignment.')

    # open alignment laser shutter
    shutter_controller.openShutter()

    i = 0
    for O3_stage_pos in O3_stage_positions:

        mmc.setPosition(O3_stage_pos)
        mmc.waitForDevice(O3_stage_name)
        mmc.snapImage()
        test_image = mmc.getImage()
        #test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        focus_metrics[i] = calculate_focus_metric(test_image)
        if verbose: print(f'Current position: {O3_stage_pos}; Focus metric: {focus_metrics[i]}')
        i = i+1

    # find best rough focus position
    rough_best_O3_stage_index = np.argmax(focus_metrics)
    rough_best_O3_stage_pos=O3_stage_positions[rough_best_O3_stage_index]

    if verbose: print(f'Rough align position: {rough_best_O3_stage_pos} vs starting position: {O3_stage_pos_start}')

    if np.abs(rough_best_O3_stage_pos-O3_stage_pos_start) < 2.:
        mmc.setPosition(rough_best_O3_stage_pos)
        mmc.waitForDevice(O3_stage_name)
        perform_fine = True
    else:
        mmc.setPosition(O3_stage_pos_start)
        mmc.waitForDevice(O3_stage_name)
        if verbose: print('Rough focus failed to find better position.')
        best_03_stage_pos = O3_stage_pos_start
        perform_fine = False
    
    # generate arrays
    del n_O3_stage_steps, O3_stage_step_size, O3_stage_positions, focus_metrics
    
    if perform_fine:
        n_O3_stage_steps=10.
        O3_stage_step_size = .1
        O3_stage_positions = np.round(np.arange(rough_best_O3_stage_pos-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),rough_best_O3_stage_pos+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float)
        focus_metrics = np.zeros(O3_stage_positions.shape[0])
        if verbose: print('Starting fine alignment.')

        i = 0
        for O3_stage_pos in O3_stage_positions:

            mmc.setPosition(O3_stage_pos)
            mmc.waitForDevice(O3_stage_name)
            mmc.snapImage()
            test_image = mmc.getImage()
            #test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
            focus_metrics[i] = calculate_focus_metric(test_image)
            if verbose: print(f'Current position: {O3_stage_pos}; Focus metric: {focus_metrics[i]}')
            i = i+1
    
        # find best fine focus position
        fine_best_O3_stage_index = np.argmax(focus_metrics)
        fine_best_O3_stage_pos=O3_stage_positions[fine_best_O3_stage_index]
        
        if verbose: print(f'Fine align position: {fine_best_O3_stage_pos} vs starting position: {rough_best_O3_stage_pos}')
        
        if np.abs(fine_best_O3_stage_pos-rough_best_O3_stage_pos) < .5:
            mmc.setPosition(fine_best_O3_stage_pos)
            mmc.waitForDevice(O3_stage_name)
            best_03_stage_pos = fine_best_O3_stage_pos
        else:
            mmc.setPosition(rough_best_O3_stage_pos)
            mmc.waitForDevice(O3_stage_name)
            if verbose: print('Fine focus failed to find better position.')
            best_03_stage_pos = O3_stage_pos_start
            perform_fine = False

    shutter_controller.closeShutter()
        
    # set focus device back to MM experiment focus stage
    mmc.setFocusDevice(exp_zstage_name)
    mmc.waitForDevice(exp_zstage_name)
    mmc.setPosition(exp_zstage_pos)
    mmc.waitForDevice(exp_zstage_name)

    return best_03_stage_pos

def manage_O3_focus(mmc: CMMCorePlus,shutter_controller,O3_stage_name,verbose=False):
    """
    helper function to manage autofocus of O3 with respect to O2

    :param mmc: mmc
        PyMMCorePlus mmc object
    :param shutter_controller: PicardShutter
        Picard shutter controller
    :param O3_piezo_stage: str
        String for the O3 piezo stage
    :param verbose: bool
        print information on autofocus

    :return updated_O3_stage_position: float
        automatically determined focus metric. Defaults to original position if not found
    """

    # get exposure for experiment
    exposure_experiment_ms = mmc.getExposure()

    # set camera to fast readout mode
    readout_mode_experiment = mmc.getCurrentConfig('Camera-Setup')
    mmc.setConfig('Camera-Setup','ScanMode3')
    mmc.waitForConfig('Camera-Setup','ScanMode3')
    
    # set camera to internal control
    mmc.setConfig('Camera-TriggerSource','INTERNAL')
    mmc.waitForConfig('Camera-TriggerSource','INTERNAL')

    # set exposure to 10 ms
    mmc.setExposure(5)

    updated_O3_stage_position = find_best_O3_focus_metric(mmc,shutter_controller,O3_stage_name,verbose)
   
    # put camera back into operational readout mode
    mmc.setConfig('Camera-Setup',readout_mode_experiment)
    mmc.waitForConfig('Camera-Setup',readout_mode_experiment)
    mmc.setExposure(exposure_experiment_ms)
    mmc.snapImage()

    return updated_O3_stage_position