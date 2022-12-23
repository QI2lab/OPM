#!/usr/bin/env python
'''
Optimize O2-O3 coupling by capturing images of collimated 532 alignment laser injected into system 
using the back of pentaband dichroic with O3 at different positions along the (tilted) optical axis.

Shepherd 11/2022
'''

import numpy as np
from scipy import ndimage
import time

def apply_O3_focus_offset(core,O3_stage_name,current_O3_focus,O3_stage_offset,verbose=False):
    """
    :param core: Core
        pycromanager Core object
    :param O3_piezo_stage: str
        name of O3 piezo stage in MM config
    :param current_03_focus: float
        current O3 position
    :param O3_stage_offset: float
        offset in microns
    :param verbose: bool
        print information on autofocus
    
    :returns O3_stage_pos:
        offset O3 stage position
    """

    # grab position and name of current MM focus stage
    exp_zstage_pos = np.round(core.get_position(),2)
    exp_zstage_name = core.get_focus_device()
    if verbose: print(f'Current z-stage: {exp_zstage_name} with position {exp_zstage_pos}')

    # set MM focus stage to O3 piezo stage
    core.set_focus_device(O3_stage_name)
    core.wait_for_device(O3_stage_name)

    # grab O3 focus stage position

    O3_stage_pos = current_O3_focus + O3_stage_offset

    core.set_position(np.round(O3_stage_pos,2))
    core.wait_for_device(O3_stage_name)
    time.sleep(.1)

    core.set_focus_device(exp_zstage_name)
    exp_zstage_pos = np.round(core.get_position(),2)
    core.wait_for_device(exp_zstage_name)

    return O3_stage_pos


def calculate_focus_metric(image):
    """
    calculate focus metric

    :param image: ndarray
        image to test

    :return focus_metric: float
        focus metric
    """

    # calculate focus metric
    image[image>60000]=0
    image[image<100]=0
    kernel = [[0,1,0],[1,1,1],[0,1,0]]
    focus_metric = np.max(ndimage.minimum_filter(image,footprint=kernel))

    # return focus metric
    return focus_metric
 
def find_best_O3_focus_metric(core,shutter_controller,O3_stage_name,verbose=False):
    """
    optimize position of O3 with respect to O2 using TTL control of a Thorlabs K101 controller, Thorlabs PIA25 piezo motor, and Thorlabs 1" translation stage.

    :param core: Core
        pycromanager Core object
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
    exp_zstage_pos = np.round(core.get_position(),2)
    exp_zstage_name = core.get_focus_device()
    if verbose: print(f'Current z-stage: {exp_zstage_name} with position {exp_zstage_pos}')

    # set MM focus stage to O3 piezo stage
    core.set_focus_device(O3_stage_name)
    core.wait_for_device(O3_stage_name)

    # grab O3 focus stage position
    O3_stage_pos_start = np.round(core.get_position(),2)
    core.wait_for_device(O3_stage_name)
    if verbose: print(f'O3 z-stage: {O3_stage_name} with position {O3_stage_pos_start}')

    # generate arrays
    n_O3_stage_steps=20.
    O3_stage_step_size = .1
    O3_stage_positions = np.round(np.arange(O3_stage_pos_start-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_pos_start+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float)
    focus_metrics = np.zeros(O3_stage_positions.shape[0])
    if verbose: print('Starting rough alignment.')

    # open alignment laser shutter
    shutter_controller.openShutter()

    i = 0
    for O3_stage_pos in O3_stage_positions:

        core.set_position(O3_stage_pos)
        core.wait_for_device(O3_stage_name)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        focus_metrics[i] = calculate_focus_metric(test_image)
        if verbose: print(f'Current position: {O3_stage_pos}; Focus metric: {focus_metrics[i]}')
        i = i+1

    # find best rough focus position
    rough_best_O3_stage_index = np.argmax(focus_metrics)
    rough_best_O3_stage_pos=O3_stage_positions[rough_best_O3_stage_index]

    if verbose: print(f'Rough align position: {rough_best_O3_stage_pos} vs starting position: {O3_stage_pos_start}')

    if np.abs(rough_best_O3_stage_pos-O3_stage_pos_start) < 1.:
        core.set_position(rough_best_O3_stage_pos)
        core.wait_for_device(O3_stage_name)
        perform_fine = True
    else:
        core.set_position(O3_stage_pos_start)
        core.wait_for_device(O3_stage_name)
        if verbose: print('Rough focus failed to find better position.')
        best_03_stage_pos = O3_stage_pos_start
        perform_fine = False
    
    # generate arrays
    del n_O3_stage_steps, O3_stage_step_size, O3_stage_positions, focus_metrics
    
    if perform_fine:
        n_O3_stage_steps=10.
        O3_stage_step_size = .05
        O3_stage_positions = np.round(np.arange(rough_best_O3_stage_pos-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),rough_best_O3_stage_pos+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float)
        focus_metrics = np.zeros(O3_stage_positions.shape[0])
        if verbose: print('Starting fine alignment.')

        i = 0
        for O3_stage_pos in O3_stage_positions:

            core.set_position(O3_stage_pos)
            core.wait_for_device(O3_stage_name)
            core.snap_image()
            tagged_image = core.get_tagged_image()
            test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
            focus_metrics[i] = calculate_focus_metric(test_image)
            if verbose: print(f'Current position: {O3_stage_pos}; Focus metric: {focus_metrics[i]}')
            i = i+1
    
        # find best fine focus position
        fine_best_O3_stage_index = np.argmax(focus_metrics)
        fine_best_O3_stage_pos=O3_stage_positions[fine_best_O3_stage_index]
        
        if verbose: print(f'Fine align position: {fine_best_O3_stage_pos} vs starting position: {rough_best_O3_stage_pos}')
        
        if np.abs(fine_best_O3_stage_pos-rough_best_O3_stage_pos) < .2:
            core.set_position(fine_best_O3_stage_pos)
            core.wait_for_device(O3_stage_name)
            best_03_stage_pos = fine_best_O3_stage_pos
        else:
            core.set_position(rough_best_O3_stage_pos)
            core.wait_for_device(O3_stage_name)
            if verbose: print('Fine focus failed to find better position.')
            best_03_stage_pos = O3_stage_pos_start
            perform_fine = False

    shutter_controller.closeShutter()
        
    # set focus device back to MM experiment focus stage
    core.set_focus_device(exp_zstage_name)
    core.wait_for_device(exp_zstage_name)
    core.set_position(exp_zstage_pos)
    core.wait_for_device(exp_zstage_name)

    return best_03_stage_pos

def manage_O3_focus(core,shutter_controller,O3_stage_name,verbose=False):
    """
    helper function to manage autofocus of O3 with respect to O2

    :param core: Core
        Pycromanager Core object
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
    exposure_experiment_ms = core.get_exposure()

    # set camera to fast readout mode
    readout_mode_experiment = core.get_current_config('Camera-Setup')
    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')
    
    # set camera to internal control
    core.set_config('Camera-TriggerSource','INTERNAL')
    core.wait_for_config('Camera-TriggerSource','INTERNAL')

    # set exposure to 5 ms
    core.set_exposure(5)

    updated_O3_stage_position = find_best_O3_focus_metric(core,shutter_controller,O3_stage_name,verbose)
   
    # put camera back into operational readout mode
    core.set_config('Camera-Setup',readout_mode_experiment)
    core.wait_for_config('Camera-Setup',readout_mode_experiment)
    core.set_exposure(exposure_experiment_ms)
    core.snap_image()

    return updated_O3_stage_position