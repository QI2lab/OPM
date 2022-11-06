#!/usr/bin/env python
'''
Optimize O2-O3 coupling by capturing images of collimated 532 alignment laser injected into system 
using the back of pentaband dichroic with O3 at different positions along the (tilted) optical axis.

Shepherd 11/2022
'''

import numpy as np

def calculate_focus_metric(image):
    """
    calculate focus metric

    :param image: ndarray
        image to test

    :return focus_metric: float
        focus metric
    """

    # calculate focus metric
    focus_metric = np.max(image)

    # return focus metric
    return focus_metric
 
def find_best_O3_focus_metric(core,roi_alignment,shutter_controller,O3_stage_name,verbose=False):
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
    exp_zstage_pos = core.get_position()
    exp_zstage_name = core.get_focus_device()

    # set MM focus stage to O3 piezo stage
    core.set_focus_device(O3_stage_name)
    core.wait_for_device(O3_stage_name)

    # grab O3 focus stage position
    O3_stage_pos_start = core.get_position()
    core.wait_for_device(O3_stage_name)

    # generate arrays
    n_O3_stage_steps=16.
    O3_stage_step_size = .25 #arb. step size on piezo controller
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
        test_image = test_image[roi_alignment[1]:roi_alignment[1]+roi_alignment[3],roi_alignment[0]:roi_alignment[0]+roi_alignment[2]]
        focus_metrics[i] = calculate_focus_metric(test_image)
        if verbose: print('Current position: '+str(O3_stage_pos)+'; Focus metric:'+str(focus_metrics[i]))
        i = i+1

    shutter_controller.closeShutter()

    # find best rough focus position
    rough_best_O3_stage_index = np.argmin(focus_metrics)
    rough_best_O3_stage_pos=O3_stage_positions[rough_best_O3_stage_index]

    if verbose: print('Rough align position: '+str(rough_best_O3_stage_pos)+'vs starting align position:'+str(O3_stage_pos_start))

    if np.abs(rough_best_O3_stage_pos-O3_stage_pos_start) < 2.:
        core.set_position(rough_best_O3_stage_pos)
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
        O3_stage_step_size = .1 #arb. step size on piezo controller
        O3_stage_positions = np.round(np.arange(rough_best_O3_stage_pos-(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),rough_best_O3_stage_pos+(O3_stage_step_size*np.round(n_O3_stage_steps/2,0)),O3_stage_step_size),2).astype(np.float)
        focus_metrics = np.zeros(O3_stage_positions.shape[0])
        if verbose: print('Starting fine alignment.')

        # open alignment laser shutter
        shutter_controller.openShutter()

        i = 0
        for O3_stage_pos in O3_stage_positions:

            core.set_position(O3_stage_pos)
            core.wait_for_device(O3_stage_name)
            core.snap_image()
            tagged_image = core.get_tagged_image()
            test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
            test_image = test_image[roi_alignment[1]:roi_alignment[1]+roi_alignment[3],roi_alignment[0]:roi_alignment[0]+roi_alignment[2]]
            focus_metrics[i] = calculate_focus_metric(test_image)
            if verbose: print('Current position: '+str(O3_stage_pos)+'; Focus metric:'+str(focus_metrics[i]))
            i = i+1

        shutter_controller.closeShutter()
    
        # find best fine focus position
        fine_best_O3_stage_index = np.argmin(focus_metrics)
        fine_best_O3_stage_pos=O3_stage_positions[fine_best_O3_stage_index]
        
        print('Fine align position: '+str(fine_best_O3_stage_pos)+'vs starting align position:'+str(rough_best_O3_stage_pos))
        
        if np.abs(fine_best_O3_stage_pos-rough_best_O3_stage_pos) < 0.25:
            core.set_position(fine_best_O3_stage_pos)
            best_03_stage_pos = fine_best_O3_stage_pos
        else:
            core.set_position(rough_best_O3_stage_pos)
            core.wait_for_device(O3_stage_name)
            if verbose: print('Fine focus failed to find better position.')
            best_03_stage_pos = O3_stage_pos_start
            perform_fine = False
        
    # set focus device back to MM experiment focus stage
    core.set_focus_device(exp_zstage_name)
    core.wait_for_device(exp_zstage_name)
    core.set_position(exp_zstage_pos)
    core.wait_for_device(exp_zstage_name)

    return best_03_stage_pos

def manage_O3_focus(core,roi_alignment,shutter_controller,O3_stage_name,verbose=False):
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

    # set exposure to 10 ms
    core.set_exposure(10)

    updated_O3_stage_position = find_best_O3_focus_metric(core,roi_alignment,shutter_controller,O3_stage_name,verbose)
   
    # put camera back into operational readout mode
    core.set_config('Camera-Setup',readout_mode_experiment)
    core.wait_for_config('Camera-Setup',readout_mode_experiment)
    core.set_exposure(exposure_experiment_ms)
    core.snap_image()

    return updated_O3_stage_position