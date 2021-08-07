#!/usr/bin/env python
'''
Optimize O2-O3 coupling by capturing images of collimated 532 alignment laser injected into system 
using the back of pentaband dichroic with O3 at different positions along the (tilted) optical axis.

Shepherd 07/21
'''

import numpy as np
import time
from skimage.metrics import normalized_root_mse as nrmse
from skimage.registration import phase_cross_correlation

def calculate_focus_metric(image,reference_image):
    """
    calculate focus metric

    :param image: ndarray
        image to test
    :param reference_image: ndarray
        reference image

    :return focus_metric: float
        focus metric
    """

    # calculate focus metric
    #focus_metric = nrmse(reference_image, image)
    _, phase_err, _ = phase_cross_correlation(reference_image, image,upsample_factor=1000)
    nrmse_err = nrmse(reference_image,image)

    focus_metric = phase_err + nrmse_err

    # return focus metric
    return focus_metric
 
def find_best_O3_focus_metric(core,roi_alignment,reference_image,shutter_controller,piezo_channel):
    """
    optimize position of O3 with respect to O2 using TTL control of a Thorlabs K101 controller, Thorlabs PIA25 piezo motor, and Thorlabs 1" translation stage.

    :param core: Core
        pycromanager Core object
    :param n_piezo_step: int
        range of piezo search in steps. Need to limit to some reasonable maximum.
    :param shutter_controller: ArduinoShutter
        object to control Edmund Optics TTL shutter
    :param piezo_channel: APTPiezoInertiaActuator
        channel on Thorlabs KIM101 piezo controller    
    :return found_focus_metric: float
        automatically determined focus metric
    """

    # grab current piezo position
    current_piezo_position = piezo_channel.position_count

    # generate arrays
    n_piezo_steps=10.
    piezo_step_size = 5. #arb. step size on piezo controller
    piezo_positions = np.round(np.arange(current_piezo_position-(piezo_step_size*np.round(n_piezo_steps/2,0)),current_piezo_position+(piezo_step_size*np.round(n_piezo_steps/2,0)),piezo_step_size),0).astype(np.int16)
    focus_metrics = np.zeros(int(n_piezo_steps))

    # open alignment laser shutter
    shutter_controller.openShutter()

    i = 0
    for piezo_position in piezo_positions:

        piezo_channel.move_abs(piezo_position)
        time.sleep(0.5)
        while not(piezo_channel.position_count==piezo_position):
            time.sleep(0.5)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        test_image = test_image[roi_alignment[1]:roi_alignment[1]+roi_alignment[3],roi_alignment[0]:roi_alignment[0]+roi_alignment[2]]
        focus_metrics[i] = calculate_focus_metric(test_image,reference_image)
        i = i+1

    shutter_controller.closeShutter()

    # find best rough focus position
    rough_best_piezo_index = np.argmin(focus_metrics)
    rough_best_piezo_position=piezo_positions[rough_best_piezo_index]

    print('Rough alignment: '+str(rough_best_piezo_position)+'vs actual alignment:'+str(current_piezo_position))

    if np.abs(rough_best_piezo_position-(current_piezo_position+1e-16))/(current_piezo_position+1e-16) < 0.5:
        piezo_channel.move_abs(rough_best_piezo_position)
    else:
        piezo_channel.move_abs(current_piezo_position)
        print('Fallback position.')
    time.sleep(0.2)


    # generate arrays
    del n_piezo_steps, piezo_step_size, piezo_positions, focus_metrics
    n_piezo_steps=10.
    piezo_step_size = 2. #arb. step size on piezo controller
    piezo_positions = np.round(np.arange(rough_best_piezo_position-(piezo_step_size*np.round(n_piezo_steps/2,0)),rough_best_piezo_position+(piezo_step_size*np.round(n_piezo_steps/2,0)),piezo_step_size),0).astype(np.int16)
    focus_metrics = np.zeros(int(n_piezo_steps))

    # open alignment laser shutter
    shutter_controller.openShutter()

    i = 0
    for piezo_position in piezo_positions:

        piezo_channel.move_abs(piezo_position)
        time.sleep(0.5)
        while not(piezo_channel.position_count==piezo_position):
            time.sleep(0.5)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        test_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        test_image = test_image[roi_alignment[1]:roi_alignment[1]+roi_alignment[3],roi_alignment[0]:roi_alignment[0]+roi_alignment[2]]
        focus_metrics[i] = calculate_focus_metric(test_image,reference_image)
        i = i+1

    shutter_controller.closeShutter()
 
    # find best fine focus position
    best_piezo_index = np.argmin(focus_metrics)
    best_piezo_position=piezo_positions[best_piezo_index]
    
    print('Fine alignment: '+str(best_piezo_position)+'vs actual alignment:'+str(current_piezo_position))
    
    if np.abs(best_piezo_position-(current_piezo_position+1e-16))/(current_piezo_position+1e-16) < 0.5:
        piezo_channel.move_abs(best_piezo_position)
    else:
        piezo_channel.move_abs(current_piezo_position)
        print('Fallback position.')
    
    time.sleep(0.2)

    return best_piezo_position

def manage_O3_focus(core,roi_alignment,shutter_controller,piezo_channel,initialize=True,reference_image=None):
    """
    helper function to manage autofocus of O3 with respect to O2

    :param initialize: boolean
        calculate focus metric without running optimization
    :param reference_focus_metric: float
        optional focus metric defined by user during manual alignment
    :param tolerance: float
        error tolerance

    :return found_focus_metric: float
        automatically determined focus metric
    :return success: boolean
        found focus metric is within tolerance of reference focus metric
    """

    # TO DO
    # should n_sweep be increased if function fails to optimize focus within tolerance?
    # cleaner logic for tolerance check
    # include delay and repeat anyways to make sure focus is stable?

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

    if (initialize):
        shutter_controller.openShutter()
        core.snap_image()
        tagged_image = core.get_tagged_image()
        shutter_controller.closeShutter()
        reference_image = np.reshape(tagged_image.pix,newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
        reference_image = reference_image[roi_alignment[1]:roi_alignment[1]+roi_alignment[3],roi_alignment[0]:roi_alignment[0]+roi_alignment[2]]
        found_focus_position = piezo_channel.position_count
    else:
        found_focus_position = find_best_O3_focus_metric(core,roi_alignment,reference_image,shutter_controller,piezo_channel)
   

    # put camera back into experimental mode
    core.set_config('Camera-Setup',readout_mode_experiment)
    core.wait_for_config('Camera-Setup',readout_mode_experiment)
    core.set_exposure(exposure_experiment_ms)
    core.snap_image()

    return reference_image, found_focus_position