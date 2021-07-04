#!/usr/bin/env python
'''
Optimize O2-O3 coupling by capturing images of collimated 532 alignment laser injected into system 
using the back of pentaband dichroic with O3 at different positions along the (tilted) optical axis.

Shepherd 07/21
'''

from pycromanager import Bridge
import PyDAQmx
import numpy as np
import time
from scipy.interpolate import InterpolatedUnivariateSpline

def calculate_focus_metric(image):
    """
    calculate focus metric

    :param image: ndarray
        image to calculate focus metric on

    :return focus_metric: float
        focus metric
    """

    # calculate focus metric


    # return focus metric
    return focus_metric
 
def find_best_O3_focus_metric(core,n_piezo_steps=20,shutter_DO_line=5,backward_DO_line=6,forward_DO_line=7):
    """
    optimize position of O3 with respect to O2 using TTL control of a Thorlabs K101 controller, , and 1" translation stage.

    :param core: Core
        pycromanager Core object
    :param n_piezo_step: int
        range of piezo search in steps. Need to limit to some reasonable maximum.
    :param shutter_DO_line: int
        DO line on DAQ connected to alignment laser shutter
    :param backward_DO_line: int
        DO line on DAQ connected to piezo controller input that is set for a backward move on TTL
    :param forward_DO_line: int
        DO line on DAQ connected to piezo controller input that is set for a forward move on TTL

    :return found_focus_metric: float
        automatically determined focus metric
    """

    # TO DO
    # what is best practice to generate a square TTL for piezo?
    # need one for backward pin and one for forward pin on piezo controller
    # how to open shutter and keep open while triggering piezo?
    # write check for N_max so that Snouty doesn't step too far

    # step backward N/2 + 1 steps
    step = 0
    while step <= (n_piezo_steps//2):

        # step backward

        step = step + 1

    # create empty focus metric array
    focus_metrics = np.zeros([n_piezo_steps],dtype=np.float64)

    # step forward N steps, record image and calculate focus metric at each step
    step = 0
    while (step < n_piezo_steps):

        # step forward

        # grab image and calculate metric
        image = core.snap_image()
        focus_metrics[step]=calculate_focus_metric(image)

        step = step + 1

    # interpolate curve and find best focus position
    x_axis = range(n_piezo_steps)
    f = InterpolatedUnivariateSpline(x_axis, focus_metrics, k=4)
    cr_pts = f.derivative().roots()
    cr_pts = np.append(cr_pts, (x_axis[0], x_axis[-1]))  # also check the endpoints of the interval
    cr_vals = f(cr_pts)
    max_index = np.argmax(cr_vals)
    best_piezo_steps = np.ceil(cr_pts[max_index]).astype(np.uint16)

    # move piezo controller to recorded position by stepping back and then forward

    # capture focus metric at this position
    image = core.snap_image()
    found_focus_metric=calculate_focus_metric(image)

    return found_focus_metric

def manage_O3_focus(initialize=False,reference_focus_metric=None,tolerance=0.1):
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

    bridge=Bridge()
    core = bridge.get_core()

    if (initialize):
        image = core.snap_image()
        found_focus_metric = calculate_focus_metric(image)
        success = True
    else:
        found_focus_metric = find_best_O3_focus_metric(core=core)
        n_trials = 0
        max_trials = 10
        while (found_focus_metric > (1+tolerance)*reference_focus_metric) or (found_focus_metric < (1-tolerance)*reference_focus_metric) or (n_trials >= max_trials):
            time.sleep(10)
            found_focus_metric = find_best_O3_focus_metric(core=core)
            n_trials=n_trials+1

        if n_trials >= max_trials:
            success=False
        else:
            success=True
    
    return found_focus_metric, success