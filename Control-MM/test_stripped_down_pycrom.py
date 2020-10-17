#!/usr/bin/env python

'''
OPM stage control
Stripped down example

Shepherd 10/20
'''

# imports
from pycromanager import Bridge, Acquisition
import numpy as np
from pathlib import Path
import time
from functools import partial

def hook_fn(event,bridge,event_queue):
    
    core=bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)

    return event


def setup_scan_fn(scan_array,event,bridge,event_queue):

    scan_axis_start_mm=scan_array[0]
    scan_axis_end_mm=scan_array[1]

    core=bridge.get_core()

     # set scan axis to true 1D scan with no backlash
    command = '1SCAN X? Y=0 Z=9 F=0'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set range for scan axis
    # expects mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=50'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)
        
    # turn off repeated commands to Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')
    answer = core.get_property('TigerCommHub','SerialCommand')

    return event

def main():

    bridge = Bridge()
    core = bridge.get_core()

    # FOV parameters
    ROI = [0, 1024, 2048, 256] #unit: pixels

    # camera exposure
    exposure_ms = 10 #unit: ms

    # crop FOV
    core.set_roi(*ROI)

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_asi = 280000 #unit: 1/10 um
    scan_axis_end_asi = 295000 #unit: 1/10 um

    save_directory = Path('F:/20201016/')

    # set exposure
    core.set_exposure(exposure_ms)

    # set camera to longer timeout
    core.set_property('Camera','Trigger Timeout (secs)',60)
    time.sleep(5)

    # get actual readout time
    actual_readout_ns = float(core.get_property('Camera','Timing-ExposureTimeNs')) + float(core.get_property('Camera','Timing-ReadoutTimeNs')) #unit: ns
    actual_readout_ms = actual_readout_ns/1000000. #unit: ms

    # scan axis setup
    scan_axis_step_um = 0.2  # unit: um
    scan_axis_step_mm = scan_axis_step_um / 1000.
    scan_axis_start_mm = scan_axis_start_asi / 10000.
    scan_axis_end_mm = scan_axis_end_asi / 10000.
    scan_axis_range_asi = np.abs(scan_axis_end_asi-scan_axis_start_asi)  # unit: 1/10 um
    scan_axis_range_mm = scan_axis_range_asi / 10000 #unit: mm
    actual_readout_s = actual_readout_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / actual_readout_s,2) #unit: mm/s
    number_of_images = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)

    # Setup PLC card to give start trigger
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    addrOutputBNC3 = 35
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC3)
    core.set_property(plcName, propCellConfig, addrStageSync)

    # set scan axis speed to correct speed for continuous stage scan
    # expects mm/s
    command = 'SPEED X='+str(scan_axis_speed)
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set camera to trigger first mode
    core.set_property('Camera','TriggerMode','Trigger first')
    time.sleep(5)

    # setup partial function for stage scan
    scan_array=[scan_axis_start_mm,scan_axis_end_mm]
    setup_scan = partial(setup_scan_fn,scan_array)

    # setup file name
    save_name = 'scan_'+'y_'+str(0).zfill(4)+'x_'+str(0).zfill(4)+'c_'+str(0).zfill(2)

    # create events to hold all of the scan axis images during constant speed stage scan
    # we call this 'z' here, even though it is actually oblique images acquired by moving scan axis (x) in our system
    events = []
    for z in range(number_of_images):
        events.append({'axes': {'z': z}})

    # run acquisition
    # TO DO: properly handle an error here if camera driver fails to return expected number of images.
    with Acquisition(directory=save_directory, name=save_name, pre_hardware_hook_fn=setup_scan, post_camera_hook_fn=hook_fn, show_display=False, max_multi_res_index=0) as acq:
        acq.acquire(events)

# run
if __name__ == "__main__":
    main()