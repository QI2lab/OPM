#!/usr/bin/env python

'''
OPM stage control
initial attempt at using hooks

Shepherd 10/20
'''

# imports
from pycromanager import Bridge, Acquisition
from pathlib import Path
import numpy as np
import time


def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand') # might be able to remove

    return event
    
def post_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()
    
    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    return event

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # lasers to use
    # 0 -> inactive
    # 1 -> active
    state_405 = 0
    state_488 = 0
    state_561 = 0
    state_635 = 1
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 2
    power_488 = 2
    power_561 = 2
    power_635 = 50    
    power_730 = 0

    # exposure time
    exposure_ms = 5.

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = 1614. #unit: um
    scan_axis_end_um = 1625. #unit: um

    tile_axis_start_um = -265.
    height_axis_start_um = 9500.

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [1024, 0, 256, 2048] #unit: pixels

    # setup file name
    save_directory=Path('E:/20201023/')
    save_name = 'scan_test'

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    bridge = Bridge()
    core = bridge.get_core()

    # set camera into 16bit readout mode
    # give camera time to change modes if necessary
    core.set_property('Camera','ReadoutRate','100MHz 16bit')
    time.sleep(5)

    # set camera into low noise readout mode
    # give camera time to change modes if necessary
    core.set_property('Camera','Gain','2-CMS')
    time.sleep(5)

    # set camera to trigger first mode
    # give camera time to change modes if necessary
    core.set_property('Camera','Trigger Timeout (secs)',60)
    time.sleep(5)

    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Internal Trigger')
    time.sleep(5)

    # set exposure
    core.set_exposure(exposure_ms)

    # get actual framerate, accounting for readout time
    actual_readout_ms = float(core.get_property('Camera','ActualInterval-ms')) #unit: ms

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # scan axis setup
    scan_axis_step_um = 0.2  # unit: um
    scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
    scan_axis_start_mm = scan_axis_start_um / 1000. #unit: mm
    scan_axis_end_mm = scan_axis_end_um / 1000. #unit: mm
    scan_axis_range_um = np.abs(scan_axis_end_um-scan_axis_start_um)  # unit: um
    scan_axis_range_mm = scan_axis_range_um / 1000 #unit: mm
    actual_exposure_s = actual_readout_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / actual_exposure_s,2) #unit: mm/s
    scan_axis_positions = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # Setup PLC card to give start trigger
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    addrOutputBNC3 = 35
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC3)
    core.set_property(plcName, propCellConfig, addrStageSync)

    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set tile axis speed for all moves
    command = 'SPEED Y=.5'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis speed for large move to initial position
    command = 'SPEED X=.5'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # move scan scan stage to initial position
    core.set_xy_position(scan_axis_start_um,tile_axis_start_um)
    core.wait_for_device(xy_stage)
    core.set_position(height_axis_start_um)
    core.wait_for_device(z_stage)

    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set scan axis speed to correct speed for continuous stage scan
    # expects mm/s
    command = 'SPEED X='+str(scan_axis_speed)
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis to true 1D scan with no backlash
    command = '1SCAN X? Y=0 Z=9 F=0'
    core.set_property('TigerCommHub','SerialCommand',command)

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

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    #command = '1SCANV X=0 Y=0 Z=2'
    #core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)
  
    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # set all lasers to off and user defined power
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    print(scan_axis_positions)
    # create events to execute scan
    events = []
    #for t in range(2):
    for x in range(scan_axis_positions):
            evt = { 'axes': {'x':x}}

            events.append(evt)


    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Trigger first')
    time.sleep(1)

    # run acquisition
    with Acquisition(directory=save_directory, name=save_name, post_hardware_hook_fn=post_hook_fn,post_camera_hook_fn=camera_hook_fn, show_display=True, max_multi_res_index=0) as acq:
        acq.acquire(events)

    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Internal Trigger')
    time.sleep(1)

# run
if __name__ == "__main__":
    main()