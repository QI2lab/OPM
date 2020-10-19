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

    print('post camera hook fn.')
    print(event)

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    return event
    
def post_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()
    
    print('post hardware hook fn.')
    if (('y' in event) and ('z' in event)):
        print(event)

        # allow for repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

        # check to make sure Tiger is not busy
        ready='B'
        while(ready!='N'):
            command = 'STATUS'
            core.set_property('TigerCommHub','SerialCommand',command)
            ready = core.get_property('TigerCommHub','SerialResponse')
            time.sleep(.500)

        # allow for repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

        # setup laser
        # turn off lasers
        core.set_config('Obis-State-405','Off')
        core.wait_for_config('Obis-State-405','Off')
        core.set_config('Obis-State-488','Off')
        core.wait_for_config('Obis-State-488','Off')
        core.set_config('Obis-State-561','Off')
        core.wait_for_config('Obis-State-561','Off')
        core.set_config('Obis-State-637','Off')
        core.wait_for_config('Obis-State-637','Off')
        core.set_config('Obis-State-730','Off')
        core.wait_for_config('Obis-State-730','Off')

        # turn on current channel
        c = event['axes']['c']
        if (c==0):
            core.set_config("Obis-State-405","On")
            core.wait_for_config("Obis-State-405","On")
        elif (c==1):
            core.set_config("Obis-State-488","On")
            core.wait_for_config("Obis-State-488","On")
        elif (c==2):
            core.set_config("Obis-State-561","On")
            core.wait_for_config("Obis-State-561","On")
        elif (c==3):
            core.set_config("Obis-State-637","On")
            core.wait_for_config("Obis-State-637","On")
        elif (c==4):
            core.set_config("Obis-State-730","On")
            core.wait_for_config("Obis-State-730","On")

        # allow for repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

        # check to make sure Tiger is not busy
        ready='B'
        while(ready!='N'):
            command = 'STATUS'
            core.set_property('TigerCommHub','SerialCommand',command)
            ready = core.get_property('TigerCommHub','SerialResponse')
            time.sleep(.500)

        # allow for repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

        # do not return event, so that this setup step is removed from the queue and camera is not called

    else:
        # call before constant speed stage scan, return large list of events
        print(event)

        return event

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # lasers to use
    # 0 -> inactive
    # 1 -> active
    state_405 = 1
    state_488 = 1
    state_561 = 1
    state_635 = 1
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 0
    power_488 = 0
    power_561 = 0
    power_635 = 0
    power_730 = 0

    # exposure time
    exposure_ms = 10.

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = 28000. #unit: um
    scan_axis_end_um = 28100. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = 14600. #unit: um
    tile_axis_end_um = 14650. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = 66160. #unit: um
    height_axis_end_um = 66200. #unit:  um

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [1024, 0, 256, 1024] #unit: pixels

    # setup file name
    save_directory=Path('F:/20201018/')
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

    # crop FOV
    core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # get actual framerate, accounting for readout time
    actual_exposure_ns = float(core.get_property('Camera','Timing-ExposureTimeNs')) #unit: ns
    actual_readout_ns = float(core.get_property('Camera','Timing-ReadoutTimeNs')) #unit: ns
    if (actual_readout_ns>actual_exposure_ns):
        actual_readout_ms = actual_readout_ns/1000000. #unit: ms
    else:
        actual_readout_ms = actual_exposure_ns/1000000. #unit: ms

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

    # tile axis setup
    tile_axis_overlap=0.2 #unit: percentage
    tile_axis_range_um = np.abs(tile_axis_end_um - tile_axis_start_um) #unit: um
    tile_axis_range_mm = tile_axis_range_um / 1000 #unit: mm
    tile_axis_ROI = ROI[3]*pixel_size_um  #unit: um
    tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
    tile_axis_step_mm = tile_axis_step_um / 1000 #unit: mm
    tile_axis_positions = np.rint(tile_axis_range_mm / tile_axis_step_mm).astype(int)

    # height axis setup
    # this is more complicated, since we have an oblique light sheet
    # the height of the scan is the length of the ROI in the tilted direction * sin(tilt angle)
    # however, it may be better to hardcode displacement based on measurements of the light sheet Rayleigh length
    # for now, go with overlap calculation
    height_axis_overlap=0.2 #unit: percentage
    height_axis_range_um = np.abs(height_axis_end_um-height_axis_start_um) #unit: 1/10 um
    height_axis_range_mm = height_axis_range_um / 1000 #unit: mm
    height_axis_ROI = ROI[2]*pixel_size_um*np.sin(30*(np.pi/180.)) #unit: um
    height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
    height_axis_step_mm = height_axis_step_um / 1000  #unit: mm
    height_axis_positions = np.rint(height_axis_range_mm / height_axis_step_mm).astype(int)

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

    # turn on for repeated commands to Tiger
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

    # turn off for repeated commands to Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # move scan scan stage to initial position
    core.set_xy_position(scan_axis_start_um,tile_axis_start_um)
    core.wait_for_device(xy_stage)
    core.set_position(height_axis_start_um)
    core.wait_for_device(z_stage)

    # turn on for repeated commands to Tiger
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
        
    # turn off repeated commands to Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # set all lasers to off and user defined power
    core.set_config('Obis-State-405','Off')
    core.wait_for_config('Obis-State-405','Off')
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])

    core.set_config('Obis-State-488','Off')
    core.wait_for_config('Obis-State-488','Off')
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])

    core.set_config('Obis-State-561','Off')
    core.wait_for_config('Obis-State-561','Off')
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])

    core.set_config('Obis-State-637','Off')
    core.wait_for_config('Obis-State-637','Off')
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])

    core.set_config('Obis-State-730','Off')
    core.wait_for_config('Obis-State-730','Off')
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # create events to execute scan
    events = []
    for y in range(tile_axis_positions):
        for z in range(height_axis_positions):
            for c in range(len(channel_states)):
                for x in range(scan_axis_positions+1):
                        if channel_states[c]==1:
                            if x==0:
                                tile_position_um = tile_axis_start_um+(tile_axis_step_um*y)
                                height_position_um = height_axis_start_um+(height_axis_step_um*z)
                                evt = { 'axes': {'x': x, 'y': y, 'z': z, 'c': c}, 'y': tile_position_um, 'z': height_position_um}
                            else:
                                evt = { 'axes': {'x': x, 'y': y, 'z': z, 'c': c}}
                            events.append(evt)

    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Trigger first')
    time.sleep(1)

    # run acquisition
    with Acquisition(directory=save_directory, name=save_name, post_hardware_hook_fn=post_hook_fn,post_camera_hook_fn=camera_hook_fn, show_display=True, max_multi_res_index=0) as acq:
        acq.acquire(events)

    core.set_config('Obis-State-405','Off')
    core.wait_for_config('Obis-State-405','Off')
    core.set_config('Obis-State-488','Off')
    core.wait_for_config('Obis-State-488','Off')
    core.set_config('Obis-State-561','Off')
    core.wait_for_config('Obis-State-561','Off')
    core.set_config('Obis-State-637','Off')
    core.wait_for_config('Obis-State-637','Off')
    core.set_config('Obis-State-730','Off')
    core.wait_for_config('Obis-State-730','Off')

# run
if __name__ == "__main__":
    main()