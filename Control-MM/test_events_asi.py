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
    #answer = core.get_property('TigerCommHub','SerialCommand') # might be able to remove

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
    state_405 = 1
    state_488 = 1
    state_561 = 1
    state_635 = 0
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 50
    power_488 = 70
    power_561 = 50
    power_635 = 0
    power_730 = 0

    # exposure time
    exposure_ms = 5.

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = -1700. #unit: um
    scan_axis_end_um = 700. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = 1300. #unit: um
    tile_axis_end_um = 2300. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = 0.#unit: um
    height_axis_end_um = 20. #unit:  um

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [0, 1024, 1600, 256] #unit: pixels

    # setup file name
    save_directory=Path('E:/20201106/')
    save_name = 'human_lung_test2new'

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    bridge = Bridge()
    core = bridge.get_core()

    # turn off lasers
    core.set_config('Coherent-State','off')
    core.wait_for_config('Coherent-State','off')

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
    core.set_property('Camera','Trigger Timeout (secs)',300)
    time.sleep(5)

    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Internal Trigger')
    time.sleep(5)

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',20000)

    # crop FOV
    #core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # get actual framerate from micromanager properties
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
    scan_axis_positions = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)  #unit: number of positions

    # tile axis setup
    tile_axis_overlap=0.2 #unit: percentage
    tile_axis_range_um = np.abs(tile_axis_end_um - tile_axis_start_um) #unit: um
    tile_axis_range_mm = tile_axis_range_um / 1000 #unit: mm
    tile_axis_ROI = ROI[2]*pixel_size_um  #unit: um
    tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
    tile_axis_step_mm = tile_axis_step_um / 1000 #unit: mm
    tile_axis_positions = np.rint(tile_axis_range_mm / tile_axis_step_mm).astype(int)  #unit: number of positions
    # if tile_axis_positions rounded to zero, make sure we acquire at least one position
    if tile_axis_positions == 0:
        tile_axis_positions=1

    # height axis setup
    # this is more complicated, since we have an oblique light sheet
    # the height of the scan is the length of the ROI in the tilted direction * sin(tilt angle)
    # however, it may be better to hardcode displacement based on measurements of the light sheet Rayleigh length
    # for now, go with overlap calculation
    height_axis_overlap=0.2 #unit: percentage
    height_axis_range_um = np.abs(height_axis_end_um-height_axis_start_um) #unit: um
    height_axis_range_mm = height_axis_range_um / 1000 #unit: mm
    height_axis_ROI = ROI[3]*pixel_size_um*np.sin(30*(np.pi/180.)) #unit: um
    height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
    height_axis_step_mm = height_axis_step_um / 1000  #unit: mm
    height_axis_positions = np.rint(height_axis_range_mm / height_axis_step_mm).astype(int) #unit: number of positions
    # if height_axis_positions rounded to zero, make sure we acquire at least one position
    if height_axis_positions==0:
        height_axis_positions=1

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # Setup PLC card to give start trigger
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    #addrOutputBNC3 = 35
    addrOutputBNC1 = 33
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC1)
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

    # set range and return speed (25% of max) for scan axis
    # expects mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=25'
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

    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # set lasers to user defined power
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    print('Number of X positions: '+str(scan_axis_positions))
    print('Number of Y tiles: '+str(tile_axis_positions))
    print('Number of Z slabs: '+str(height_axis_positions))

    #time.sleep(10)

    for z in range(height_axis_positions):
        # calculate height axis position
        height_position_um = height_axis_start_um+(height_axis_step_um*z)

        # move Z stage to new height axis position
        core.set_position(height_position_um)
        core.wait_for_device(z_stage)

        # create events to execute scan across this z plane
        events = []
        for y in range(tile_axis_positions):
            # calculate tile axis position
            tile_position_um = tile_axis_start_um+(tile_axis_step_um*y)
            for c in range(len(channel_states)):
                for x in range(scan_axis_positions+10):
                    if channel_states[c]==1:
                        if (c==0):
                            evt = { 'axes': {'x': x, 'y':y, 'z':z}, 'x': scan_axis_start_um, 'y': tile_position_um, 'z': height_position_um,
                                'channel' : {'group': 'Coherent-State', 'config': '405nm'}}
                        elif (c==1):
                            evt = { 'axes': {'x': x, 'y':y, 'z':z}, 'x': scan_axis_start_um, 'y': tile_position_um, 'z': height_position_um,
                                'channel' : {'group': 'Coherent-State', 'config': '488nm'}}
                        elif (c==2):
                            evt = { 'axes': {'x': x, 'y':y, 'z':z}, 'x': scan_axis_start_um, 'y': tile_position_um, 'z': height_position_um,
                                'channel' : {'group': 'Coherent-State', 'config': '561nm'}}
                        elif (c==3):
                            evt = { 'axes': {'x': x, 'y':y, 'z':z}, 'x': scan_axis_start_um, 'y': tile_position_um, 'z': height_position_um,
                                'channel' : {'group': 'Coherent-State', 'config': '637nm'}}
                        elif (c==4):
                            evt = { 'axes': {'x': x, 'y':y, 'z':z}, 'x': scan_axis_start_um, 'y': tile_position_um, 'z': height_position_um,
                                'channel' : {'group': 'Coherent-State', 'config': '730nm'}}

                        events.append(evt)

        # set camera to trigger first mode for stage synchronization
        # give camera time to change modes
        core.set_property('Camera','TriggerMode','Trigger first')
        time.sleep(5)

        # update save_name with current Z plane
        save_name_z = save_name + '_z'+str(z).zfill(3)

        # run acquisition at this Z plane
        with Acquisition(directory=save_directory, name=save_name_z, post_hardware_hook_fn=post_hook_fn,
                        post_camera_hook_fn=camera_hook_fn, show_display=True, max_multi_res_index=0) as acq:
            acq.acquire(events)

            # added this code in an attempt to clean up resources, given the ZMQ error we are getting when using two hooks
            acq.acquire(None)
            acq.await_completion()
        
        # turn off lasers
        core.set_config('Coherent-State','off')
        core.wait_for_config('Coherent-State','off')

        # set camera to internal trigger
        # this is necessary to avoid PVCAM driver issues that we keep having for long acquisitions.
        # give camera time to change modes
        core.set_property('Camera','TriggerMode','Internal Trigger')
        time.sleep(5)

# run
if __name__ == "__main__":
    main()