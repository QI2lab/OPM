#!/usr/bin/env python

'''
OPM stage control with fluidics

Shepherd 02/21
'''

# imports
from pycromanager import Bridge, Acquisition
from pathlib import Path
import numpy as np
import time
import sys
import msvcrt
import pandas as pd

def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)
    #answer = core.get_property('TigerCommHub','SerialCommand') # might be able to remove

    return event
    
def post_hook_fn(event,bridge,event_queue):
    core = bridge.get_core()

    # set camera to external control
    core.set_config('Camera-TriggerSource','EXTERNAL')
    core.wait_for_config('Camera-TriggerSource','EXTERNAL')
  
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
    state_635 = 1
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 10
    power_488 = 10
    power_561 = 50
    power_635 = 50
    power_730 = 0

    # exposure time
    exposure_ms = 5.0

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = 200. #unit: um
    scan_axis_end_um = 700. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = 1000 #unit: um
    tile_axis_end_um = 1500. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = -300. #unit: um
    height_axis_end_um = -295 #unit:  um

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [0, 1024, 1600, 512] #unit: pixels

    # setup file name
    save_directory=Path('E:/20210211f/')
    save_name = 'flash_cd31_pdgfa'

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # connect to Micromanager instance
    bridge = Bridge()
    core = bridge.get_core()

    # turn off lasers
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set camera to fast readout mode
    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')

    # set camera to fast readout mode
    core.set_config('Camera-TriggerType','START')
    core.wait_for_config('Camera-TriggerType','START')

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',100000)
    time.sleep(1)

    # set exposure
    core.set_exposure(exposure_ms)

    # grab exposure
    true_exposure = core.get_exposure()

    # get actual framerate from micromanager properties
    actual_readout_ms = true_exposure+float(core.get_property('Camera','ReadoutTime')) #unit: ms

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # scan axis setup
    scan_axis_step_um = 0.4  # unit: um
    scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
    scan_axis_start_mm = scan_axis_start_um / 1000. #unit: mm
    scan_axis_end_mm = scan_axis_end_um / 1000. #unit: mm
    scan_axis_range_um = np.abs(scan_axis_end_um-scan_axis_start_um)  # unit: um
    scan_axis_range_mm = scan_axis_range_um / 1000 #unit: mm
    actual_exposure_s = actual_readout_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / actual_exposure_s,2) #unit: mm/s
    scan_axis_positions = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)  #unit: number of positions

    # tile axis setup
    tile_axis_overlap=0.3 #unit: percentage
    tile_axis_range_um = np.abs(tile_axis_end_um - tile_axis_start_um) #unit: um
    tile_axis_range_mm = tile_axis_range_um / 1000 #unit: mm
    tile_axis_ROI = ROI[2]*pixel_size_um  #unit: um
    tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
    tile_axis_step_mm = tile_axis_step_um / 1000 #unit: mm
    tile_axis_positions = np.rint(tile_axis_range_mm / tile_axis_step_mm).astype(int)+1  #unit: number of positions
    # if tile_axis_positions rounded to zero, make sure we acquire at least one position
    if tile_axis_positions == 0:
        tile_axis_positions=1

    # height axis setup
    height_axis_overlap=0.2 #unit: percentage
    height_axis_range_um = np.abs(height_axis_end_um-height_axis_start_um) #unit: um
    height_axis_range_mm = height_axis_range_um / 1000 #unit: mm
    height_axis_ROI = ROI[3]*pixel_size_um*np.sin(30.*np.pi/180.) #unit: um 
    height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
    height_axis_step_mm = height_axis_step_um / 1000  #unit: mm
    height_axis_positions = np.rint(height_axis_range_mm / height_axis_step_mm).astype(int)+1 #unit: number of positions
    # if height_axis_positions rounded to zero, make sure we acquire at least one position
    if height_axis_positions==0:
        height_axis_positions=1

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # Setup Tiger controller to pass signal when the scan stage cross the start position to the PLC
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    #addrOutputBNC3 = 35 # BNC3 on the PLC front panel
    addrOutputBNC1 = 33 # BNC1 on the PLC front panel
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC1)
    core.set_property(plcName, propCellConfig, addrStageSync)

    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set tile axis speed for all moves
    command = 'SPEED Y=.1'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis speed for large move to initial position
    command = 'SPEED X=.1'
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

    # set range and return speed (5% of max) for scan axis
    # expects mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=10'
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

    # output experiment info
    print('Number of X positions: '+str(scan_axis_positions))
    print('Number of Y tiles: '+str(tile_axis_positions))
    print('Number of Z slabs: '+str(height_axis_positions))
    print('Number of channels: '+str(np.sum(channel_states)))

    # create events to execute scan
    events = []
    for c in range(len(channel_states)):
        for x in range(scan_axis_positions+30): #TO DO: Fix need for extra frames in ASI setup, not here.
            if channel_states[c]==1:
                if (c==0):
                    evt = { 'axes': {'z': x}, 'channel' : {'group': 'Laser', 'config': '405'}}
                    events.append(evt)
                elif (c==1):
                    evt = { 'axes': {'z': x}, 'channel' : {'group': 'Laser', 'config': '488'}}
                    events.append(evt)
                elif (c==2):
                    evt = { 'axes': {'z': x},'channel' : {'group': 'Laser', 'config': '561'}}
                    events.append(evt)
                elif (c==3):
                    evt = { 'axes': {'z': x},'channel' : {'group': 'Laser', 'config': '637'}}
                    events.append(evt)
                elif (c==4):
                    evt = { 'axes': {'z': x}, 'channel' : {'group': 'Laser', 'config': '730'}}
                    events.append(evt)

    for y in range(tile_axis_positions):
        # calculate tile axis position
        tile_position_um = tile_axis_start_um+(tile_axis_step_um*y)
        
        # move XY stage to new tile axis position
        core.set_xy_position(scan_axis_start_um,tile_position_um)
        core.wait_for_device(xy_stage)
            
        for z in range(height_axis_positions):
            # calculate height axis position
            height_position_um = height_axis_start_um+(height_axis_step_um*z)

            # move Z stage to new height axis position
            core.set_position(height_position_um)
            core.wait_for_device(z_stage)

            # update save_name with current tile information
            save_name_z = save_name +'_y'+str(y).zfill(4)+'_z'+str(z).zfill(4)

            # save actual stage positions
            xy_pos = core.get_xy_stage_position()
            stage_x = xy_pos.x
            stage_y = xy_pos.y
            stage_z = core.get_position()
            current_stage_data = [{'stage_x': stage_x, 'stage_y': stage_y, 'stage_z': stage_z}]
            df_current_stage = pd.DataFrame(current_stage_data)
            
            # run acquisition at this Z plane
            with Acquisition(directory=save_directory, name=save_name_z, post_hardware_hook_fn=post_hook_fn,
                            post_camera_hook_fn=camera_hook_fn, show_display=False, max_multi_res_index=0, 
                            saving_queue_size=5000) as acq:
                acq.acquire(events)

            # save stage scan positions
            save_name_stage_positions = Path(str(y).zfill(4)+'_'+str(z).zfill(4)+'_stage_scan_positions.pkl')
            save_name_stage_positions = save_directory / save_name_stage_positions
            df_current_stage.to_pickle(save_name_stage_positions)

            # turn off lasers
            core.set_config('Laser','Off')
            core.wait_for_config('Laser','Off')

            # try to clean up acquisition so that AcqEngJ releases directory. This way we can move it to the network storage
            # in the background.
            # NOTE: This is a bug, the directory is not released until Micromanager is shutdown
            # https://github.com/micro-manager/pycro-manager/issues/218
            acq = None

    # save stage scan parameters
    scan_param_data = [{'theta': 30.0, 'scan step': scan_axis_step_um*1000., 'pixel size': pixel_size_um*1000.}]
    df_stage_scan_params = pd.DataFrame(scan_param_data)
    save_name_stage_params = save_directory / 'stage_scan_params.pkl'
    df_stage_scan_params.to_pickle(save_name_stage_params)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()