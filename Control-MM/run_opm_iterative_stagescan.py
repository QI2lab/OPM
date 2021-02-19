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
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP

def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)
    #answer = core.get_property('TigerCommHub','SerialCommand') # might be able to remove

    return event
    
def post_hook_fn(event,bridge,event_queue):
    global df_stage_positions

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

    # save actual stage positions
    xy_pos = core.get_xy_stage_position()
    stage_x = xy_pos.x
    stage_y = xy_pos.y
    stage_z = core.get_position()
    current_stage_data = [{'stage_x': stage_x, 'stage_y': stage_y, 'stage_z': stage_z}]
    df_current_stage = pd.DataFrame(current_stage_data)
    df_stage_positions = df_stage_positions.append(df_current_stage, ignore_index=True)

    # set camera to software control
    core.set_config('Camera-TriggerInput','INTERNAL')
    core.wait_for_config('Camera-TriggerInput','INTERNAL')

    # set camera to external control
    core.set_config('Camera-TriggerInput','EXTERNALSTART')
    core.wait_for_config('Camera-TriggerInput','EXTERNALSTART')

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

def lookup_valve(source_name):

    valve_dict = {'B01': [0,1], 'B02': [0,2], 'B03': [0,3], 'B04': [0,4], 'B05': [0,5], 'B06': [0,6], 'B07': [0,7], 
                  'B08': [1,1], 'B09': [1,2], 'B10': [1,3], 'B11': [1,4], 'B12': [1,5], 'B13': [1,6], 'B14': [1,7], 
                  'B15': [2,1], 'B16': [2,2], 'B17': [2,3], 'B18': [2,4], 'B19': [2,5], 'B20': [2,6], 'B21': [2,7], 
                  'B22': [3,1], 'B23': [3,2], 'B24': [3,3],
                  'SSC': [3,0], 'READOUT WASH': [3,4], 'IMAGING BUFFER': [3,5], 'CLEAVE': [3,7]}

    valve_position = valve_dict.get(source_name)

    return valve_position

def run_fluidic_program(r, df_program, mvp_controller, pump_controller):

    # select current round
    df_current_program = df_program[(df_program['round']==r+1)]

    for index, row in df_current_program.iterrows():
        # extract source name
        source_name = str(row['source'])

        # extract volume to pump and time
        pump_amount_ml = float(row['volume'])
        pump_time_min = float(row['time'])

        if source_name == 'RUN':
            pump_controller.stopFlow()
            print('Fluidics round done, running imaging.')
        elif source_name == 'PAUSE':
            pump_controller.stopFlow()
            print('Pausing for:' +str(pump_time_min*60)+' seconds.')
            time.sleep(pump_time_min*60)
        else:
            # extract and set valve
            valve_position = lookup_valve(source_name)
            mvp_unit = valve_position[0]
            valve_number = valve_position[1]
            if mvp_unit == 0:
                mvp_controller.changePort(valve_ID=0,port_ID=valve_number)
                mvp_controller.changePort(valve_ID=1,port_ID=0)
                mvp_controller.changePort(valve_ID=2,port_ID=0)
                mvp_controller.changePort(valve_ID=3,port_ID=0)
            elif mvp_unit == 1:
                mvp_controller.changePort(valve_ID=0,port_ID=0)
                mvp_controller.changePort(valve_ID=1,port_ID=valve_number)
                mvp_controller.changePort(valve_ID=2,port_ID=0)
                mvp_controller.changePort(valve_ID=3,port_ID=0)
            elif mvp_unit == 2:
                mvp_controller.changePort(valve_ID=0,port_ID=0)
                mvp_controller.changePort(valve_ID=1,port_ID=0)
                mvp_controller.changePort(valve_ID=2,port_ID=valve_number)
                mvp_controller.changePort(valve_ID=3,port_ID=0)
            elif mvp_unit == 3:
                mvp_controller.changePort(valve_ID=0,port_ID=0)
                mvp_controller.changePort(valve_ID=1,port_ID=0)
                mvp_controller.changePort(valve_ID=2,port_ID=0)
                mvp_controller.changePort(valve_ID=3,port_ID=valve_number)
            time.sleep(3)

            print('MVP unit: '+str(mvp_unit)+'; Valve #: '+str(valve_number))

            # convert ml/min rate to pump rate
            # this is hardcoded to our fluidic setup
            # please check for your own setup
            pump_rate = -1.0

            if np.round((pump_amount_ml/pump_time_min),2) == 0.50:
                pump_rate = 11.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.40:
                pump_rate = 10.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.36:
                pump_rate = 9.5
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.33:
                pump_rate = 9.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.22:
                pump_rate = 5.0

            print('Pump setting: '+str(pump_rate))

            if pump_rate == -1.0:
                print('Error in determining pump rate. Exiting.')
                sys.exit()

            # run pump
            pump_controller.startFlow(pump_rate,direction='Forward')
            time.sleep(pump_time_min*60)
            pump_controller.stopFlow()
    
    return True

def main():

    # hacky way to generate our own metadata on actual stage positions, not requested ones.
    global df_stage_positions

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # lasers to use
    # 0 -> inactive
    # 1 -> active
    state_405 = 0
    state_488 = 1
    state_561 = 1
    state_635 = 1
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 5
    power_488 = 80
    power_561 = 80
    power_635 = 80
    power_730 = 0

    # exposure time
    exposure_ms = 5.0

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = 2500. #unit: um
    scan_axis_end_um = 3000. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = -6500 #unit: um
    tile_axis_end_um = -6000. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = 95. #unit: um
    height_axis_end_um = 97 #unit:  um

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [0, 1024, 1600, 256] #unit: pixels

    # setup file name
    save_directory=Path('E:/20210208a/')
    program_name = Path('E:/20210208_firstround.csv')
    save_name = 'fresh_bDNA'

    run_fluidics = False
    run_scope = True

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # connect to Micromanager instance
    bridge = Bridge()
    core = bridge.get_core()

    # turn off lasers
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set camera to external START trigger
    # give camera time to change modes if necessary
    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')

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

    if run_fluidics == True:
        # setup pump parameters
        pump_parameters = {'pump_com_port': 'COM3',
                        'pump_ID': 30,
                        'verbose': True,
                        'simulate_pump': False,
                        'serial_verbose': False,
                        'flip_flow_direction': False}

        # connect to pump
        pump_controller = APump(pump_parameters)

        # set pump to remote control
        pump_controller.enableRemoteControl(True)

        # connect to valves
        valve_controller = HamiltonMVP(com_port='COM4')

        # initialize valves
        valve_controller.autoAddress()
        valve_controller.autoDetectValves()

        df_program = pd.read_csv(program_name)
        iterative_rounds = df_program['round'].max()

    else:
        iterative_rounds = 1

    # output experiment info
    print('Number of labeling rounds: '+str(iterative_rounds))
    print('Number of X positions: '+str(scan_axis_positions))
    print('Number of Y tiles: '+str(tile_axis_positions))
    print('Number of Z slabs: '+str(height_axis_positions))

    for r in range(iterative_rounds):

        if run_fluidics == True:
            success_fluidics = False
            success_fluidics = run_fluidic_program(r, df_program, valve_controller, pump_controller)
            if not(success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()

        if run_scope == True:
            # create events to execute scan
            events = []
            for y in range(tile_axis_positions):
                # calculate tile axis position
                tile_axis_position_um = tile_axis_start_um+(tile_axis_step_um*y)
                for z in range(height_axis_positions):
                    # calculate height axis position
                    height_axis_position_um = height_axis_start_um+(height_axis_step_um*z)
                    for c in range(len(channel_states)):
                        for x in range(scan_axis_positions): #TO DO: Fix need for extra frames in ASI setup, not here.
                            if channel_states[c]==1:
                                if (c==0):
                                    evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': scan_axis_start_um,
                                            'y': tile_axis_position_um, 'z': height_axis_position_um, 'channel' : {'group': 'Laser', 'config': '405'}}
                                    events.append(evt)
                                elif (c==1):
                                    evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': scan_axis_start_um,
                                            'y': tile_axis_position_um, 'z': height_axis_position_um, 'channel' : {'group': 'Laser', 'config': '488'}}
                                    events.append(evt)
                                elif (c==2):
                                    evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': scan_axis_start_um,
                                            'y': tile_axis_position_um, 'z': height_axis_position_um, 'channel' : {'group': 'Laser', 'config': '561'}}
                                    events.append(evt)
                                elif (c==3):
                                    evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': scan_axis_start_um,
                                            'y': tile_axis_position_um, 'z': height_axis_position_um, 'channel' : {'group': 'Laser', 'config': '637'}}
                                    events.append(evt)
                                elif (c==4):
                                    evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': scan_axis_start_um,
                                            'y': tile_axis_position_um, 'z': height_axis_position_um, 'channel' : {'group': 'Laser', 'config': '730'}}
                                    events.append(evt)

            #evt = None
            #events.append(evt)

            # run acquisition
            save_name_r = 'r'+str(r).zfill(3)+'_'+save_name
            '''
            with Acquisition(directory=save_directory, name=save_name_r, post_hardware_hook_fn=post_hook_fn, 
                             post_camera_hook_fn=camera_hook_fn, show_display=False, max_multi_res_index=None, debug=False) as acq:
                acq.acquire(events)

            '''
            with Acquisition(directory=save_directory, name=save_name_r, image_process_fn=None, event_generation_hook_fn=None, 
                             pre_hardware_hook_fn=None, post_hardware_hook_fn=post_hook_fn, post_camera_hook_fn=camera_hook_fn, 
                             show_display=False, tile_overlap=None, max_multi_res_index=None, magellan_acq_index=None, 
                             magellan_explore=False, process=False, saving_queue_size = 100, debug=True, mem_map_index=True) as acq:
                acq.acquire(events)
            
            # turn off lasers
            core.set_config('Laser','Off')
            core.wait_for_config('Laser','Off')

            # clean up acqusition object
            acq = None

            # save stage scan positions
            save_name_stage_positions = Path('r_'+str(r)+'_stage_scan_positions.pkl')
            save_name_stage_positions = save_directory / save_name_stage_positions
            df_stage_positions.to_pickle(save_name_stage_positions)
            df_stage_positions = pd.DataFrame(columns=['stage_x','stage_y','stage_z'])

    # save stage scan parameters
    scan_param_data = [{'theta': 30.0, 'scan step': scan_axis_step_um*1000., 'pixel size': pixel_size_um*1000.}]
    df_stage_scan_params = pd.DataFrame(scan_param_data)
    save_name_stage_params = save_directory / 'stage_scan_params.pkl'
    df_stage_scan_params.to_pickle(save_name_stage_params)
    # put camera back to software control
    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')


#-----------------------------------------------------------------------------

# create empty dataframe to hold stage positions for BigStitcher H5 creation
df_stage_positions = pd.DataFrame(columns=['stage_x','stage_y','stage_z'])

if __name__ == "__main__":
    main()