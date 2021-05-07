#!/usr/bin/env python

'''
OPM stage control with fluidics.

Shepherd 05/21 - add in fluidics control. Will refactor once it works into separate files.
Shepherd 04/21 - large-scale changes for new metadata and on-the-fly uploading to server for simultaneous reconstruction
'''

# imports
from pycromanager import Bridge, Acquisition
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP
from pathlib import Path
import numpy as np
import time
import sys
import msvcrt
import pandas as pd
import subprocess
import PyDAQmx as daq
import ctypes as ct
from itertools import compress
import shutil
from threading import Thread
import data_io

def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)

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

        # extract pump rate
        pump_amount_ml = float(row['volume'])
        pump_time_min  = float(row['time'])

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

            if np.round((pump_amount_ml/pump_time_min),2) == 1:
                pump_rate = 48.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.50:
                pump_rate = 11.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.40:
                pump_rate = 10.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.36:
                pump_rate = 9.5
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.33:
                pump_rate = 9.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.22:
                pump_rate = 5.0
            elif np.round((pump_amount_ml/pump_time_min),2) == 0.2:
                pump_rate = 4.0

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

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # set up lasers
    channel_labels = ["405", "488", "561", "635", "730"]
    channel_states = [True, True, False, False, False] # true -> active, false -> inactive
    channel_powers = [15, 75, 75, 75, 0] # (0 -> 100%)
    do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

    # parse which channels are active
    active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
    n_active_channels = len(active_channel_indices)
    
    print("%d active channels: " % n_active_channels, end="")
    for ind in active_channel_indices:
        print("%s " % channel_labels[ind], end="")
    print("")

    # exposure time
    exposure_ms = 10.0

    # galvo voltage at neutral
    galvo_neutral_volt = 0.0 # unit: volts

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = 4000. #unit: um
    scan_axis_end_um = 5000. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = -8000 #unit: um
    tile_axis_end_um = -7000. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = 15977. #unit: um
    height_axis_end_um = 15978. #unit:  um

    # setup file name
    save_directory=Path('E:/20210506temp/')
    program_name = Path('D:/WASH_FLUIDICS.csv')
    save_name = 'temp'

    run_fluidics = True
    run_scope = False

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    if run_scope == True:
        # connect to Micromanager instance
        bridge = Bridge()
        core = bridge.get_core()

        # turn off lasers
        core.set_config('Laser','Off')
        core.wait_for_config('Laser','Off')

        # set camera to fast readout mode
        core.set_config('Camera-Setup','ScanMode3')
        core.wait_for_config('Camera-Setup','ScanMode3')

        # set camera to START mode upon input trigger
        core.set_config('Camera-TriggerType','START')
        core.wait_for_config('Camera-TriggerType','START')

        # set camera to positive input trigger
        core.set_config('Camera-TriggerPolarity','POSITIVE')
        core.wait_for_config('Camera-TriggerPolarity','POSITIVE')

        # set camera to internal control
        core.set_config('Camera-TriggerSource','INTERNAL')
        core.wait_for_config('Camera-TriggerSource','INTERNAL')

        # set camera to output positive triggers on all lines for exposure
        core.set_property('Camera','OUTPUT TRIGGER KIND[0]','EXPOSURE')
        core.set_property('Camera','OUTPUT TRIGGER KIND[1]','EXPOSURE')
        core.set_property('Camera','OUTPUT TRIGGER KIND[2]','EXPOSURE')
        core.set_property('Camera','OUTPUT TRIGGER POLARITY[0]','POSITIVE')
        core.set_property('Camera','OUTPUT TRIGGER POLARITY[1]','POSITIVE')
        core.set_property('Camera','OUTPUT TRIGGER POLARITY[2]','POSITIVE')

        # change core timeout for long stage moves
        core.set_property('Core','TimeoutMs',100000)
        time.sleep(1)

        # set exposure
        core.set_exposure(exposure_ms)

        # determine image size
        core.snap_image()
        y_pixels = core.get_image_height()
        x_pixels = core.get_image_width()

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
        tile_axis_overlap=0.2 #unit: percentage
        tile_axis_range_um = np.abs(tile_axis_end_um - tile_axis_start_um) #unit: um
        tile_axis_range_mm = tile_axis_range_um / 1000 #unit: mm
        tile_axis_ROI = x_pixels*pixel_size_um  #unit: um
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
        height_axis_ROI = y_pixels*pixel_size_um*np.sin(30.*np.pi/180.) #unit: um 
        height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
        height_axis_step_mm = height_axis_step_um / 1000  #unit: mm
        height_axis_positions = np.rint(height_axis_range_mm / height_axis_step_mm).astype(int)+1 #unit: number of positions
        # if height_axis_positions rounded to zero, make sure we acquire at least one position
        if height_axis_positions==0:
            height_axis_positions=1

        # get handle to xy and z stages
        xy_stage = core.get_xy_stage_device()
        z_stage = core.get_focus_device()

        # set the galvo to the neutral position if it is not already
        try: 
            taskAO_first = daq.Task()
            taskAO_first.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)
            taskAO_first.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
            taskAO_first.StopTask()
            taskAO_first.ClearTask()
        except:
            print("DAQmx Error %s"%err)

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

        # set all laser to external triggering
        core.set_config('Trigger-405','External-Digital')
        core.wait_for_config('Trigger-405','External-Digital')
        core.set_config('Trigger-488','External-Digital')
        core.wait_for_config('Trigger-488','External-Digital')
        core.set_config('Trigger-561','External-Digital')
        core.wait_for_config('Trigger-561','External-Digital')
        core.set_config('Trigger-637','External-Digital')
        core.wait_for_config('Trigger-637','External-Digital')
        core.set_config('Trigger-730','External-Digital')
        core.wait_for_config('Trigger-730','External-Digital')

        # turn all lasers on
        core.set_config('Laser','AllOn')
        core.wait_for_config('Laser','AllOn')

        # set lasers to user defined power
        core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
        core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
        core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
        core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
        core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

        # setup DAQ
        samples_per_ch = 2
        DAQ_sample_rate_Hz = 10000
        num_DI_channels = 8

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
        #valve_controller.autoDetectValves()

        df_program = pd.read_csv(program_name)
        iterative_rounds = df_program['round'].max()

    else:
        iterative_rounds = 1

    # flags for metadata and processing
    setup_processing=False
    setup_metadata=False

    if run_scope == True:
        # output experiment info
        print('Number of labeling rounds: '+str(iterative_rounds))
        print('Number of channels: '+str(n_active_channels))
        print('Number of Z slabs: '+str(height_axis_positions))
        print('Number of Y tiles: '+str(tile_axis_positions))
        print('Number of X positions: '+str(scan_axis_positions))
    
        # create events to execute scan
        events = []
        for x in range(scan_axis_positions):
            evt = { 'axes': {'z': x}}
            events.append(evt)

    for r_idx in range(iterative_rounds):

        if run_fluidics == True:
            success_fluidics = False
            success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
            if not(success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()

        if run_scope == True:
            # move scan scan stage to initial position
            core.set_xy_position(scan_axis_start_um,tile_axis_start_um)
            core.wait_for_device(xy_stage)
            core.set_position(height_axis_start_um)
            core.wait_for_device(z_stage)

            for y_idx in range(tile_axis_positions):
                # calculate tile axis position
                tile_position_um = tile_axis_start_um+(tile_axis_step_um*y_idx)
                
                # move XY stage to new tile axis position
                core.set_xy_position(scan_axis_start_um,tile_position_um)
                core.wait_for_device(xy_stage)
                    
                for z_idx in range(height_axis_positions):
                    # calculate height axis position
                    height_position_um = height_axis_start_um+(height_axis_step_um*z_idx)

                    # move Z stage to new height axis position
                    core.set_position(height_position_um)
                    core.wait_for_device(z_stage)

                    for ch_idx in active_channel_indices:

                        # create DAQ pattern for laser strobing controlled via rolling shutter
                        dataDO = np.zeros((samples_per_ch,num_DI_channels),dtype=np.uint8)
                        dataDO[0,ch_idx]=1
                        dataDO[1,ch_idx]=0
                        #print(dataDO)
                    
                        # update save_name with current tile information
                        save_name_ryzc = save_name +'_r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)

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

                        # setup DAQ for laser strobing
                        try:    
                            # ----- DIGITAL input -------
                            taskDI = daq.Task()
                            taskDI.CreateDIChan("/Dev1/PFI0","",daq.DAQmx_Val_ChanForAllLines)
                            #taskDI.CreateDIChan("OnboardClock","",daq.DAQmx_Val_ChanForAllLines)
                            
                            ## Configure change detection timing (from wave generator)
                            taskDI.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
                            taskDI.CfgChangeDetectionTiming("/Dev1/PFI0","/Dev1/PFI0",daq.DAQmx_Val_ContSamps,0)
                            #taskDI.CfgChangeDetectionTiming("/Dev1/PFI0","/Dev1/PFI0",daq.DAQmx_Val_FiniteSamps,samples_per_ch)

                            ## Set where the starting trigger 
                            taskDI.CfgDigEdgeStartTrig("/Dev1/PFI0",daq.DAQmx_Val_Rising)
                            #taskDI.SetTrigAttribute(daq.DAQmx_StartTrig_Retriggerable,retriggerable) # only available for finite task sampling
                            
                            ## Export DI signal to unused PFI pins, for clock and start
                            taskDI.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI2")
                            taskDI.ExportSignal(daq.DAQmx_Val_StartTrigger,"/Dev1/PFI1")
                            
                            # ----- DIGITAL output ------   
                            taskDO = daq.Task()
                            # TO DO: Write each laser line separately!
                            taskDO.CreateDOChan("/Dev1/port0/line0:7","",daq.DAQmx_Val_ChanForAllLines)

                            ## Configure timing (from DI task) 
                            taskDO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_ContSamps,samples_per_ch)
                            
                            ## Write the output waveform
                            samples_per_ch_ct_digital = ct.c_int32()
                            taskDO.WriteDigitalLines(samples_per_ch,False,10.0,daq.DAQmx_Val_GroupByChannel,dataDO,ct.byref(samples_per_ch_ct_digital),None)
                            #rint("WriteDigitalLines sample per channel count = %d" % samples_per_ch_ct_digital.value)

                            ## ------ Start digital input and output tasks ----------
                            taskDO.StartTask()    
                            taskDI.StartTask()

                        except daq.DAQError as err:
                            print("DAQmx Error %s"%err)

                        # set camera to external control
                        # DCAM sets the camera back to INTERNAL mode after each acquisition
                        core.set_config('Camera-TriggerSource','EXTERNAL')
                        core.wait_for_config('Camera-TriggerSource','EXTERNAL')

                        # verify that camera actually switched back to external trigger mode
                        trigger_state = core.get_property('Camera','TRIGGER SOURCE')

                        # if not in external control, keep trying until camera changes settings
                        while not(trigger_state =='EXTERNAL'):
                            time.sleep(2.0)
                            core.set_config('Camera-TriggerSource','EXTERNAL')
                            core.wait_for_config('Camera-TriggerSource','EXTERNAL')
                            trigger_state = core.get_property('Camera','TRIGGER SOURCE')

                        
                        print('R: '+str(r_idx)+' Y: '+str(y_idx)+' Z: '+str(z_idx)+' C: '+str(ch_idx))
                        # run acquisition for this tyzc combination
                        with Acquisition(directory=save_directory, name=save_name_ryzc, saving_queue_size=5000,
                                        post_camera_hook_fn=camera_hook_fn, show_display=False, max_multi_res_index=0) as acq:
                            acq.acquire(events)

                        # clean up acquisition so that AcqEngJ releases directory.
                        # NOTE: This currently does not work. 
                        acq = None

                        # stop DAQ and make sure it is at zero
                        try:
                            ## Stop and clear both tasks
                            taskDI.StopTask()
                            taskDO.StopTask()
                            taskDI.ClearTask()
                            taskDO.ClearTask()
                        except daq.DAQError as err:
                            print("DAQmx Error %s"%err)

                        # save experimental info after first tile. 
                        # we do it this way so that Pycromanager can manage the directories.
                        if (setup_metadata):
                            # save stage scan parameters
                            scan_param_data = [{'root_name': str(save_name),
                                'scan_type': str('stage'),
                                'theta': float(30.0), 
                                'scan_step': float(scan_axis_step_um*1000.), 
                                'pixel_size': float(pixel_size_um*1000.),
                                'num_r': int(iterative_rounds),
                                'num_y': int(tile_axis_positions),
                                'num_z': int(height_axis_positions),
                                'num_ch': int(n_active_channels),
                                'scan_axis_positions': int(scan_axis_positions),
                                'y_pixels': int(y_pixels),
                                'x_pixels': int(x_pixels),
                                '405_active': channel_states[0],
                                '488_active': channel_states[1],
                                '561_active': channel_states[2],
                                '635_active': channel_states[3],
                                '730_active': channel_states[4]}]

                            # df_stage_scan_params = pd.DataFrame(scan_param_data)
                            # save_name_stage_params = save_directory / 'scan_metadata.csv'
                            # df_stage_scan_params.to_csv(save_name_stage_params)
                            data_io.write_metadata(scan_param_data[0], save_directory / 'scan_metadata.csv')

                            setup_metadata=False

                        # save stage scan positions after each tile
                        save_name_stage_positions = Path('r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_stage_positions.csv')
                        save_name_stage_positions = save_directory / save_name_stage_positions
                        # todo: use data_io instead
                        df_current_stage.to_csv(save_name_stage_positions)

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
                        
                        # if first tile, make parent directory on NAS and start reconstruction script on the server
                        if setup_processing:
                            # make home directory on NAS
                            save_directory_path = Path(save_directory)
                            remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
                            cmd='mkdir ' + str(remote_directory)
                            status_mkdir = subprocess.run(cmd, shell=True)

                            # copy full experiment metadata to NAS
                            src= Path(save_directory) / Path('scan_metadata.csv') 
                            dst= Path(remote_directory) / Path('scan_metadata.csv') 
                            Thread(target=shutil.copy, args=[str(src), str(dst)]).start()

                            setup_processing=False
                        
                        # copy current tyzc metadata to NAS
                        save_directory_path = Path(save_directory)
                        remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
                        src= Path(save_directory) / Path(save_name_stage_positions.parts[2])
                        dst= Path(remote_directory) / Path(save_name_stage_positions.parts[2])
                        Thread(target=shutil.copy, args=[str(src), str(dst)]).start()

                        # copy current tyzc data to NAS
                        save_directory_path = Path(save_directory)
                        remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
                        src= Path(save_directory) / Path(save_name_ryzc+ '_1') 
                        dst= Path(remote_directory) / Path(save_name_ryzc+ '_1') 
                        Thread(target=shutil.copytree, args=[str(src), str(dst)]).start()
                        
    # set lasers to zero power
    channel_powers = [0.,0.,0.,0.,0.]
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # turn all lasers off
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set all lasers back to software control
    core.set_config('Trigger-405','CW (constant power)')
    core.wait_for_config('Trigger-405','CW (constant power)')
    core.set_config('Trigger-488','CW (constant power)')
    core.wait_for_config('Trigger-488','CW (constant power)')
    core.set_config('Trigger-561','CW (constant power)')
    core.wait_for_config('Trigger-561','CW (constant power)')
    core.set_config('Trigger-637','CW (constant power)')
    core.wait_for_config('Trigger-637','CW (constant power)')
    core.set_config('Trigger-730','CW (constant power)')
    core.wait_for_config('Trigger-730','CW (constant power)')

    # set camera to internal control
    core.set_config('Camera-TriggerSource','INTERNAL')
    core.wait_for_config('Camera-TriggerSource','INTERNAL')

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()