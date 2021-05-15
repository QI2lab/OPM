#!/usr/bin/env python

'''
OPM stage control with iterative fluidics.

Shepherd 05/21 - pull all settings from MM GUI and prompt user to setup experiment that way using "easygui" package
Shepherd 05/21 - add in fluidics control. Recently refactored into seaprate files
Shepherd 04/21 - large-scale changes for new metadata and on-the-fly uploading to server for simultaneous reconstruction
'''

# imports
from pycromanager import Bridge, Acquisition
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP
from fluidics.FluidicsControl import lookup_valve, run_fluidic_program
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
import easygui

def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)

    return event
    
def main():

    # load fluidics program
    fluidics_path = easygui.fileopenbox('Load fluidics program')
    program_name = Path(fluidics_path)

    # define ports for pumps and valves
    pump_COM_port = 'COM3'
    valve_COM_port = 'COM4'

    # connect to Micromanager instance
    bridge = Bridge()
    core = bridge.get_core()
    studio = bridge.get_studio()

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # turn off lasers
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set all lasers to software control
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

    # flags for metadata and processing
    setup_processing=True
    setup_metadata=True
    
    # galvo voltage at neutral
    galvo_neutral_volt = 0.0 # unit: volts

    # setup digital trigger buffer on DAQ
    samples_per_ch = 2
    DAQ_sample_rate_Hz = 10000
    num_DI_channels = 8

    # set the galvo to the neutral position if it is not already
    try: 
        taskAO_first = daq.Task()
        taskAO_first.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)
        taskAO_first.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
        taskAO_first.StopTask()
        taskAO_first.ClearTask()
    except:
        print("DAQmx Error %s"%err)

    # setup pump parameters
    pump_parameters = {'pump_com_port': pump_COM_port,
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
    valve_controller = HamiltonMVP(com_port=valve_COM_port)

    # initialize valves
    valve_controller.autoAddress()

    # load user defined program from hard disk
    # TO DO: this into data_io.py
    df_program = pd.read_csv(program_name)
    iterative_rounds = df_program['round'].max()
    print('Number of iterative rounds: '+str(iterative_rounds))

    # iterate over user defined program
    for r_idx in range(iterative_rounds):

        # set motors to on to actively maintain position during fluidics run
        # TO DO: figure out how to make this work
        #core.set_property('XYStage:XY:31','MaintainState-MA',2)
        #core.set_property('ZStage:M:37','MaintainState-MA',2)

        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
        if not(success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()

        # set motors to standard drift correction setting
        # TO DO: figure out how to make this work
        #core.set_property('XYStage:XY:31','MaintainState-MA',0)
        #core.set_property('ZStage:M:37','MaintainState-MA',0)

        # if first round, have user setup positions, laser intensities, and exposure time in MM GUI
        if r_idx == 0:
            
            # setup imaging parameters using MM GUI
            run_imaging = False
            while not(run_imaging):

                setup_done = False
                while not(setup_done):
                    setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

                # pull current MDA window settings
                acq_manager = studio.acquisitions()
                acq_settings = acq_manager.get_acquisition_settings()

                # grab settings from MM
                # grab and setup save directory and filename
                save_directory=Path(acq_settings.root())
                save_name=Path(acq_settings.prefix())

                # pull active lasers from MDA window
                channel_labels = ['405', '488', '561', '637', '730']
                channel_states = [False,False,False,False,False] #define array to keep active channels
                channels = acq_settings.channels() # get active channels in MDA window
                for idx in range(channels.size()):
                    channel = channels.get(idx) # pull channel information
                    if channel.config() == channel_labels[0]: 
                        channel_states[0]=True
                    if channel.config() == channel_labels[1]: 
                        channel_states[1]=True
                    elif channel.config() == channel_labels[2]: 
                        channel_states[2]=True
                    elif channel.config() == channel_labels[3]: 
                        channel_states[3]=True
                    elif channel.config() == channel_labels[4]: 
                        channel_states[4]=True
                do_ch_pins = [0, 1, 2, 3, 4] # digital output line corresponding to each channel
                
                # pull laser powers from main window
                channel_powers = [0.,0.,0.,0.,0.]
                channel_powers[0] = core.get_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)')
                channel_powers[1] = core.get_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)')
                channel_powers[2] = core.get_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)')
                channel_powers[3] = core.get_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)')
                channel_powers[4] = core.get_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)')

                # parse which channels are active
                active_channel_indices = [ind for ind, st in zip(do_ch_pins, channel_states) if st]
                n_active_channels = len(active_channel_indices)

                # set up XY positions
                position_list_manager = studio.positions()
                position_list = position_list_manager.get_position_list()
                number_positions = position_list.get_number_of_positions()
                x_positions = np.empty(number_positions)
                y_positions = np.empty(number_positions)
                z_positions = np.empty(number_positions)

                # iterate through position list to extract XY positions    
                for idx in range(number_positions):
                    pos = position_list.get_position(idx)
                    for ipos in range(pos.size()):
                        stage_pos = pos.get(ipos)
                        if (stage_pos.get_stage_device_label() == 'XYStage:XY:31'):
                            x_positions[idx] = stage_pos.x
                            y_positions[idx] = stage_pos.y
                        if (stage_pos.get_stage_device_label() == 'ZStage:M:37'):
                            z_positions[idx] = stage_pos.x

                # determine corners for XY stage and stop/bottom for Z stage
                # TO DO: setup interpolation and split up XY positions to avoid brute force Z scanning
                scan_axis_start_um = np.round(x_positions.min(),0)
                scan_axis_end_um = np.round(x_positions.max(),0)

                tile_axis_start_um = np.round(y_positions.min(),0)
                tile_axis_end_um = np.round(y_positions.max(),0)

                height_axis_start_um = np.round(z_positions.min(),0)
                height_axis_end_um = np.round(z_positions.max(),0)
         
                # set pixel size
                pixel_size_um = 0.115 # unit: um 

                # get exposure time from main window
                exposure_ms = core.get_exposure()

                # enforce exposure time
                core.set_exposure(exposure_ms)

                # determine image size
                core.snap_image()
                y_pixels = core.get_image_height()
                x_pixels = core.get_image_width()

                # grab exposure
                true_exposure = core.get_exposure()

                # get actual framerate from micromanager properties
                actual_readout_ms = true_exposure+float(core.get_property('Camera','ReadoutTime')) #unit: ms

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

                # construct and display imaging summary to user
                scan_settings = (f"Number of labeling rounds: {str(iterative_rounds)} \n\n"
                                f"Number of Y tiles:  {str(tile_axis_positions)} \n"
                                f"Tile start:  {str(tile_axis_start_um)} \n"
                                f"Tile end:  {str(tile_axis_end_um)} \n\n"
                                f"Number of Z slabs:  {str(height_axis_positions)} \n"
                                f"Height start:  {str(height_axis_start_um)} \n"
                                f"Height end:  {str(height_axis_end_um)} \n\n"
                                f"Number of channels:  {str(n_active_channels)} \n"
                                f"Active lasers: {str(channel_states)} \n"
                                f"Lasers powers: {str(channel_powers)} \n\n"
                                f"Number of scan positions:  {str(scan_axis_positions)} \n"
                                f"Scan start: {str(scan_axis_start_um)}  \n"
                                f"Scan end:  {str(scan_axis_end_um)} \n")
                                
                output = easygui.textbox(scan_settings, 'Please review scan settings')

                # verify user actually wants to run imaging
                run_imaging = easygui.ynbox('Run acquistion?', 'Title', ('Yes', 'No'))

                if run_imaging == True:
                    # disable joystick
                    core.set_property('XYStage:XY:31','JoystickEnabled','No')
                    core.set_property('ZStage:M:37','JoystickInput','0 - none')
                    
                    # set flag to change configuration
                    config_changed = True

        # if last round, switch to DAPI + alexa488 readout instead
        if r_idx == (iterative_rounds - 1):

            # grab settings from MM
            # set up lasers and exposure times
            channel_states = [True,True,False,False,False]
            do_ch_pins = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

            # parse which channels are active
            active_channel_indices = [ind for ind, st in zip(do_ch_pins, channel_states) if st]
            n_active_channels = len(active_channel_indices)

            channel_powers[0] = 5

             # set flag to change configuration
            config_changed = True
      
        if config_changed:

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

            # set lasers to user defined power
            core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
            core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
            core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
            core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
            core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

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

                    # TO DO: set Z stage to constantly hold position
                    # TO DO: figure out autofocusing solution!

                    # update save_name with current tile information
                    save_name_ryz = str(save_name) +'_r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)

                    # use one Acquisition object for all channels at this RYZ position.
                    # see: https://github.com/micro-manager/pycro-manager/issues/286
                    # would be better to go back to one Acquisition per laser line to aid reconstruction and transferring data
                    with Acquisition(directory=str(save_directory), name=save_name_ryz, saving_queue_size=5000,
                                     post_camera_hook_fn=camera_hook_fn, show_display=False, max_multi_res_index=0) as acq:
                        for ch_idx in active_channel_indices:

                            # create events to execute scan for this channel and all stage scan positions
                            events = []
                            for x in range(scan_axis_positions):
                                evt = { 'axes': {'c': ch_idx, 'z': x}}
                                events.append(evt)

                            # create DAQ pattern for laser strobing controlled via rolling shutter
                            dataDO = np.zeros((samples_per_ch,num_DI_channels),dtype=np.uint8)
                            dataDO[0,ch_idx]=1
                            dataDO[1,ch_idx]=0

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
                            current_stage_data = [{'stage_x': stage_x, 
                                                   'stage_y': stage_y, 
                                                   'stage_z': stage_z}]

                            # setup DAQ for laser strobing
                            try:    
                                # ----- DIGITAL input -------
                                taskDI = daq.Task()
                                taskDI.CreateDIChan("/Dev1/PFI0","",daq.DAQmx_Val_ChanForAllLines)
                                
                                ## Configure change detection timing (from wave generator)
                                taskDI.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
                                taskDI.CfgChangeDetectionTiming("/Dev1/PFI0","/Dev1/PFI0",daq.DAQmx_Val_ContSamps,0)

                                ## Set where the starting trigger 
                                taskDI.CfgDigEdgeStartTrig("/Dev1/PFI0",daq.DAQmx_Val_Rising)
                                
                                ## Export DI signal to unused PFI pins, for clock and start
                                taskDI.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI2")
                                taskDI.ExportSignal(daq.DAQmx_Val_StartTrigger,"/Dev1/PFI1")
                                
                                # ----- DIGITAL output ------   
                                taskDO = daq.Task()
                                taskDO.CreateDOChan("/Dev1/port0/line0:7","",daq.DAQmx_Val_ChanForAllLines)

                                ## Configure timing (from DI task) 
                                taskDO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_ContSamps,samples_per_ch)
                                
                                ## Write the output waveform
                                samples_per_ch_ct_digital = ct.c_int32()
                                taskDO.WriteDigitalLines(samples_per_ch,False,10.0,daq.DAQmx_Val_GroupByChannel,dataDO,ct.byref(samples_per_ch_ct_digital),None)

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

                            # run acquisition for this tyzc combination
                            print('R: '+str(r_idx+1)+' Y: '+str(y_idx)+' Z: '+str(z_idx)+' C: '+str(ch_idx))
                            acq.acquire(events)

                            # stop DAQ and make sure it is at zero
                            try:
                                ## Stop and clear both tasks
                                taskDI.StopTask()
                                taskDO.StopTask()
                                taskDI.ClearTask()
                                taskDO.ClearTask()
                            except daq.DAQError as err:
                                print("DAQmx Error %s"%err)

                            # save experimental info after first acquistion. 
                            if (setup_metadata):
                                # save acquistion parameters
                                scan_param_data = [{'root_name': str(save_name),
                                                    'scan_type': str('ierative-stage'),
                                                    'theta': float(30.0), 
                                                    'scan_step_um': float(scan_axis_step_um*1000.), 
                                                    'pixel_size_um': float(pixel_size_um*1000.),
                                                    'axial_step_um': str(''),
                                                    'y_pixels': int(y_pixels),
                                                    'x_pixels': int(x_pixels),
                                                    'exposure_ms': float(exposure_ms),
                                                    'camera': str('Orca_FusionBT'),
                                                    'readout_mode': str('16bit_mode3'),
                                                    'num_r': int(iterative_rounds),
                                                    'num_y': int(tile_axis_positions),
                                                    'num_z': int(height_axis_positions),
                                                    'num_ch': int(n_active_channels),
                                                    'scan_axis_positions': int(scan_axis_positions),
                                                    '405_active': channel_states[0],
                                                    '488_active': channel_states[1],
                                                    '561_active': channel_states[2],
                                                    '635_active': channel_states[3],
                                                    '730_active': channel_states[4],
                                                    '405_power': float(channel_powers[0]),
                                                    '488_power': float(channel_powers[1]),
                                                    '561_power': float(channel_powers[2]),
                                                    '635_power': float(channel_powers[3]),
                                                    '730_power': float(channel_powers[4])}]

                                save_name_metdata = save_directory / 'scan_metadata.csv'
                                data_io.write_metadata(scan_param_data[0], save_name_metdata)

                                setup_metadata=False

                            # save stage scan positions after each run. This allows for alignment of individual channels better
                            save_name_stage_positions = Path('r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_stage_positions.csv')
                            save_name_stage_positions = save_directory / save_name_stage_positions
                            data_io.write_metadata(current_stage_data[0], save_name_stage_positions)

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
                            
                            # copy current ryzc stage position metadata to NAS
                            save_directory_path = Path(save_directory)
                            remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
                            src= Path(save_directory) / Path(save_name_stage_positions.parts[2])
                            dst= Path(remote_directory) / Path(save_name_stage_positions.parts[2])
                            Thread(target=shutil.copy, args=[str(src), str(dst)]).start()

                    # clean up acquisition so that AcqEngJ releases directory.
                    acq = None
                            
                    # copy just acquired ryz data to NAS
                    save_directory_path = Path(save_directory)
                    remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
                    src= Path(save_directory) / Path(save_name_ryz+ '_1') 
                    dst= Path(remote_directory) / Path(save_name_ryz+ '_1') 
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

    # enable joystick
    core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
    core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()