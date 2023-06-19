#!/usr/bin/env python

'''
ASU high-NA OPM multi-channel stage scanning with iterative fluidics using pycromanager.

Instrument rebuild (11/2022) TO DO (in order of priority):
1. Fix "last round" code
  - find cleaner way to note in the instrument setup and metadata so that this is easier to process
2. ETL control. Device works in MM - can MM MDA stage widget record and then we grab during setup?
3. Write out instrument configuration (.json instead of .csv?)
  - stage and ETL positions
  - laser powers
  - camera exposure
  - fluidics program path
  - what else?
4. Create "metadata" and "instrument" directories to keep file structure cleaner 
5. Find a better way to resume a broken acquisition
  - in addition to instrument setup on disk, update file with last succesful tile?

Shepherd 01/23 - breaking change -> v2.0. Scans are now split up into smaller chunks to account for small coverslip tilt on our setup.
                 this should help solve coverslip tilt issue and make registration easier on the deskewed data (more flexibility for tile positioning)
Shepherd 12/22 - hack together O3 focus changing to account for RI mismatch causing defocus during z stacks.
Shepherd 11/22 - many changes to support instrument being rebuilt in new lab space.
Shepherd 08/21 - refactor for easier reading, O2-O3 autofocus, and managing file transfer in seperate Python process
Shepherd 07/21 - switch to interleaved excitation during stage scan and initial work on O2-O3 autofocusing
Shepherd 06/21 - clean up code and bring inline with widefield bypass GUI code
Shepherd 05/21 - pull all settings from MM GUI and prompt user to setup experiment using "easygui" package
Shepherd 05/21 - add in fluidics control. Recently refactored into seaprate files
Shepherd 04/21 - large-scale changes for new metadata and on-the-fly uploading to server for simultaneous reconstruction
'''

# imports
# qi2lab fluidics and autfocus shutter controllers
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP
from hardware.PicardShutter import PicardShutter

# qi2lab OPM stage scan control functions for pycromanager
from utils.data_io import read_config_file, read_fluidics_program, write_metadata, time_stamp, append_index_filepath
from utils.fluidics_control import run_fluidic_program
from utils.opm_setup_v2 import setup_asi_tiger, setup_obis_laser_boxx, camera_hook_fn, retrieve_setup_from_MM
from utils.autofocus_remote_unit import manage_O3_focus, apply_O3_focus_offset

# pycromanager
from pycromanager import Core, Acquisition, Studio

# NI DAQ control
import PyDAQmx as daq
import ctypes as ct

# python libraries
import time
import sys
import gc
from pathlib import Path
import numpy as np
from functools import partial
import easygui

def main():
    """"
    Execute iterative, interleaved OPM stage scan using MM GUI
    """
    # flags for resuming acquisition
    # TO DO: Make this a user setting during acquisition so that it can't screw up a restart.
    resume_x_tile = 0
    resume_y_tile = 0

    # O3 offset per plane
    O3_offset_per_z = -3.5 # DOUBLE CHECK BEFORE RUNNING!
    # This is for 30% overlap of 30 um ROI (512)

    # flags for metadata, processing, drift correction, and O2-O3 autofocusing
    setup_metadata=True
    debug_flag = True
    switch_last_round = True
    avoid_overwriting = True

    # check if user wants to flush system?
    run_fluidics = False
    flush_system = False
    run_type = easygui.choicebox('Type of run?', 'OPM setup', ['Run fluidics (no imaging)', 'Iterative imaging', 'Single round (test)'])
    if run_type == str('Run fluidics (no imaging)'):
        flush_system = True
        run_fluidics = True
        # load fluidics program
        fluidics_path = easygui.fileopenbox('Load fluidics program')
        program_name = Path(fluidics_path)
    elif run_type == str('Iterative imaging'):
        flush_system = False
        run_fluidics = True
        # load fluidics program
        fluidics_path = easygui.fileopenbox('Load fluidics program')
        program_name = Path(fluidics_path)
    elif run_type == str('Single round (test)'):
        flush_system = False
        run_fluidics = False
        n_iterative_rounds = 1
        fluidics_rounds= [1]

    file_directory = Path(__file__).resolve().parent
    config_file = file_directory / Path('opm_config.csv')
    df_config = read_config_file(config_file)

    if run_fluidics:
        # define ports for pumps and valves
        pump_COM_port = str(df_config['pump_com_port'])
        valve_COM_port = str(df_config['valve_com_port'])

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
        df_program = read_fluidics_program(program_name)
        fluidics_rounds = df_program["round"].unique()
        print('Will execute fluidics rounds:', fluidics_rounds)
        # When resuming fluidics experiments, the max round name and number can differ
        max_round_name = df_program["round"].max()
        n_iterative_rounds = len(fluidics_rounds)
        print('Number of iterative rounds: '+str(n_iterative_rounds))
        if max_round_name != n_iterative_rounds:
            print(f"Max round label is {max_round_name}")

    if flush_system:
        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program(1, df_program, valve_controller, pump_controller, core=None,shutter_controller=None,O3_stage_name=None)
        if not(success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()
        print('Flushed fluidic system.')
        sys.exit()

    # connect to alignment laser shutter
    shutter_id = int(df_config['shutter_id'])
    shutter_controller = PicardShutter(shutter_id=shutter_id,verbose=False)
    shutter_controller.closeShutter()
    
    # setup O3 piezo stage
    O3_stage_name = str(df_config['O3_stage_name'])

    # connect to Micromanager core instance
    core = Core()

    core.set_focus_device("ZStage:M:37")
 
    # make sure camera does not have an ROI set
    core.snap_image()
    y_pixels = core.get_image_height()
    x_pixels = core.get_image_width()
    while not(y_pixels==2304) or not(x_pixels==2304):
        roi_reset = False
        while not(roi_reset):
            roi_reset = easygui.ynbox('Removed camera ROI?', 'Title', ('Yes', 'No'))
        core.snap_image()
        y_pixels = core.get_image_height()
        x_pixels = core.get_image_width()
    
    # set ROI
    roi_selection = easygui.choicebox('Imaging volume setup.', 'ROI size', ['256x1700', '512x1700', '1024x1700'])
    if roi_selection == str('256x1700'):
        roi_y_corner = 928
        roi_x_corner = 368
        roi_imaging = [roi_x_corner,roi_y_corner,1700,256]
        y_pixels = 256
        x_pixels = 1700
        core.set_roi(*roi_imaging)
    elif roi_selection == str('512x1700'):
        roi_y_corner = 928
        roi_x_corner = 368
        roi_imaging = [roi_x_corner,roi_y_corner,1700,512]
        y_pixels = 512
        x_pixels = 1700
        core.set_roi(*roi_imaging)
    elif roi_selection == str('1024x1700'):
        roi_y_corner = 928
        roi_x_corner = 368
        roi_imaging = [roi_x_corner,roi_y_corner,1700,1024]
        y_pixels = 1024
        x_pixels = 1700
        core.set_roi(*roi_imaging)
    
    # set lasers to zero power and software control
    channel_powers = [0.,0.,0.,0.,0.]
    setup_obis_laser_boxx(core,channel_powers,state='software')

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
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[0]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[1]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[2]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[0]','POSITIVE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[1]','POSITIVE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[2]','POSITIVE')

    # enable joystick
    core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
    core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',500000)
    time.sleep(1)

    # galvo voltage at neutral
    galvo_neutral_volt = 0.0 # unit: volts

    # pull galvo line from config file
    galvo_ao_line = str(df_config['galvo_ao_pin'])

    # set the galvo to the neutral position if it is not already
    try: 
        taskAO_first = daq.Task()
        taskAO_first.CreateAOVoltageChan(galvo_ao_line,"",-4.0,4.0,daq.DAQmx_Val_Volts,None)
        taskAO_first.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
        taskAO_first.StopTask()
        taskAO_first.ClearTask()
    except daq.DAQError as err:
        print("DAQmx Error %s"%err)

    gc.collect()

    # iterate over user defined program
    for r_idx, r_name in enumerate(fluidics_rounds):
 
        #studio = bridge.get_studio()
        studio = Studio()
        
        # get handle to xy and z stages
        xy_stage = core.get_xy_stage_device()
        z_stage = core.get_focus_device()

        # run O3 focus optimizer
        O3_focus_position = np.round(manage_O3_focus(core,shutter_controller,O3_stage_name,verbose=False),2)
        print(time_stamp(), f'O3 focus stage position (um) = {O3_focus_position}.')

        time_now = time.perf_counter()

        if run_fluidics:
            success_fluidics = False
            success_fluidics = run_fluidic_program(r_name, df_program, valve_controller, pump_controller,core,shutter_controller,O3_stage_name)
            if not(success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()
        
            O3_focus_position = np.round(manage_O3_focus(core,shutter_controller,O3_stage_name,verbose=False),2)
            print(time_stamp(), f'O3 focus stage position (um) = {O3_focus_position}.')

        # if first round, have user setup positions, laser intensities, and exposure time in MM GUI
        if r_idx == 0:
            
            # setup imaging parameters using MM GUI
            run_imaging = False
            while not(run_imaging):

                setup_done = False
                while not(setup_done):
                    setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

                df_MM_setup, active_channel_indices, scan_axis_start_pos_mm, scan_axis_end_pos_mm, height_axis_start_pos_um, tile_axis_pos_um = retrieve_setup_from_MM(core,studio,df_config,debug=debug_flag)

                channel_states = [bool(df_MM_setup['405_active']),
                                bool(df_MM_setup['488_active']),
                                bool(df_MM_setup['561_active']),
                                bool(df_MM_setup['635_active']),
                                bool(df_MM_setup['730_active'])]

                channel_powers = [float(df_MM_setup['405_power']),
                                float(df_MM_setup['488_power']),
                                float(df_MM_setup['561_power']),
                                float(df_MM_setup['635_power']),
                                float(df_MM_setup['730_power'])]

                z_stack_height_input = easygui.integerbox(msg = "Z stack height above coverslip", default =0, lowerbound=0,upperbound=150)
                if z_stack_height_input == 0:
                    z_stack = False
                    z_stack_height_um = 0.0
                    z_stack_offset_um = [0]
                else:
                    z_stack = True
                    z_stack_height_um = float(z_stack_height_input) #unit: um
                    z_axis_overlap=0.1 #unit: percentage
                    z_axis_ROI = y_pixels*.115*np.sin(30.*np.pi/180.) #unit: um  
                    z_axis_step_um = np.round((z_axis_ROI)*(1-z_axis_overlap),2) #unit: um
                    z_stack_offset_um = np.round(np.arange(0,z_stack_height_um,z_axis_step_um),2) #unit: um

                    print(f'Z axis ROI: {z_axis_ROI}')
                    print(f'Z axis height: {z_stack_height_input}')
                    print(f'Z axis step: {z_axis_step_um}')
                    print(f'Z axis offset: {z_stack_offset_um}')

                # construct and display imaging summary to user
                scan_settings = (f"Number of labeling rounds: {str(n_iterative_rounds)} \n\n"
                                f"Number of Y tiles:  {str(df_MM_setup['tile_axis_positions'])} \n"
                                f"Tile start:  {str(df_MM_setup['tile_axis_start_um'])} \n"
                                f"Tile end:  {str(df_MM_setup['tile_axis_end_um'])} \n\n"
                                f"Coverslip start:  {str(df_MM_setup['coverslip_axis_start_um'])} \n"
                                f"Coverslip end:  {str(df_MM_setup['coverslip_axis_end_um'])} \n\n"
                                f"Acquiring z stack: {str(z_stack)} \n"
                                f"Height of z stack above coverslip: {str(z_stack_height_um)} \n"
                                f"Number of Z slabs above coverslip: {str(len(z_stack_offset_um))} \n\n"
                                f"Number of channels:  {str(df_MM_setup['n_active_channels'])} \n"
                                f"Active lasers: {str(channel_states)} \n"
                                f"Lasers powers: {str(channel_powers)} \n\n"
                                f"Number of scan axis tiles:  {str(df_MM_setup['scan_axis_tile_positions'])} \n"
                                f"Number of scan axis positions per tile:  {str(df_MM_setup['scan_axis_positions'])} \n"
                                f"Scan start: {str(df_MM_setup['scan_axis_start_um'])}  \n"
                                f"Scan end:  {str(df_MM_setup['scan_axis_end_um'])} \n")
                                
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
        if switch_last_round and (r_idx == (n_iterative_rounds - 1)) and (run_fluidics):

            # enable joystick
            core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
            core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')

            setup_done = False
            while not(setup_done):
                setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

            df_MM_setup, active_channel_indices, _, _, _, _ = retrieve_setup_from_MM(core,studio,df_config,debug=debug_flag)

            channel_states = [bool(df_MM_setup['405_active']),
                                bool(df_MM_setup['488_active']),
                                bool(df_MM_setup['561_active']),
                                bool(df_MM_setup['635_active']),
                                bool(df_MM_setup['730_active'])]

            channel_powers = [float(df_MM_setup['405_power']),
                                float(df_MM_setup['488_power']),
                                float(df_MM_setup['561_power']),
                                float(df_MM_setup['635_power']),
                                float(df_MM_setup['730_power'])]

            # construct and display imaging summary to user
            scan_settings = (f"Number of labeling rounds: {str(n_iterative_rounds)} \n\n"
                            f"Number of Y tiles:  {str(df_MM_setup['tile_axis_positions'])} \n"
                            f"Tile start:  {str(df_MM_setup['tile_axis_start_um'])} \n"
                            f"Tile end:  {str(df_MM_setup['tile_axis_end_um'])} \n\n"
                            f"Coverslip start:  {str(df_MM_setup['coverslip_axis_start_um'])} \n"
                            f"Coverslip end:  {str(df_MM_setup['coverslip_axis_end_um'])} \n\n"
                            f"Acquiring z stack: {str(z_stack)} \n"
                            f"Height of z stack above coverslip: {str(z_stack_height_um)} \n"
                            f"Number of Z slabs above coverslip: {str(len(z_stack_offset_um))} \n\n"
                            f"Number of channels:  {str(df_MM_setup['n_active_channels'])} \n"
                            f"Active lasers: {str(channel_states)} \n"
                            f"Lasers powers: {str(channel_powers)} \n\n"
                            f"Number of scan axis tiles:  {str(df_MM_setup['scan_axis_tile_positions'])} \n"
                            f"Number of scan axis positions per tile:  {str(df_MM_setup['scan_axis_positions'])} \n"
                            f"Scan start: {str(df_MM_setup['scan_axis_start_um'])}  \n"
                            f"Scan end:  {str(df_MM_setup['scan_axis_end_um'])} \n")
                            
            output = easygui.textbox(scan_settings, 'Please review last round scan settings')

            # verify user actually wants to run imaging
            run_imaging = easygui.ynbox('Run acquistion of last round?', 'Title', ('Yes', 'No'))

            if run_imaging == True:
                # disable joystick
                core.set_property('XYStage:XY:31','JoystickEnabled','No')
                core.set_property('ZStage:M:37','JoystickInput','0 - none')
                
                # set flag to change configuration
                config_changed = True

        if config_changed:

            # turn on 'transmit repeated commands' for Tiger
            core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

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

            core.set_xy_position(float(df_MM_setup['scan_axis_start_um']),float(df_MM_setup['tile_axis_start_um']))
            core.wait_for_device(xy_stage)

            # set lasers to external digital control
            setup_obis_laser_boxx(core,channel_powers,state='digital')
            
            # create events to execute scan
            # DPS: trial new event structure to disambiguate excess stage positions and channel names
            events = []
            channel_names = ['ch405','ch488','ch561','ch635','ch730']
            for excess_scan_idx in range(int(df_config['excess_scan_positions'])):
                for c in active_channel_indices:
                    evt = { 'axes': {'s': 0, 'channel': str(channel_names[int(c)]), 'e': excess_scan_idx}}
                    events.append(evt)
            for scan_idx in range(int(df_MM_setup['scan_axis_positions'])):
                for c in active_channel_indices:
                    evt = { 'axes': {'s': scan_idx,'channel': str(channel_names[int(c)]), 'e': (int(df_config['excess_scan_positions'])+1)}}
                    events.append(evt)

            # setup digital trigger buffer on DAQ
            n_active_channels = int(df_MM_setup['n_active_channels'])
            samples_per_ch = 2 * n_active_channels
            DAQ_sample_rate_Hz = 10000
            num_DI_channels = 8

            # create DAQ pattern for laser strobing controlled via rolling shutter
            dataDO = np.zeros((samples_per_ch, num_DI_channels), dtype=np.uint8)
            for ii, ind in enumerate(active_channel_indices):
                dataDO[2*ii::2*n_active_channels, int(ind)] = 1

        # set camera to internal control
        core.set_config('Camera-TriggerSource','INTERNAL')
        core.wait_for_config('Camera-TriggerSource','INTERNAL')

        gc.collect()

        n_x_tiles = int(df_MM_setup['scan_axis_tile_positions'])
        n_y_tiles = int(df_MM_setup['tile_axis_positions'])
        n_z_tiles = int(len(z_stack_offset_um))

        for x_idx in range(resume_x_tile,n_x_tiles):

            scan_axis_start_mm = scan_axis_start_pos_mm[x_idx]
            scan_axis_end_mm = scan_axis_end_pos_mm[x_idx]
            height_position_start_um = height_axis_start_pos_um[x_idx]

            for y_idx in range(resume_y_tile,n_y_tiles):
            
                # calculate tile axis position
                tile_position_um = tile_axis_pos_um[y_idx]
                
                # change X stage speed for move
                # set scan axis speed for large move to initial position
                command = 'SPEED X=.1'
                core.set_property('TigerCommHub','SerialCommand',command)

                # move XY stage to new tile axis position
                core.set_xy_position(float(scan_axis_start_mm*1000.),tile_position_um)
                core.wait_for_device(xy_stage)

                # setup constant speed stage scanning on ASI Tiger controller
                setup_asi_tiger(core,float(df_MM_setup['scan_axis_speed']),scan_axis_start_mm,scan_axis_end_mm)

                for z_idx in range(n_z_tiles):

                    if z_stack:
                        height_position_um = height_position_start_um + z_stack_offset_um[z_idx]
                    else:
                        height_position_um = height_position_start_um

                    # move Z stage to new height axis position
                    core.set_focus_device(z_stage)
                    core.wait_for_device(z_stage)
                    core.set_position(height_position_um)
                    core.wait_for_device(z_stage)

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
                    
                    # update save_name with current tile information
                    save_name_rxyz = Path(str(df_MM_setup['save_name'])+'_r'+str(r_name).zfill(4)+'_x'+str(x_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4))
                    if avoid_overwriting:
                        save_name_rxyz = append_index_filepath(save_name_rxyz)

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

                    # query current stage positions
                    xy_pos = core.get_xy_stage_position()
                    stage_x = np.round(float(xy_pos.x),2)
                    stage_y = np.round(float(xy_pos.y),2)
                    stage_z = np.round(float(core.get_position()),2)

                    # run O3 focus optimizer
                    O3_focus_position = np.round(manage_O3_focus(core,shutter_controller,O3_stage_name,verbose=False),2)
                    print(time_stamp(), f'O3 focus stage position (um) = {O3_focus_position}.')

                    # move O3 to correct position to account for RI mismatch
                    if z_stack:
                       O3_focus_offset = z_idx * O3_offset_per_z
                       offset_O3_focus_position = apply_O3_focus_offset(core,O3_stage_name,O3_focus_position,O3_focus_offset,verbose=False)
                       print(time_stamp(), f'Offset O3 focus stage position (um) = {offset_O3_focus_position}.')
                    
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

                    # turn all lasers on
                    core.set_config('Laser','AllOn')
                    core.wait_for_config('Laser','AllOn')

                    # set camera to external control
                    # DCAM sets the camera back to INTERNAL mode after each acquisition
                    core.set_config('Camera-TriggerSource','EXTERNAL')
                    core.wait_for_config('Camera-TriggerSource','EXTERNAL')

                    # verify that camera actually switched back to external trigger mode
                    trigger_state = core.get_property('OrcaFusionBT','TRIGGER SOURCE')

                    # if not in external control, keep trying until camera changes settings
                    while not(trigger_state =='EXTERNAL'):
                        time.sleep(2.0)
                        core.set_config('Camera-TriggerSource','EXTERNAL')
                        core.wait_for_config('Camera-TriggerSource','EXTERNAL')
                        trigger_state = core.get_property('OrcaFusionBT','TRIGGER SOURCE')

                    # create stage position dictionary 
                    current_stage_data = [{'stage_x': float(stage_x), 
                                        'stage_y': float(stage_y), 
                                        'stage_z': float(stage_z),
                                        'stage_O3': float(O3_focus_position),
                                        '405_active': bool(channel_states[0]),
                                        '488_active': bool(channel_states[1]),
                                        '561_active': bool(channel_states[2]),
                                        '635_active': bool(channel_states[3]),
                                        '730_active': bool(channel_states[4])}]

                    # run acquisition for this rxyz combination
                    print(time_stamp(), f'round {r_idx+1}/{n_iterative_rounds}; x tile {x_idx+1}/{n_x_tiles}; y tile {y_idx+1}/{n_y_tiles}; z tile {z_idx+1}/{n_z_tiles}.')
                    print(time_stamp(), f'Stage location (um): x={stage_x}, y={stage_y}, z={stage_z}.')
                    with Acquisition(directory=str(df_MM_setup['save_directory']), name=str(save_name_rxyz),
                                    post_camera_hook_fn=camera_hook_fn, show_display=False) as acq:
                        acq.acquire(events)
                    acq = None
                    del acq
                    gc.collect()

                    # stop DAQ and make sure it is at zero
                    try:
                        ## Stop and clear both tasks
                        taskDI.StopTask()
                        taskDO.StopTask()
                        taskDI.ClearTask()
                        taskDO.ClearTask()
                    except daq.DAQError as err:
                        print("DAQmx Error %s"%err)

                    core.set_config('Laser','Off')
                    core.wait_for_config('Laser','Off')

                    # move O3 back to "real" focus
                    offset_O3_focus_position = apply_O3_focus_offset(core,O3_stage_name,O3_focus_position,0.0,verbose=False)
                    print(time_stamp(), f'O3 focus stage position (um) = {offset_O3_focus_position}.')

                    # save experimental info after first tile. 
                    # we do it this way so that Pycromanager can manage directory creation
                    if (setup_metadata):
                        # save stage scan parameters
                        scan_param_data = [{'root_name': str(df_MM_setup['save_name']),
                                            'scan_type': str('OPM-stage-v3'),
                                            'interleaved': bool(True),
                                            'exposure': float(df_MM_setup['exposure_ms']),
                                            'theta': float(30.0), 
                                            'scan_step': float(float(df_config['scan_axis_step_um'])*1000.), 
                                            'pixel_size': float(float(df_config['pixel_size'])*1000.),
                                            'num_t': int(1),
                                            'num_r': int(n_iterative_rounds),
                                            'num_x': int(n_x_tiles),
                                            'num_y': int(n_y_tiles),
                                            'num_z': int(n_z_tiles),
                                            'num_ch': int(df_MM_setup['n_active_channels']),
                                            'scan_axis_positions': int(df_MM_setup['scan_axis_positions']),
                                            'excess_scan_positions': int(df_config['excess_scan_positions']),
                                            'y_pixels': int(df_MM_setup['y_pixels']),
                                            'x_pixels': int(df_MM_setup['x_pixels']),
                                            '405_active': bool(channel_states[0]),
                                            '488_active': bool(channel_states[1]),
                                            '561_active': bool(channel_states[2]),
                                            '635_active': bool(channel_states[3]),
                                            '730_active': bool(channel_states[4]),
                                            '405_power': float(channel_powers[0]),
                                            '488_power': float(channel_powers[1]),
                                            '561_power': float(channel_powers[2]),
                                            '635_power': float(channel_powers[3]),
                                            '730_power': float(channel_powers[4])}]
                        scan_metadata_path = Path(df_MM_setup['save_directory']) / 'scan_metadata.csv'
                        if avoid_overwriting:
                            scan_metadata_path = append_index_filepath(scan_metadata_path)
                        write_metadata(scan_param_data[0], scan_metadata_path)

                        setup_metadata=False

                    # save stage scan positions after each tile
                    save_name_stage_positions = Path(str(df_MM_setup['save_name'])+'_r'+str(r_name).zfill(4)+'_x'+str(x_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_stage_positions.csv')
                    save_name_stage_path = Path(df_MM_setup['save_directory']) / save_name_stage_positions
                    if avoid_overwriting:
                        save_name_stage_path = append_index_filepath(save_name_stage_path)
                    write_metadata(current_stage_data[0], save_name_stage_path)

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

                    gc.collect()
            resume_y_tile = 0
        resume_x_tile = 0

        # run empty acquisition to close file handle to previous acquistion
        # need to do this because we process and delete raw data on the fly
        events_flush = None
        with Acquisition(directory=None,name=None,show_display=False) as acq_flush:
            acq_flush.acquire(events_flush)
        acq_flush = None
        del acq_flush
        gc.collect()
        

    # set lasers to zero power and software control
    channel_powers = [0.,0.,0.,0.,0.]
    setup_obis_laser_boxx(core,channel_powers,state='software')
    
    # set camera to internal control
    core.set_config('Camera-TriggerSource','INTERNAL')
    core.wait_for_config('Camera-TriggerSource','INTERNAL')           

    # enable joystick
    core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
    core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')

    # shut down python initialized hardware
    if (run_fluidics):
        # shutter_controller.close()
        valve_controller.close()
        pump_controller.close()
    shutter_controller.shutDown()

    del core, studio
    gc.collect()
    
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    main()