#!/usr/bin/env python

'''
Widefield bypass mode of ASU OPM with fluidics.

Shepherd 08/21 - implement software autofocus
Shepherd 05/21 - modify for full automation and setup through GUI
Shepherd 05/21 - modify acquisition control for widefield mode. Move fluidics helper function to own file.
Shepherd 05/21 - add in fluidics control. Will refactor once it works into separate files.
Shepherd 04/21 - large-scale changes for new metadata and on-the-fly uploading to server for simultaneous reconstruction
'''

# imports
from pycromanager import Bridge, Acquisition
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP
from utils.fluidics_control import run_fluidic_program
from pathlib import Path
import numpy as np
import time
import sys
import pandas as pd
import utils.data_io as data_io
import easygui
import gc

def main():

    start_round_idx = 5 # fluidics round in file - 1 due to zero indexing
    start_tile_idx = 21

    # load fluidics program
    fluidics_path = easygui.fileopenbox('Load fluidics program')
    program_name = Path(fluidics_path)

    # check if user wants to flush system?
    run_type = easygui.choicebox('Type of run?', 'Iterative multiplexing setup', ['Flush fluidics (no imaging)', 'Iterative imaging'])
    if run_type == str('Flush fluidics (no imaging)'):
        flush_system = True
    else:
        flush_system = False

    # define ports for pumps and valves
    pump_COM_port = 'COM5'
    valve_COM_port = 'COM6'

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
    df_program = pd.read_csv(program_name)
    iterative_rounds = df_program['round'].max()
    print('Number of iterative rounds: '+str(iterative_rounds))

    if flush_system:
        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program(0, df_program, valve_controller, pump_controller)
        if not(success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()
        print('Flushed fluidic system.')
        sys.exit()

    # connect to Micromanager instance
    with Bridge() as bridge: 
        core = bridge.get_core()
    
        # get handle to xy and z stages
        xy_stage = core.get_xy_stage_device()
        z_stage = core.get_focus_device()

        # enable joystick
        core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
        core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')

        # turn off lasers
        core.set_config('Laser','Off')
        core.wait_for_config('Laser','Off')

        # change core timeout for long stage moves
        core.set_property('Core','TimeoutMs',100000)
        time.sleep(1)

    # iterate over user defined program
    for r_idx in range(start_round_idx,iterative_rounds):

        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
        if not(success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()

        # if first round, have user setup positions, laser intensities, and exposure time in MM GUI
        if r_idx == start_round_idx:
            
            # setup imaging parameters using MM GUI
            run_imaging = False
            while not(run_imaging):

                setup_done = False
                while not(setup_done):
                    setup_done = easygui.ynbox('Finished setting up MM?', 'Title', ('Yes', 'No'))

                with Bridge() as bridge:
                    core = bridge.get_core()
                    studio = bridge.get_studio()

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

                    # grab Z stack information
                    z_relative_start = acq_settings.slice_z_bottom_um()
                    z_relative_end = acq_settings.slice_z_top_um()
                    z_relative_step = acq_settings.slice_z_step_um()
                    number_z_positions = int(np.abs(z_relative_end-z_relative_start)/z_relative_step + 1)

                    # populate XYZ array
                    xyz_positions = np.empty([number_positions,3])
                    xyz_positions[:,0]= x_positions
                    xyz_positions[:,1]= y_positions
                    xyz_positions[:,2]= z_positions
                    
                    # grab autofocus information
                    autofocus_manager = studio.get_autofocus_manager()
                    autofocus_method = autofocus_manager.get_autofocus_method()

                    # camera pixel size
                    camera_pixel_size_um = 6.5 # unit: um
                    objective_mag = 100
                    pixel_size_um = camera_pixel_size_um/objective_mag # unit: um 

                    # get exposure time from main window
                    exposure_ms = core.get_exposure()

                    # enforce exposure time
                    core.set_exposure(exposure_ms)

                    # determine image size
                    #core.snap_image()
                    y_pixels = 2048
                    x_pixels = 2048

                    # construct imaging summary for user
                    scan_settings = (f"Number of labeling rounds: {str(iterative_rounds)} \n"
                                    f"Number of XY tiles:  {str(number_positions)} \n"
                                    f"Number of Z slices:  {str(number_z_positions)} \n"
                                    f"Number of channels:  {str(n_active_channels)} \n"
                                    f"Active lasers: {str(channel_states)} \n"
                                    f"Lasers powers: {str(channel_powers)} \n"
                                    f"Z stage start: {str(z_relative_start)} \n"
                                    f"Z stage end: {str(z_relative_end)} \n"
                                    f"Z step: {str(z_relative_step)} \n")

                    # display summary to user to make sure settings are correct
                    easygui.textbox(scan_settings,'Please review scan settings')

                    run_imaging = easygui.ynbox('Run acquistion?', 'Title', ('Yes', 'No'))

                    if run_imaging == True:
                        # disable joystick
                        core.set_property('XYStage:XY:31','JoystickEnabled','No')
                        core.set_property('ZStage:M:37','JoystickInput','0 - none')
                        
                        # set flag to change DAQ settings
                        config_changed = True

        # if last round, switch to DAPI + alexa488 readout instead
        if r_idx == (iterative_rounds - 1):

            # grab settings from MM
            # set up lasers and exposure times
            channel_states = [True,True,False,False,False]
            channel_powers[0] = 10
            channel_powers[1] = channel_powers[1]
            channel_powers[2] = 0
            channel_powers[3] = 0
            channel_powers[4] = 0

             # set flag to change DAQ settings
            config_changed = True

        if config_changed:

            with Bridge() as bridge:
                core = bridge.get_core()

                # turn all lasers on
                core.set_config('Laser','Off')
                core.wait_for_config('Laser','Off')
                
                # set lasers to user defined power
                core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
                core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
                core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
                core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
                core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

                core.set_config('Modulation-730','External-Digital')
                core.wait_for_config('Modulation-730','External-Digital')

                config_changed = False

        if r_idx == start_round_idx:
            resume_xyz_idx = start_tile_idx
        else:
            resume_xyz_idx = 0

        for xyz_idx in range(resume_xyz_idx,number_positions): 

            with Bridge() as bridge:
                core = bridge.get_core()

                # move to next XYZ position
                core.set_xy_position(xyz_positions[xyz_idx,0],xyz_positions[xyz_idx,1])
                core.wait_for_device(xy_stage)
                core.set_position(xyz_positions[xyz_idx,2])
                core.wait_for_device(z_stage)

                # run software autofocus at this postion on bead channel
                #autofocus_offset = autofocus_method.full_focus()
                #print(autofocus_offset)
                
                # move Z stage to defined distanced below determined autofocus position
                #start_z_position = xyz_positions[xyz_idx,2] + autofocus_offset + z_relative_start
                #core.set_position(start_z_position)
                #core.wait_for_device(z_stage)

                # save actual stage positions
                xy_pos = core.get_xy_stage_position()
                stage_x = xy_pos.x
                stage_y = xy_pos.y
                stage_z = core.get_position()
                current_stage_data = [{'stage_x': stage_x, 
                                        'stage_y': stage_y, 
                                        'stage_z': stage_z}]

                # define event structure for each position
                events = []
                for z in range(number_z_positions):
                    for c in range(len(channel_states)):
                        if channel_states[c]:
                            evt = { 'axes': {'z': z, 'c': c}, 
                                    'z': stage_z+z_relative_start+(z*z_relative_step), 
                                    'channel': {'group': 'Laser',
                                                'config': channel_labels[c]}}
                            events.append(evt)

                # run stack for this XY position
                if r_idx == start_round_idx and xyz_idx == resume_xyz_idx:
                    save_name_round_xyz = str(save_name)+'_r'+str(r_idx).zfill(4)+'_xyz'+str(xyz_idx).zfill(4)+'a'
                else:
                    save_name_round_xyz = str(save_name)+'_r'+str(r_idx).zfill(4)+'_xyz'+str(xyz_idx).zfill(4)
                
                with Acquisition(directory=str(save_directory), name=save_name_round_xyz, show_display=False, max_multi_res_index=0,saving_queue_size=5000) as acq:
                    acq.acquire(events)

                acq = None

                # turn all lasers off
                core.set_config('Laser','Off')
                core.wait_for_config('Laser','Off')

                time.sleep(2)
                acq_deleted = False
                while (acq_deleted):
                    try:
                        del acq
                    except:
                        acq_deleted = False
                        time.sleep(2)
                    else:
                        acq_deleted = True
                        gc.collect()


                # save stage scan positions after each tile
                save_name_stage_positions = Path('r'+str(r_idx).zfill(4)+'_xyz'+str(xyz_idx).zfill(4)+'_stage_positions.csv')
                save_name_stage_positions = save_directory / save_name_stage_positions
                data_io.write_metadata(current_stage_data[0], save_name_stage_positions)

        with Bridge() as bridge:
            core = bridge.get_core()

            # save metadata for this round
            scan_param_data = [{'root_name': str(save_name),
                'scan_type': str('iterative-widefield'),
                'theta': str(''), 
                'scan_step_um': str(''), 
                'pixel_size_um': float(pixel_size_um),
                'axial_step_um': float(z_relative_step),
                'y_pixels': int(y_pixels),
                'x_pixels': int(x_pixels),
                'exposure_ms': float(exposure_ms),
                'camera': str('BSI_Express'),
                'readout_mode': str('11bit_sensitive'),
                'num_r': int(iterative_rounds),
                'num_y': int(number_positions),
                'num_z': int(number_z_positions),
                'num_ch': int(n_active_channels),
                'scan_axis_positions': str(''),
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

            round_metadata_name = Path('r'+str(r_idx).zfill(4)+'_scan_metadata.csv')
            round_metadata_name = save_directory / round_metadata_name
            data_io.write_metadata(scan_param_data[0], round_metadata_name)

            # move to intial XYZ position
            core.set_xy_position(xyz_positions[0,0],xyz_positions[0,1])
            core.wait_for_device(xy_stage)
            core.set_position(xyz_positions[0,2])
            core.wait_for_device(z_stage)

    with Bridge() as bridge:

        core = bridge.get_core()

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
        core.set_config('Modulation-405','CW (constant power)')
        core.wait_for_config('Modulation-405','CW (constant power)')
        core.set_config('Modulation-488','CW (constant power)')
        core.wait_for_config('Modulation-488','CW (constant power)')
        core.set_config('Modulation-561','CW (constant power)')
        core.wait_for_config('Modulation-561','CW (constant power)')
        core.set_config('Modulation-637','CW (constant power)')
        core.wait_for_config('Modulation-637','CW (constant power)')
        core.set_config('Modulation-730','CW (constant power)')
        core.wait_for_config('Modulation-730','CW (constant power)')

        # enable joystick
        core.set_property('XYStage:XY:31','JoystickEnabled','Yes')
        core.set_property('ZStage:M:37','JoystickInput','22 - right wheel')
        
#-----------------------------------------------------------------------------
    print('Finished.')

if __name__ == "__main__":
    main()