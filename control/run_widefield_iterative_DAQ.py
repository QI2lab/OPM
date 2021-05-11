#!/usr/bin/env python

'''
Widefield bypass mode of ASU OPM with fluidics.

Shepherd 05/21 - modify for full automation
Shepherd 05/21 - modify acquisition control for widefield mode. Move fluidics helper function to own file.
Shepherd 05/21 - add in fluidics control. Will refactor once it works into separate files.
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
import pandas as pd
import data_io
import easygui

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

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',100000)
    time.sleep(1)

    # setup Photometrics BSI Express camera

    # set fan to HIGH

    # set temperature to -20

    # wait for temperature to reach -20

    # set readout mode to 11-bit "sensitive" and turn off all onboard pixel corrections
    
    # set output trigger to be HIGH when all rows are active ("rolling shutter")

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
    print('Number of labeling + imaging rounds: '+str(iterative_rounds))

    # iterate over user defined program
    for r_idx in range(iterative_rounds):

        # run fluidics program for this round
        success_fluidics = False
        success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
        if not(success_fluidics):
            print('Error in fluidics! Stopping scan.')
            sys.exit()

        # if first round, have user setup positions, laser intensities, and exposure time in MM GUI
        if r_idx == 0:
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
                save_directory=Path('E:/20210507cells/')
                save_name=Path( )

                # pull active lasers from MDA window
                channel_labels = ['405', '488', '561', '635', '730']
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
                do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel
                
                # pull laser powers from main window
                channel_powers = [5, 50, 90, 90, 0 ] # (0 -> 100%)
                
                # parse which channels are active
                active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
                n_active_channels = len(active_channel_indices)

                # turn all lasers on
                core.set_config('Laser','AllOff')
                core.wait_for_config('Laser','AllOff')
                            
                # setup lasers
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
                        if (stage_pos.get_stage_device_label() == 'XYStage'):
                            x_positions[idx] = stage_pos.x
                            y_positions[idx] = stage_pos.y
                        if (stage_pos.get_stage_device_label() == 'ZStage'):
                            z_positions[idx] = stage_pos.z

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
                core.snap_image()
                y_pixels = core.get_image_height()
                x_pixels = core.get_image_width()

                # construct imaging summary for user

                # display summary to user to make sure settings are correct
                easygui.textbox('Please review scan settings')

                run_imaging = easygui.ynbox('Run acquistion?', 'Title', ('Yes', 'No'))

                if run_imaging == True:
                    # set flag to change DAQ settings
                    config_changed = True

        # if last round, switch to DAPI + alexa488 readout instead
        if r_idx == (iterative_rounds - 1):

            # grab settings from MM
            # set up lasers and exposure times
            channel_states = [True,True,False,False,False]
            channel_powers[0] = 2

             # set flag to change DAQ settings
            config_changed = True

        # setup DAQ triggering
        # channels then stage trigger. Stage trigger goes high on last falling edge

        if config_changed:
            # setup DAQ 

            # setup relative move size for fast Z stepping

            # define event structure for each position
            events = []
            for z in range(z_positions):
                for c in range(n_active_channels):
                    evt = { 'axes': {'z': z, 'c': c}}
                    events.append(evt)

            config_changed = False

        for xyz_idx in range(xyz_positions): 

            # move to next XYZ position
            core.set_xy_position(xyz_positions[xyz_idx,0],xyz_positions[xyz_idx,1])
            core.wait_for_device(xy_stage)
            core.set_position(xyz_positions[xyz_idx,2])
            core.wait_for_device(z_stage)

            # run software autofocus at this postion on bead channel
            autofocus_offset = autofocus_method.full_focus()

            # move Z stage to defined distanced below determined autofocus position
            start_z_position = xyz_positions[xyz_idx,2] + autofocus_offset - z_relative_start 
            core.set_position(start_z_position)
            core.wait_for_device(z_stage)

            # save actual stage positions
            xy_pos = core.get_xy_stage_position()
            stage_x = xy_pos.x
            stage_y = xy_pos.y
            stage_z = core.get_position()
            current_stage_data = [{'stage_x': stage_x, 'stage_y': stage_y, 'stage_z': stage_z}]
            df_current_stage = pd.DataFrame(current_stage_data)

            # run hardware triggered stack for this XY position
            save_name_round_xyz = str(save_name)+'_r'+str(r_idx).zfill(4)+'_xyz'+str(xyz_idx).zfill(4)
            with Acquisition(directory=str(save_directory), name=save_name_round_xyz, show_display=False, max_multi_res_index=0,saving_queue_size=300) as acq:
                acq.acquire(events)
            acq = None

            # save stage scan positions after each tile
            save_name_stage_positions = Path('r'+str(r_idx).zfill(4)+'_xyz'+str(xyz_positions).zfill(4)+'_stage_positions.csv')
            save_name_stage_positions = save_directory / save_name_stage_positions
            data_io.write_metadata(save_name_stage_positions[0], save_name_stage_positions)

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
            '405_active': channel_states[0],
            '488_active': channel_states[1],
            '561_active': channel_states[2],
            '635_active': channel_states[3],
            '730_active': channel_states[4],
            '405_power': channel_powers[0],
            '488_power': channel_powers[1],
            '561_power': channel_powers[2],
            '635_power': channel_powers[3],
            '730_power': channel_powers[4]}]

        round_metadata_name = Path('r'+str(r_idx).zfill(4)+'_scan_metadata.csv')
        round_metadata_name = save_directory / round_metadata_name
        data_io.write_metadata(scan_param_data[0], round_metadata_name)

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()