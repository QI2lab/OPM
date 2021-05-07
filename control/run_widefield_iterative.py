#!/usr/bin/env python

'''
Widefield bypass mode of ASU OPM with fluidics.

Shepherd 05/21 - modify acquisition control for widefield mode. Move fluidics helper function to own file.
Shepherd 05/21 - add in fluidics control. Will refactor once it works into separate files.
Shepherd 04/21 - large-scale changes for new metadata and on-the-fly uploading to server for simultaneous reconstruction
'''

# imports
from pycromanager import Bridge, Acquisition, multi_d_acquisition_events
from hardware.APump import APump
from hardware.HamiltonMVP import HamiltonMVP
from fluidics.FluidicsControl import lookup_valve, run_fluidic_program
from pathlib import Path
import numpy as np
import time
import sys
import pandas as pd
import data_io

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # set up lasers
    channel_labels = ["488", "561", "635"]
    channel_powers = [0, 10, 75, 75, 0 ] # (0 -> 100%)
    channel_exposures_ms [100, 100, 100]
    do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

    x = []
    y = []
    z = []
    xy = np.hstack([x[:, None], y[:, None], z[:, None]])

    # setup file name
    save_directory=Path('E:/20210507cells/')
    program_name = Path('D:/20210507_afterfive.csv')
    save_name = 'sina_cellculture_trial002'

    run_fluidics = True
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

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',100000)
    time.sleep(1)
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
        #valve_controller.autoDetectValves()

        df_program = pd.read_csv(program_name)
        iterative_rounds = df_program['round'].max()

    else:
        iterative_rounds = 1

    # output experiment info
    print('Number of labeling rounds: '+str(iterative_rounds))
    print('Number of XY positions: '+str(height_axis_positions))
    print('Number of channels: '+str(n_active_channels))
    print('Number of Z positions: '+str(tile_axis_positions))


    events = multi_d_acquisition_events(xyz_positions=xyz, 
                                        z_start=-10.0, z_end=10.0, z_step=0.25,
                                        channels="Laser",
                                        channel_group=channel_labels,
                                        channel_exposures_ms=channel_exposures_ms,
                                        order='pzc')
    
    for r_idx in range(iterative_rounds):

        if run_fluidics == True:
            success_fluidics = False
            success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
            if not(success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()

        if run_scope == True:

            with Acquisition(directory='/path/to/saving/dir', name='acquisition_name') as acq:
                acq.acquire(events)

            acq = None
            
    # turn all lasers off
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()