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

    # setup file name
    save_directory=Path('E:/20210507cells/')
    program_name = Path('D:/20210507_dapi.csv')
    save_name = 'sina_cellculture_trial002'

    run_fluidics = False
    run_scope = True

    if run_scope == True:
        # set up lasers
        channel_labels = ["405","488"]
        channel_powers = [5, 50, 90, 90, 0 ] # (0 -> 100%)
        channel_exposures_ms = [10, 100]
        
        xy = np.empty([9,2]).astype(float)
        xy[0:3,0]=9500
        xy[3:6,0]=9650
        xy[6:9,0]=9800
        xy[[0,5,6],1]=-4700
        xy[[1,4,7],1]=-4850
        xy[[2,3,8],1]=-5000

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

    if run_scope == True:
        # output experiment info
        print('Number of labeling rounds: '+str(iterative_rounds))
        print('Number of XY positions: '+str(xy.shape[0]))

        events = multi_d_acquisition_events(xy_positions=xy, 
                                            z_start=15974, z_end=15989, z_step=0.25,
                                            channels=channel_labels,
                                            channel_group="Laser",
                                            channel_exposures_ms=channel_exposures_ms,
                                            order='pzc')
    
    for r_idx in range(8,9):

        if run_fluidics == True:
            success_fluidics = False
            success_fluidics = run_fluidic_program(r_idx, df_program, valve_controller, pump_controller)
            if not(success_fluidics):
                print('Error in fluidics! Stopping scan.')
                sys.exit()

        if run_scope == True:

            save_name_round = str(save_name)+'_r'+str(r_idx).zfill(4)

            with Acquisition(directory=str(save_directory), name=save_name_round,show_display=True, max_multi_res_index=0,saving_queue_size=200) as acq:
                acq.acquire(events)

            acq = None

            # turn off lasers
            core.set_config('Laser','Off')
            core.wait_for_config('Laser','Off')
            
    if run_scope == True:
        # turn all lasers off
        core.set_config('Laser','Off')
        core.wait_for_config('Laser','Off')

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    main()