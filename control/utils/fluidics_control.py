#!/usr/bin/env python

'''
Functions to execute fluidics programs

Shepherd 11/22 - Updates from Alexis Colloumb changes on his widefield setup. 
                 This has a breaking change! Rounds must now start at "1", not "0".
Shepherd 05/21 - initial commit
'''

import numpy as np
import pandas as pd
import time
import sys
from .data_io import time_stamp
import sys

def lookup_valve(source_name):
    """
    Convert name of well using ASU controller convention to ASU MVP valve settings 
    :param source_name: string
    :return valve_position: ndarray
    """

    valve_dict = {'B01': [0,1], 'B02': [0,2], 'B03': [0,3], 'B04': [0,4], 'B05': [0,5], 'B06': [0,6], 'B07': [0,7], 
                  'B08': [1,1], 'B09': [1,2], 'B10': [1,3], 'B11': [1,4], 'B12': [1,5], 'B13': [1,6], 'B14': [1,7], 
                  'B15': [2,1], 'B16': [2,2], 'B17': [2,3], 'B18': [2,4], 'B19': [2,5], 'B20': [2,6], 'B21': [2,7], 
                  'B22': [3,1], 'B23': [3,2], 'B24': [3,3],
                  'SSC': [3,0], 'READOUT WASH': [3,4], 'IMAGING BUFFER': [3,5], 'CLEAVE': [3,7]}

    valve_position = valve_dict.get(source_name)

    return valve_position

def run_fluidic_program(r_idx, df_program, mvp_controller, pump_controller):

    """
    Run fluidics program for a given round. Requires data structure generated by ASU fluidics program generator (define_fluidics_program.ipynb)
    :param r_idx: int
        fluidics round to execute, expected as human notation,
        not numpy's (first round is 1, not 0)
    :param df_program: dataframe
        dataframe containing entire fluidics program
    :param mvp_controller: HamiltonMVP
        handle to initialized chain of Hamilton MVP valves
    :param pump_controller: APump
        handle to initialized pump

    :return True: boolean
        TO DO: need to work this into try/except statements to catch any pump errors
    """

    # select current round
    df_current_program = df_program[(df_program['round']==r_idx+1)]
    print(time_stamp(), ': Executing iterative round '+str(r_idx+1)+'.')
    for index, row in df_current_program.iterrows():
        # extract source name
        source_name = str(row['source']).strip()

        # extract pump rate
        # pump_amount_ml = float(row['volume'])  # not used
        pump_time_min  = float(row['time'])
        try:
            pump_rate = float(row['pump'])
        except:
            pump_rate = -1.0

        if source_name == 'RUN':
            pump_controller.stopFlow()
            print(time_stamp(), ': Fluidics round done, running imaging.')
        elif source_name == 'PAUSE':
            pump_controller.stopFlow()
            print(time_stamp(), ': Pausing for:' +str(pump_time_min*60)+' seconds.')
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

            print(time_stamp(), ': MVP unit: '+str(mvp_unit)+'; Valve #: '+str(valve_number))
            print(time_stamp(), f': Pump setting: {pump_rate} for {source_name}')

            if pump_rate == -1.0:
                print(time_stamp(), ': Error in determining pump rate. Exiting.')
                sys.exit()

            # run pump
            pump_controller.startFlow(pump_rate,direction='Forward')
            time.sleep(pump_time_min*60)
            pump_controller.stopFlow()
    
    return True