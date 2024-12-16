#!/usr/bin/env python

'''
Functions to execute fluidics programs

Sheppard 08/04 - Updated to run with ElveFlow controls
Shepherd 03/23 - Add "REFRESH" setting to pause until user approves. Allows for solution change out.
Shepherd 11/22 - Updates from Alexis Colloumb changes on his widefield setup.
                 This has a breaking change! Rounds must now start at "1", not "0".
Shepherd 05/21 - initial commit
'''

import numpy as np
import pandas as pd
import time
import sys
import easygui
from .data_io import time_stamp
from .autofocus_remote_unit import manage_O3_focus


def run_fluidic_program(df_program,
                        flow_controller):

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
    df_current_program = df_program[(df_program['round']==r_idx)]
    print(time_stamp(), ': Executing iterative round '+str(r_idx)+'.')
    fluidics_success = False
    for index, row in df_current_program.iterrows():
        # Extract program functions
        type_name = str(row["type"]).strip()
        source_name = str(row['source']).strip()
        flow_rate = str(row['rate']).strip()
        volume = str(row['volume']).strip()
        pause = str(row['pause']).strip()
        prime_buffer = str(row['pause']).strip()

        if type_name=="FLUSH":
            # Run system flush with given source/volume/rate
            print(f"Fluidics starting flush:\n",
                  f"   source:{source_name}, volume:{volume:.2f}uL, rate:{flow_rate:.2f}uL/min")
            try:
                fluidics_success = flow_controller.run_flush(source=source_name,
                                                             rate=flow_rate,
                                                             volume=volume,
                                                             wait=pause,
                                                             verbose=False)
            except Exception as e:
                print(f"Exception occurred in Fluidics flush: {e}")

            if fluidics_success:
                print("Fluidics flush success!")

        elif type_name=="PID":
            # Run a flow contronlled loop to pump specfified volume
            print(f"Fluidics starting controlled flow:\n",
                  f"   source:{source_name}, volume:{volume:.2f}uL, rate:{flow_rate:.2f}uL/min")
            try:
                fluidics_success = flow_controller.run_pid(source=source_name,
                                                           rate=flow_rate,
                                                           volume=volume,
                                                           wait=pause,
                                                           verbose=False)
            except Exception as e:
                print(f"Exception occurred in Fluidics flow: {e}")

        elif type_name=="PRIME":
            # Run a the prime program to push a volume of fluid to the sample chamber
            print(f"Fluidics starting priming to sample chamber:\n",
                  f"   source:{source_name}, volume:{volume:.2f}uL, rate:{flow_rate:.2f}uL/min")
            if prime_buffer=="None":
                prime_buffer=="SSC"

            try:
                fluidics_success = flow_controller.run_prime_source(source=source_name,
                                                                    prime_buffer=prime_buffer,
                                                                    rate=flow_rate,
                                                                    volume=volume,
                                                                    wait=pause,
                                                                    verbose=False)
            except Exception as e:
                print(f"Exception occurred in Fluidics flow: {e}")

        elif type_name=="REFRESH":
            # Pause program until the user confirms fluid refresh is complete
            refresh_approved = False
            while not(refresh_approved):
                refresh_approved = easygui.ynbox('Refresh complete?', 'Title', ('Yes', 'No'))

            # Run system prime and flush to prepare lines and remove air bubbles
            flow_controller.run_system_prime()

        else:
            print(f"No valid fluidics program selected (round={r_idx}, index={index}); moving to next line")

    return True