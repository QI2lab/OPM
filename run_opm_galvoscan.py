#!/usr/bin/env python

'''
OPM galvo scan using Pycromanager.

D. Shepherd 04/21 - streamline code for fast acquisition and immediate upload to server
P. Brown 03/21 - multiline digital and analog NI DAQ control using camera as master
D. Shepherd 01/21 - initial pycromanager work, ported from stage control code
'''

# imports
from pycromanager import Bridge, Acquisition
from pathlib import Path
import numpy as np
import time
import pandas as pd
import PyDAQmx as daq
import ctypes as ct
import subprocess
from itertools import compress
import shutil
from threading import Thread
import data_io

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # set up lasers
    channel_labels = ["405", "488", "561", "635", "730"]
    channel_states = [False, False, True, False, False] # true -> active, false -> inactive
    channel_powers = [0, 10, 100, 40, 100] # (0 -> 100%)
    do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

    # parse which channels are active
    active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
    n_active_channels = len(active_channel_indices)
    
    print("%d active channels: " % n_active_channels, end="")
    for ind in active_channel_indices:
        print("%s " % channel_labels[ind], end="")
    print("")

    # exposure time
    exposure_ms = 1.5 #unit: ms

    # scan axis range
    scan_axis_range_um = 10.0 # unit: microns
    
    # galvo voltage at neutral
    galvo_neutral_volt = 0 # unit: volts

    # timepoints
    timepoints = 3000

    # setup file name
    save_directory=Path('D:/20210624l/')
    save_name = 'glycerol50'

    # automatically transfer files to NAS at end of dataset
    transfer_files = False
 
    # display data
    display_flag = False

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    bridge = Bridge()
    core = bridge.get_core()

    # give camera time to change modes if necessary
    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')

    # set camera to internal trigger
    core.set_config('Camera-TriggerSource','INTERNAL')
    core.wait_for_config('Camera-TriggerSource','INTERNAL')
    
    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[0]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[1]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER KIND[2]','EXPOSURE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[0]','POSITIVE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[1]','POSITIVE')
    core.set_property('OrcaFusionBT','OUTPUT TRIGGER POLARITY[2]','POSITIVE')

    # set exposure time
    core.set_exposure(exposure_ms)

    # determine image size
    core.snap_image()
    y_pixels = core.get_image_height()
    x_pixels = core.get_image_width()

    # turn all lasers on
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set all laser to external triggering
    core.set_config('Modulation-405','External-Digital')
    core.wait_for_config('Modulation-405','External-Digital')
    core.set_config('Modulation-488','External-Digital')
    core.wait_for_config('Modulation-488','External-Digital')
    core.set_config('Modulation-561','External-Digital')
    core.wait_for_config('Modulation-561','External-Digital')
    core.set_config('Modulation-637','External-Digital')
    core.wait_for_config('Modulation-637','External-Digital')
    core.set_config('Modulation-730','External-Digital')
    core.wait_for_config('Modulation-730','External-Digital')

    # turn all lasers on
    core.set_config('Laser','AllOn')
    core.wait_for_config('Laser','AllOn')

    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # galvo scan setup
    scan_axis_step_um = 0.4  # unit: um
    #scan_axis_calibration = 0.039 # unit: V / um
    scan_axis_calibration = 0.0453 # unit: V / um

    min_volt = -(scan_axis_range_um * scan_axis_calibration / 2.) + galvo_neutral_volt # unit: volts
    scan_axis_step_volts = scan_axis_step_um * scan_axis_calibration # unit: V
    scan_axis_range_volts = scan_axis_range_um * scan_axis_calibration # unit: V
    scan_steps = np.rint(scan_axis_range_volts / scan_axis_step_volts).astype(np.int16) # galvo steps
    
    # handle case where no scan steps
    if scan_steps == 0:
        scan_steps = 1
    
    # output experiment info
    print("Scan axis range: %.1f um = %0.3fV, Scan axis step: %.1f nm = %0.3fV , Number of galvo positions: %d" % 
          (scan_axis_range_um, scan_axis_range_volts, scan_axis_step_um * 1000, scan_axis_step_volts, scan_steps))
    print('Galvo neutral (Volt): ' + str(galvo_neutral_volt)+', Min voltage (volt): '+str(min_volt))
    print('Time points:  ' + str(timepoints))

    # create events to execute scan
    events = []
    ch_idx = 0
    
    # Changes to event structure motivated by Henry's notes that pycromanager struggles to read "non-standard" axes. 
    # https://github.com/micro-manager/pycro-manager/issues/220
    for t in range(timepoints):
        for x in range(scan_steps):
            ch_idx = 0
            for c in range(len(do_ind)):
                if channel_states[c]:
                    evt = { 'axes': {'t': t, 'z': x, 'c': ch_idx }}
                    ch_idx = ch_idx+1
                    events.append(evt)
    print("Generated %d events" % len(events))

    # setup DAQ
    nvoltage_steps = scan_steps
    # 2 time steps per frame, except for first frame plus one final frame to reset voltage
    #samples_per_ch = (nvoltage_steps * 2 - 1) + 1
    samples_per_ch = (nvoltage_steps * 2 * n_active_channels - 1) + 1
    DAQ_sample_rate_Hz = 10000
    #retriggerable = True
    num_DI_channels = 8

    # Generate values for DO
    dataDO = np.zeros((samples_per_ch, num_DI_channels), dtype=np.uint8)
    for ii, ind in enumerate(active_channel_indices):
        dataDO[2*ii::2*n_active_channels, ind] = 1
    dataDO[-1, :] = 0

    # generate voltage steps
    max_volt = min_volt + scan_axis_range_volts  # 2
    voltage_values = np.linspace(min_volt, max_volt, nvoltage_steps)

    # Generate values for AO
    waveform = np.zeros(samples_per_ch)
    # one less voltage value for first frame
    waveform[0:2*n_active_channels - 1] = voltage_values[0]

    if len(voltage_values) > 1:
        # (2 * # active channels) voltage values for all other frames
        waveform[2*n_active_channels - 1:-1] = np.kron(voltage_values[1:], np.ones(2 * n_active_channels))
    
    # set back to initial value at end
    waveform[-1] = voltage_values[0]

    #def read_di_hook(event):
    try:    
        # ----- DIGITAL input -------
        taskDI = daq.Task()
        taskDI.CreateDIChan("/Dev1/PFI0", "", daq.DAQmx_Val_ChanForAllLines)
        
        ## Configure change detectin timing (from wave generator)
        taskDI.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
        taskDI.CfgChangeDetectionTiming("/Dev1/PFI0", "/Dev1/PFI0", daq.DAQmx_Val_ContSamps, 0)

        ## Set where the starting trigger 
        taskDI.CfgDigEdgeStartTrig("/Dev1/PFI0", daq.DAQmx_Val_Rising)
        
        ## Export DI signal to unused PFI pins, for clock and start
        taskDI.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI2")
        taskDI.ExportSignal(daq.DAQmx_Val_StartTrigger, "/Dev1/PFI1")
        
        # ----- DIGITAL output ------   
        taskDO = daq.Task()
        # TO DO: Write each laser line separately!
        taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)

        ## Configure timing (from DI task) 
        taskDO.CfgSampClkTiming("/Dev1/PFI2", DAQ_sample_rate_Hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, samples_per_ch)
        
        ## Write the output waveform
        samples_per_ch_ct_digital = ct.c_int32()
        taskDO.WriteDigitalLines(samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByChannel, dataDO, ct.byref(samples_per_ch_ct_digital), None)

        # ------- ANALOG output -----------

        # first, set the galvo to the initial point if it is not already
        taskAO_first = daq.Task()
        taskAO_first.CreateAOVoltageChan("/Dev1/ao0", "", -4.0, 4.0, daq.DAQmx_Val_Volts, None)
        taskAO_first.WriteAnalogScalarF64(True, -1, waveform[0], None)
        taskAO_first.StopTask()
        taskAO_first.ClearTask()

        # now set up the task to ramp the galvo
        taskAO = daq.Task()
        taskAO.CreateAOVoltageChan("/Dev1/ao0", "", -4.0, 4.0, daq.DAQmx_Val_Volts, None)

        ## Configure timing (from DI task)
        taskAO.CfgSampClkTiming("/Dev1/PFI2", DAQ_sample_rate_Hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, samples_per_ch)
        
        ## Write the output waveform
        samples_per_ch_ct = ct.c_int32()
        taskAO.WriteAnalogF64(samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByScanNumber, waveform, ct.byref(samples_per_ch_ct), None)

        ## ------ Start both tasks ----------
        taskAO.StartTask()    
        taskDO.StartTask()    
        taskDI.StartTask()

    except daq.DAQError as err:
        print("DAQmx Error %s"%err)

    # run acquisition
    with Acquisition(directory=save_directory, name=save_name, show_display=display_flag, max_multi_res_index=0, saving_queue_size=5000) as acq:
        acq.acquire(events)

    acq = None

    # stop DAQ
    try:
        ## Stop and clear both tasks
        taskDI.StopTask()
        taskDO.StopTask()
        taskAO.StopTask()
        taskDI.ClearTask()
        taskAO.ClearTask()
        taskDO.ClearTask()
    except daq.DAQError as err:
        print("DAQmx Error %s"%err)

    # save galvo scan parameters
    scan_param_data = [{'root_name': str(save_name),
                        'scan_type': 'galvo',
                        'theta': 30.0, 
                        'scan_step': scan_axis_step_um*1000., 
                        'pixel_size': pixel_size_um*1000.,
                        'galvo_scan_range_um': scan_axis_range_um,
                        'galvo_volts_per_um': scan_axis_calibration, 
                        'num_t': int(timepoints),
                        'num_y': 1, # might need to change this eventually
                        'num_z': 1, # might need to change this eventually
                        'num_ch': int(n_active_channels),
                        'scan_axis_positions': int(scan_steps),
                        'y_pixels': y_pixels,
                        'x_pixels': x_pixels,
                        '405_active': channel_states[0],
                        '488_active': channel_states[1],
                        '561_active': channel_states[2],
                        '635_active': channel_states[3],
                        '730_active': channel_states[4]}]
    
    # df_galvo_scan_params = pd.DataFrame(scan_param_data)
    # save_name_galvo_params = save_directory / 'scan_metadata.csv'
    # df_galvo_scan_params.to_csv(save_name_galvo_params)
    data_io.write_metadata(scan_param_data[0], save_directory / 'scan_metadata.csv')

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

    # set all laser to zero power
    channel_powers=[0,0,0,0,0]
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # put the galvo back to neutral
    # first, set the galvo to the initial point if it is not already
    taskAO_last = daq.Task()
    taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)
    taskAO_last.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
    taskAO_last.StopTask()
    taskAO_last.ClearTask()

    if transfer_files:
        # make parent directory on NAS and start reconstruction script on the server
        # make home directory on NAS
        save_directory_path = Path(save_directory)
        remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
        cmd='mkdir ' + str(remote_directory)
        status_mkdir = subprocess.run(cmd, shell=True)

        # copy full experiment metadata to NAS
        src= Path(save_directory) / Path('scan_metadata.csv') 
        dst= Path(remote_directory) / Path('scan_metadata.csv') 
        Thread(target=shutil.copy, args=[str(src), str(dst)]).start()

        # copy data to NAS
        save_directory_path = Path(save_directory)
        remote_directory = Path('y:/') / Path(save_directory_path.parts[1])
        src= Path(save_directory) / Path(save_name+ '_1') 
        dst= Path(remote_directory) / Path(save_name+ '_1') 
        Thread(target=shutil.copytree, args=[str(src), str(dst)]).start()

# run
if __name__ == "__main__":
    main()
    
