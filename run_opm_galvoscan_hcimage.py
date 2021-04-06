#!/usr/bin/env python

'''
OPM galvo scan using Pyromanager.

Shepherd 01/21
'''

# imports
#from pycromanager import Bridge, Acquisition
from pathlib import Path
import numpy as np
import time
import pandas as pd
import PyDAQmx as daq
import ctypes as ct

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # set up lasers
    channel_labels = ["405", "488", "561", "635", "730"]
    channel_states = [False, False, True, False, False] # true -> active, false -> inactive
    channel_powers = [0, 0, 50, 0, 0] # (0 -> 100%)
    do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

    # parse which channels are active
    active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
    n_active_channels = len(active_channel_indices)
    
    print("%d active channels: " % n_active_channels, end="")
    for ind in active_channel_indices:
        print("%s " % channel_labels[ind], end="")
    print("")

    # exposure time
    #exposure_ms = 2. #unit: ms

    # scan axis range
    scan_axis_range_um = 10.0 # unit: microns
    
    # galvo voltage at neutral
    galvo_neutral_volt = -0.075 # unit: volts

    # setup file name
    save_directory=Path('E:/20210401f_franky/')
    save_name = 'DNAnanotube_sucrose_50'

    # set total number of frames (from HCimage)
    total_frames = 40000 # unit: number of frames
    
    # display data
    display_flag = False

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------
    # camera pixel size
    pixel_size_um = .115 # unit: um

    # galvo scan setup
    scan_axis_step_um = 0.4  # unit: um
    scan_axis_calibration = 0.034 # unit: V / um
    min_volt = -(scan_axis_range_um*scan_axis_calibration/2.) + galvo_neutral_volt # unit: volts
    timepoints = int(total_frames/(scan_axis_range_um/scan_axis_step_um)) #unit: number of timepoints
    scan_axis_step_volts = scan_axis_step_um * scan_axis_calibration # unit: V
    scan_axis_range_volts = scan_axis_range_um * scan_axis_calibration # unit: V
    scan_steps = np.rint(scan_axis_range_volts / scan_axis_step_volts).astype(np.int16) # galvo steps
    
    # output experiment info
    print('Scan axis range (um): '+str(scan_axis_range_um)+
    ', Scan axis step (nm): '+str(scan_axis_step_um*1000)+
    ', Number of galvo positions: '+str(scan_steps))
    print('Total frames: '+str(total_frames)+', Time points:  '+str(timepoints))
    print('Galvo neutral (Volt): '+str(galvo_neutral_volt)+', Min voltage (volt): '+str(min_volt))

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

    #print("Digital output array:")
    print(dataDO)

    # generate voltage steps
    max_volt = min_volt + scan_axis_range_volts  # 2
    voltage_values = np.linspace(min_volt, max_volt, nvoltage_steps)

    # Generate values for AO
    waveform = np.zeros(samples_per_ch)
    # one less voltage value for first frame
    waveform[0:2*n_active_channels - 1] = voltage_values[0]
    # (2 * # active channels) voltage values for all other frames
    waveform[2*n_active_channels - 1:-1] = np.kron(voltage_values[1:], np.ones(2 * n_active_channels))
    # set back to right value at end
    waveform[-1] = voltage_values[0]
    #print("Analog output voltage array:")
    #print(waveform)

    #def read_di_hook(event):
    try:    
        # ----- DIGITAL input -------
        taskDI = daq.Task()
        taskDI.CreateDIChan("/Dev1/PFI0","",daq.DAQmx_Val_ChanForAllLines)
        #taskDI.CreateDIChan("OnboardClock","",daq.DAQmx_Val_ChanForAllLines)
        
        ## Configure change detectin timing (from wave generator)
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
        print("WriteDigitalLines sample per channel count = %d" % samples_per_ch_ct_digital.value)

        # ------- ANALOG output -----------

        # first, set the galvo to the initial point if it is not already
        taskAO_first = daq.Task()
        taskAO_first.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)
        taskAO_first.WriteAnalogScalarF64(True, -1, waveform[0], None)
        taskAO_first.StopTask()
        taskAO_first.ClearTask()

        # now set up the task to ramp the galvo
        taskAO = daq.Task()
        taskAO.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)

        ## Configure timing (from DI task)
        taskAO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_ContSamps,samples_per_ch)
        
        ## Write the output waveform
        samples_per_ch_ct = ct.c_int32()
        taskAO.WriteAnalogF64(samples_per_ch,False,10.0,daq.DAQmx_Val_GroupByScanNumber,waveform,ct.byref(samples_per_ch_ct),None)
        print("WriteAnalogF64 sample per channel count = %d" % samples_per_ch_ct.value)

        ## ------ Start both tasks ----------
        taskAO.StartTask()    
        taskDO.StartTask()    
        taskDI.StartTask()

    except daq.DAQError as err:
        print("DAQmx Error %s"%err)


    # save galvo scan parameters
    scan_param_data = [{'theta': 30.0, 
                        'scan step': scan_axis_step_um*1000., 
                        'pixel size': pixel_size_um*1000.,
                        'galvo scan range um': scan_axis_range_um,
                        'galvo volts per um': scan_axis_calibration, 
                        'galvo start volt': min_volt,
                        'time points': timepoints,
                        'channels': 1,
                        'steps per volume': scan_steps,
                        'y_pixels': 256,
                        'x_pixels': 2304,
                        'channels': channel_labels,
                        'channel states': channel_states}]
    
    df_galvo_scan_params = pd.DataFrame(scan_param_data)
    save_name_galvo_params = save_directory / 'galvo_scan_params.pkl'
    df_galvo_scan_params.to_pickle(save_name_galvo_params)

    input("Press enter to close script")

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

# run
if __name__ == "__main__":
    main()