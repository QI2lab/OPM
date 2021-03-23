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

    # lasers to use
    # 0 -> inactive
    # 1 -> active
    state_405 = 0
    state_488 = 0
    state_561 = 1
    state_635 = 0
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 0
    power_488 = 0
    power_561 = 100
    power_635 = 0
    power_730 = 0

    # exposure time
    #exposure_ms = 2. #unit: ms

    # scan axis range
    scan_axis_range_um = 200.0 # unit: microns
    
    # voltage start
    min_volt = -3.4 # unit: volts

    # setup file name
    save_directory=Path('E:/20210318o/')
    save_name = 'beads'

    # set timepoints
    timepoints = 2 #unit: number of timepoints

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
    scan_axis_step_volts = scan_axis_step_um * scan_axis_calibration # unit: V
    scan_axis_range_volts = scan_axis_range_um * scan_axis_calibration # unit: V
    scan_steps = np.rint(scan_axis_range_volts / scan_axis_step_volts).astype(np.int16) # galvo steps
    
    # output experiment info
    print('Number of galvo positions: '+str(scan_steps))

    # setup DAQ
    nvoltage_steps = scan_steps
    # 2 time steps per frame, except for first frame plus one final frame to reset voltage
    samples_per_ch = (nvoltage_steps * 2 - 1) + 1
    DAQ_sample_rate_Hz = 10000
    #retriggerable = True
    num_DI_channels = 8

    # Generate values for DO
    dataDO = np.zeros((samples_per_ch,num_DI_channels),dtype=np.uint8)
    dataDO[::2,:] = 1
    dataDO[-1, :] = 0
    print("Digital output array:")
    print(dataDO[:, 0])

    # generate voltage steps
    max_volt = min_volt + scan_axis_range_volts  # 2
    voltage_values = np.linspace(min_volt,max_volt,nvoltage_steps)

    # Generate values for AO
    wf = np.zeros(samples_per_ch)
    # one voltage value for first frame
    wf[0] = voltage_values[0]
    # two voltage values for all other frames
    wf[1:-1:2] = voltage_values[1:] # start : end : step
    wf[2:-1:2] = voltage_values[1:]
    # set back to right value at end
    wf[-1] = voltage_values[0]
    #wf[:] = 0 # for test: galvo is not moved
    print("Analog output voltage array:")
    print(wf)

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
        taskAO = daq.Task()
        taskAO.CreateAOVoltageChan("/Dev1/ao0","",-4.0,4.0,daq.DAQmx_Val_Volts,None)
        
        ## Configure timing (from DI task)
        taskAO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_ContSamps,samples_per_ch)
        
        ## Write the output waveform
        samples_per_ch_ct = ct.c_int32()
        taskAO.WriteAnalogF64(samples_per_ch,False,10.0,daq.DAQmx_Val_GroupByScanNumber,wf,ct.byref(samples_per_ch_ct),None)
        print("WriteAnalogF64 sample per channel count = %d" % samples_per_ch_ct.value)

        ## ------ Start both tasks ----------
        taskAO.StartTask()    
        taskDO.StartTask()    
        taskDI.StartTask()

    except daq.DAQError as err:
        print("DAQmx Error %s"%err)


    # save galvo scan parameters
    scan_param_data = [{'theta': 30.0, 'scan step': scan_axis_step_um*1000., 'pixel size': pixel_size_um*1000.,
                        'galvo scan range um': scan_axis_range_um,
                        'galvo volts per um': scan_axis_calibration, 'steps per volume': scan_steps,
                        'galvo start volt': min_volt}]
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