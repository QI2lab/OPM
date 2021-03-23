#!/usr/bin/env python

'''
OPM galvo scan using Pyromanager.

Shepherd 01/21
'''

# imports
from pycromanager import Bridge, Acquisition
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
    state_488 = 1
    state_561 = 0
    state_635 = 0
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 0
    power_488 = 5
    power_561 = 0
    power_635 = 0
    power_730 = 0

    # exposure time
    exposure_ms = 40.0 #unit: ms

    # scan axis range
    scan_axis_range_um = 100.0 # unit: microns
    
    # voltage start
    min_volt = -2.5 # unit: volts

    # setup file name
    save_directory=Path('E:/20210321_plant/area_003')
    save_name = 'test_run'

    # set timepoints
    timepoints = 120 #unit: number of timepoints

    # display data
    display_flag = False

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    bridge = Bridge()
    core = bridge.get_core()

    # turn off lasers
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # give camera time to change modes if necessary
    #core.set_config('Camera-Setup','ScanMode3')
    #core.wait_for_config('Camera-Setup','ScanMode3')

    # set camera to internal trigger
    core.set_config('Camera-TriggerSource','INTERNAL')
    core.wait_for_config('Camera-TriggerSource','INTERNAL')
    
    # set camera to internal trigger
    # give camera time to change modes if necessary
    core.set_property('Camera','OUTPUT TRIGGER KIND[0]','EXPOSURE')
    core.set_property('Camera','OUTPUT TRIGGER KIND[1]','EXPOSURE')
    core.set_property('Camera','OUTPUT TRIGGER KIND[2]','EXPOSURE')
    core.set_property('Camera','OUTPUT TRIGGER POLARITY[0]','POSITIVE')
    core.set_property('Camera','OUTPUT TRIGGER POLARITY[1]','POSITIVE')
    core.set_property('Camera','OUTPUT TRIGGER POLARITY[2]','POSITIVE')
    
    # set exposure
    core.set_exposure(exposure_ms)

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # galvo scan setup
    scan_axis_step_um = 0.4  # unit: um
    scan_axis_calibration = 0.034 # unit: V / um
    scan_axis_step_volts = scan_axis_step_um * scan_axis_calibration # unit: V
    scan_axis_range_volts = scan_axis_range_um * scan_axis_calibration # unit: V
    scan_steps = np.rint(scan_axis_range_volts / scan_axis_step_volts).astype(np.int16) # galvo steps
    #print(scan_steps)
    
    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # turn all lasers on
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

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

    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # output experiment info
    print('Number of galvo positions: '+str(scan_steps))
    print('Number of channels: '+str(np.sum(channel_states)))
    print('Number of timepoints: '+str(timepoints))

    # create events to execute scan
    events = []
    
    # Changes to event structure motivated by Henry's notes that pycromanager struggles to read "non-standard" axes. 
    # https://github.com/micro-manager/pycro-manager/issues/220
    for t in range(timepoints):
        for c in range(len(channel_states)):
            for x in range(scan_steps):
                if channel_states[c]==1:
                    evt = { 'axes': {'t': t, 'x': x, 'channel': c }}
                    events.append(evt)
    print("Generated %d events" % len(events))

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

    def read_di_buffer_hook(event):
        samps = (ct.c_int8 * 10)()
        samps_per_ch_read = ct.c_uint32()
        num_bytes_per_sample = ct.c_uint32()
        #taskDI.ReadDigitalLines(1, 1e-3, False, len(samps), None, ct.byref(samps), ct.byref(samps_per_ch_read), ct.byref(num_bytes_per_sample))
        return event

    # run acquisition at this Z plane
    with Acquisition(directory=save_directory, name=save_name, show_display=display_flag, max_multi_res_index=0, saving_queue_size=5000) as acq:
        acq.acquire(events)

    #time.sleep(nvoltage_steps * timepoints * 1)

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

    # try to clean up acquisition so that AcqEngJ releases directory. This way we can move it to the network storage
    # in the background.
    # NOTE: This is a bug, the directory is not released until Micromanager is shutdown
    # https://github.com/micro-manager/pycro-manager/issues/218
    acq = None

    # turn all lasers on
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set all lasers back to software control
    core.set_config('Trigger-405','CW (constant power)')
    core.wait_for_config('Trigger-405','CW (constant power)')
    core.set_config('Trigger-488','CW (constant power)')
    core.wait_for_config('Trigger-488','CW (constant power)')
    core.set_config('Trigger-561','CW (constant power)')
    core.wait_for_config('Trigger-561','CW (constant power)')
    core.set_config('Trigger-637','CW (constant power)')
    core.wait_for_config('Trigger-637','CW (constant power)')
    core.set_config('Trigger-730','CW (constant power)')
    core.wait_for_config('Trigger-730','CW (constant power)')

    # save galvo scan parameters
    # save galvo scan parameters
    scan_param_data = [{'theta': 30.0, 
                        'scan step': scan_axis_step_um*1000., 
                        'pixel size': pixel_size_um*1000.,
                        'galvo scan range um': scan_axis_range_um,
                        'galvo volts per um': scan_axis_calibration, 
                        'galvo start volt': min_volt,
                        'time points': timepoints,
                        'channels': np.sum(channel_states),
                        'steps per volume': scan_steps,
                        'y_pixels': 256,
                        'x_pixels': 2304}]
    
    df_galvo_scan_params = pd.DataFrame(scan_param_data)
    save_name_galvo_params = save_directory / 'galvo_scan_params.pkl'
    df_galvo_scan_params.to_pickle(save_name_galvo_params)

# run
if __name__ == "__main__":
    main()