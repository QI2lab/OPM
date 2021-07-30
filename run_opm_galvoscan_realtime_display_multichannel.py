#!/usr/bin/env python

'''
Real-time display of deskewed data from a OPM galvo scan using Pycromanager.

D. Shepherd 06/21 - initial work on real-time display using Napari
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
import napari
from magicgui import magicgui
from image_post_processing import deskew
from napari.qt.threading import thread_worker
import warnings
import gc

def img_process_fn(image, metadata):

    global image_stack
    global ch_counter
    global z_counter
    global images_to_grab
    global scan_finished
    global n_active_channels
    
    image_stack[ch_counter,z_counter,:,:]=image
    
    ch_counter = ch_counter + 1
    
    if ch_counter == n_active_channels:
        z_counter = z_counter + 1
        ch_counter = 0

    if z_counter == images_to_grab:
        scan_finished = True

@thread_worker
def acquire_data():

    while True:
        
        global z_counter
        global ch_counter
        global image_stack
        global images_to_grab
        global scan_finished
        global n_active_channels

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # set up lasers
        channel_labels = ["405", "488", "561", "635", "730"]
        channel_states = [False, False, False, True, True] # true -> active, false -> inactive
        channel_powers = [5, 20, 20, 10, 100] # (0 -> 100%)
        do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

        # parse which channels are active
        active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
        n_active_channels = len(active_channel_indices)
        
        print("%d active channels: " % n_active_channels, end="")
        for ind in active_channel_indices:
            print("%s " % channel_labels[ind], end="")
        print("")

        # exposure time
        exposure_ms = 200.0 #unit: ms

        # scan axis range
        scan_axis_range_um = 50.0 # unit: microns
        
        # galvo voltage at neutral
        galvo_neutral_volt = 0 # unit: volts

        # timepoints
        timepoints = 1

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
            taskAO_first.CreateAOVoltageChan("/Dev1/ao0", "", -6.0, 6.0, daq.DAQmx_Val_Volts, None)
            taskAO_first.WriteAnalogScalarF64(True, -1, waveform[0], None)
            taskAO_first.StopTask()
            taskAO_first.ClearTask()

            # now set up the task to ramp the galvo
            taskAO = daq.Task()
            taskAO.CreateAOVoltageChan("/Dev1/ao0", "", -6.0, 6.0, daq.DAQmx_Val_Volts, None)

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

        ch_counter = 0
        z_counter = 0
        image_stack = np.empty([n_active_channels,scan_steps,y_pixels,x_pixels]).astype(np.uint16)

        deskew_parameters = np.empty([3])
        deskew_parameters[0] = 30         # (degrees)
        deskew_parameters[1] = 400        # (nm)
        deskew_parameters[2] = 115        # (nm)

        # calculate size of one volume
        # change step size from physical space (nm) to camera space (pixels)
        pixel_step = deskew_parameters[1]/deskew_parameters[2]    # (pixels)

        # calculate the number of pixels scanned during stage scan 
        scan_end = image_stack.shape[1] * pixel_step  # (pixels)
        y_pixels = image_stack.shape[2]
        x_pixels = image_stack.shape[3]

        # calculate properties for final image
        ny = np.int64(np.ceil(scan_end+y_pixels*np.cos(deskew_parameters[0]*np.pi/180))) # (pixels)
        nz = np.int64(np.ceil(y_pixels*np.sin(deskew_parameters[0]*np.pi/180)))          # (pixels)
        nx = np.int64(x_pixels)     

        deskewed_image = np.empty([n_active_channels,nz,ny,nx]).astype(np.uint16)

        images_to_grab = scan_steps
        scan_finished = False

        # run acquisition
        with Acquisition(directory=None, name=None, show_display=display_flag, 
                        max_multi_res_index=0, saving_queue_size=1000, image_process_fn=img_process_fn) as acq:
            acq.acquire(events)

        while not(scan_finished):
            time.sleep(0.5)

        acq = None

        acq_deleted = False
        while not(acq_deleted):
            try:
                del acq
            except:
                time.sleep(0.1)
                acq_deleted = False
            else:
                gc.collect()
                acq_deleted = True

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

        for i in range(n_active_channels):
            deskewed_image[i,:]= deskew(np.flipud(image_stack[i,:]),*deskew_parameters)

        yield deskewed_image.astype(np.uint16)

def main():

    global counter
    global image_stack
    global images_to_grab
    global scan_finished
    global acq_running
    global n_active_channels

    acq_running = False

    # first, set the galvo to the initial point if it is not already
    galvo_neutral_volt = 0 # unit: volts
    taskAO_last = daq.Task()
    taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
    taskAO_last.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
    taskAO_last.StopTask()
    taskAO_last.ClearTask()

    viewer = napari.Viewer(ndisplay=3)

    def update_layer(new_image):
        lookup_tables = ['bop purple','bop blue','bop orange','gray']
        channel_names = ['ch0','ch1','ch2','ch3']
        for i in range(n_active_channels):
            try:
                viewer.layers[channel_names[i]].data = new_image[i,:]
            except KeyError:
                viewer.add_image(new_image[i,:], name=channel_names[i], 
                                colormap=lookup_tables[i], 
                                contrast_limits=[100,7000], 
                                scale=[1,1,1])
    worker = acquire_data()
    worker.yielded.connect(update_layer)

    @magicgui(call_button = "Start")
    def start_acq():
        global acq_running

        if not(acq_running):
            acq_running = True
            worker.start()

    @magicgui(call_button = "Stop")
    def stop_acq():
        # set global acq_running to False to stop other workers
        global acq_running

        acq_running = False
        worker.pause()

    viewer.window.add_dock_widget(start_acq)
    viewer.window.add_dock_widget(stop_acq)
    napari.run()

    bridge = Bridge()
    core = bridge.get_core()

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
    '''
    channel_powers=[0,0,0,0,0]
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])
    '''

    # put the galvo back to neutral
    # first, set the galvo to the initial point if it is not already
    galvo_neutral_volt = 0 # unit: volts
    taskAO_last = daq.Task()
    taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
    taskAO_last.WriteAnalogScalarF64(True, -1, galvo_neutral_volt, None)
    taskAO_last.StopTask()
    taskAO_last.ClearTask()

# run
if __name__ == "__main__":
    main()