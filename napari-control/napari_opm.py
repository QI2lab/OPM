'''
Initial work on napari interface using pycro-manager, magic-class, and magic-gui
D. Shepherd - 12/2021
'''

from pycromanager import Bridge, Acquisition, start_headless
from magicclass import magicclass, set_design
from magicgui import magicgui
import napari
from pathlib import Path
import numpy as np
import time
import pandas as pd
import PyDAQmx as daq
import ctypes as ct
from image_post_processing import deskew
from napari.qt.threading import thread_worker
import warnings
import gc


def img_process_fn(image, metadata):
    """
    Pull images across Python <-> Java bridge for processing
    
    :param image: ndarray
        image from acquisition
    :param metadata: dict 
        metadata dictionary

    :return None:
    """

    global image_stack
    global counter
    global images_to_grab
    global scan_finished

    image_stack[counter,:,:]=image
    counter = counter + 1

    if counter == images_to_grab:
        scan_finished = True

@thread_worker
def acquire_3d_data(instrument_config):
    """
    Run 3D acquisition using parameters in OpmControl class
    
    :param instrument_config: magic-class
        OpmControl class

    :yield deskewed_image: ndarray
        deskewed data
    """

    while True:
        
        global counter
        global image_stack
        global images_to_grab
        global scan_finished
        global channel_idx

        scan_finished = False

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # set up lasers
        channel_labels = ["405", "488", "561", "635", "730"]
        channel_states = instrument_config.channel_states
        #[True, True, True, False, False] # true -> active, false -> inactive
        #channel_powers = [50, 25, 10, 0, 0] # (0 -> 100%)
        channel_powers = instrument_config.channel_powers
        do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

        # parse which channels are active
        active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
        n_active_channels = len(active_channel_indices)
        
        print("%d active channels: " % n_active_channels, end="")
        for ind in active_channel_indices:
            print("%s " % channel_labels[ind], end="")
        print("")

        # exposure time
        exposure_ms = instrument_config.exposure_ms #unit: ms

        # scan axis range
        scan_axis_range_um = instrument_config.scan_axis_range_um # unit: microns
        
        # galvo voltage at neutral
        galvo_neutral_volt = instrument_config.galvo_neutral_volt # unit: volts

        # scan step size
        scan_axis_step_um = instrument_config.galvo_step  # unit: um

        # display data
        display_flag = False

        timepoints = 1

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------End setup of scan parameters----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        with Bridge() as bridge:
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
            scan_axis_calibration = 0.043 # unit: V / um

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
            
            core = None

            del core
        gc.collect()

        counter = 0
        image_stack = np.empty([n_active_channels*scan_steps,y_pixels,x_pixels]).astype(np.uint16)
        images_to_grab = n_active_channels*scan_steps
        scan_finished = False

        # run acquisition
        with Acquisition(directory=None, name=None, show_display=display_flag, 
                        max_multi_res_index=0, saving_queue_size=1000, image_process_fn=img_process_fn) as acq:
            acq.acquire(events)

        while not(scan_finished):
            time.sleep(0.25)

        acq = None
        del acq
        gc.collect()
            
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

        deskew_parameters = np.empty([3])
        deskew_parameters[0] = 30                           # (degrees)
        deskew_parameters[1] = 400        # (nm)
        deskew_parameters[2] = 115                          # (nm)
        for i in range(n_active_channels):
            deskewed_image = deskew(np.flipud(image_stack[i::n_active_channels,:]),*deskew_parameters)
            channel_idx = active_channel_indices[i]
            yield deskewed_image.astype(np.uint16)

        del deskewed_image
        gc.collect()

# OPM control UI element            
@magicclass(labels=False)
@set_design(text="ASU Snouty-OPM control")
class OpmControl:

    # initialize
    def __init__(self):
        self.active_channel = "Off"
        self.channel_powers = np.zeros(5,dtype=np.int8)
        self.exposure_ms = 10.0
        self.galvo_step = 0.4
        self.scan_axis_calibration = 0.043 # unit: V / um
        self.galvo_neutral_volt = -.15 # unit: volts
        self.scan_axis_range_um = 50.0
        self.channel_states=[False,False,False,False,False]

    # change exposure time
    @magicgui(
        auto_call=True,
        exposure_ms={"widget_type": "FloatSpinBox", "min": 1, "max": 500,'label': 'Camera exposure (ms)'},
        layout='horizontal'
    )
    def set_exposure(self, exposure_ms=10.0):
        """
        Set expsosure time in milliseconds
        
        :param exposure_ms: float
            exposure time for each exposure

        :return None:
        """
        self.exposure_ms=exposure_ms

    # change laser power
    @magicgui(
        auto_call=True,
        power_405={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '405nm power (%)'},
        power_488={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '488nm power (%)'},
        power_561={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '561nm power (%)'},
        power_635={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '635nm power (%)'},
        power_730={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '730nm power (%)'},
        layout='vertical'
    )
    def set_laser_power(self, power_405=0.0, power_488=0.0, power_561=0.0, power_635=0.0, power_730=0.0,):
        """
        Set laser power in percentage from 0->100%
        
        :param power_405: float
            405nm laser power from 0-100
        :param power_488: float
            488nm laser power from 0-100
        :param power_561: float
            561nm laser power from 0-100
        :param power_635: float
            635nm laser power from 0-100
        :param power_730: float
            730nm laser power from 0-100

        :return None:
        """
        self.channel_powers=[power_405,power_488,power_561,power_635,power_730]

    # change active laser
    @magicgui(
        auto_call=True,
        active_channels = {"widget_type": "Select", "choices": ["Off","405","488","561","635","730"], "allow_multiple": True, "label": "Active channels"}
    )
    def set_active_channel(self, active_channels):
        """
        Set active channels
        
        :param active_channels: list
            List of Booleans containing active channels

        :return None:
        """
        states = [False,False,False,False,False]
        for channel in active_channels:
            if channel == 'Off':
                states = [False,False,False,False,False]
                break
            if channel == '405':
                states[0]='True'
            elif channel == '488':
                states[1]='True'
            elif channel == '561':
                states[2]='True'
            elif channel == '635':
                states[3]='True'
            elif channel == '730':
                states[4]='True'
        self.channel_states=states
    
    # set lateral galvo footprint 
    @magicgui(
        auto_call=True,
        galvo_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 275, "label": 'Galvo sweep (um)'},
        layout='horizontal'
    )
    def set_galvo_sweep(self, galvo_footprint_um=50.0):
        """
        Set footprint of galvanometer mirror sweep
        
        :param galvo_footprint_um: float
            Size of galvo footprint microns. Minimum 5, maximum 275.

        :return None:
        """
        self.scan_axis_range_um=galvo_footprint_um

    # set lateral galvo step size
    @magicgui(
        auto_call=True,
        galvo_step_um={"widget_type": "FloatSpinBox", "min": 0, "max": 1, "label": 'Galvo step (um)'},
        layout='horizontal'
    )
    def set_galvo_step(self, galvo_step_um=0.4):
        """
        Set step size between galvanometer positions
        
        :param galvo_step_um: float
            Size of galvo step in microns. Default is 0.4.

        :return None:
        """
        self.galvo_step=galvo_step_um


def main():
    """
    Control OPM through Napari
    
    :param None: 
    :return None:
    """

    # TO DO: clean up global variables 
    global counter
    global image_stack
    global first_iteration
    global images_to_grab
    global scan_finished
    global worker_started
    global channel_idx

    # initialize important global variables
    first_iteration = True
    worker_started = False

    # create viewer, OpmControl class, and thread worker for running instrument
    viewer = napari.Viewer()
    instrument_control_widget = OpmControl()
    viewer.window.add_dock_widget(instrument_control_widget,name='OPM control')
    worker = acquire_3d_data(instrument_control_widget)

    # start continuous 3D volume (hardware triggering during sweep)
    @magicgui(
        auto_call=True,
        start_3D={"widget_type": "PushButton", "label": 'Start live (3D)'},
        layout='horizontal'
    )
    def start_live_mode_3D(start_3D):
        """
        Start 3D acquisition
        
        :param start_3D: boolean
            Boolean from button

        :return None:
        """
        global worker_started

        if not(worker_started):
            worker.start()
            worker_started = True
            print('Worked started.')
        else:
            worker.resume()

    # stop continuous 3D volume (hardware triggering during sweep)
    @magicgui(
        auto_call=True,
        stop_3D={"widget_type": "PushButton", "label": 'Stop live (3D)'},
        layout='horizontal'
    )
    def stop_live_mode_3D(stop_3D):
        """
        Stop 3D acquisition
        
        :param stop_3D: boolean
            Boolean from button
            
        :return None:
        """

        global scan_finished

        worker.pause()
        while not(scan_finished):
            time.sleep(0.25)

        print('Worker paused.')


        # cleanup hardware control so MM can control OPM again
        with Bridge() as bridge:
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

            core = None
            del core

        gc.collect()

        # put the galvo back to neutral
        taskAO_last = daq.Task()
        taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
        taskAO_last.WriteAnalogScalarF64(True, -1, instrument_control_widget.galvo_neutral_volt, None)
        taskAO_last.StopTask()
        taskAO_last.ClearTask()

    def update_layer(new_image):
        """
        Update layers in Napari viewer
        
        :param new_image: ndarray
            image to update
            
        :return None:
        """

        #clunky way of tracking current channel index 
        global channel_idx
        
        try:
            if channel_idx==0:
                viewer.layers['ch0'].data = new_image
            elif channel_idx==1:
                viewer.layers['ch1'].data = new_image
            elif channel_idx==2:
                viewer.layers['ch2'].data = new_image
            elif channel_idx==3:
                viewer.layers['ch3'].data = new_image
            elif channel_idx==4:
                viewer.layers['ch4'].data = new_image
        except KeyError:
            if channel_idx==0:
                viewer.add_image(new_image, name='ch0', colormap='bop purple',contrast_limits=[100,.9*np.max(new_image)])
            elif channel_idx==1:
                viewer.add_image(new_image, name='ch1', blending='additive', colormap='bop blue',contrast_limits=[100,.9*np.max(new_image)])
            elif channel_idx==2:
                viewer.add_image(new_image, name='ch2', blending='additive', colormap='bop orange',contrast_limits=[100,.9*np.max(new_image)])
            elif channel_idx==3:
                viewer.add_image(new_image, name='ch3', blending='additive', colormap='red',contrast_limits=[100,.9*np.max(new_image)])
            elif channel_idx==4:
                viewer.add_image(new_image, name='ch4', blending='additive', colormap='gray',contrast_limits=[100,.9*np.max(new_image)])


    # setup thread worker and run Napari
    worker.yielded.connect(update_layer)
    viewer.window.add_dock_widget([start_live_mode_3D,stop_live_mode_3D],name='Live mode')
    napari.run()

    # clean up hardware once Napari closes
    with Bridge() as bridge:
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

        core = None
        del core

    gc.collect()

    # put the galvo back to neutral
    # first, set the galvo to the initial point if it is not already
    taskAO_last = daq.Task()
    taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
    taskAO_last.WriteAnalogScalarF64(True, -1, instrument_control_widget.galvo_neutral_volt, None)
    taskAO_last.StopTask()
    taskAO_last.ClearTask()

    # stop worker
    worker.quit()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
