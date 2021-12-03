'''
Initial work on napari interface to OPM using pymmcore-plus, magic-gui, and magic-class

Relevant hardware:
NI DAQ
Hamamatsu Fusion-BT
Coherent OBIS LaserBoxxx

This will work with any setup that can setup the camera as master, lasers can driven ON/OFF during camera readout time using digital input, 
galvo can be moved during camera readout time, and DAQ can update digital/analog lines based on trigger from camera during camera readout time.

For different hardware, the specific calls will have to modified to get the hardware triggering working.

D. Shepherd - 12/2021
'''

from pymmcore_plus import RemoteMMCore
from magicclass import magicclass, set_design
from magicgui import magicgui
import napari
from pathlib import Path
import numpy as np
import PyDAQmx as daq
import ctypes as ct
from image_post_processing import deskew
from napari.qt.threading import thread_worker
import time

# OPM control UI element            
@magicclass(labels=False)
@set_design(text="ASU Snouty-OPM control")
class OpmControl:

    # initialize
    def __init__(self):
        self.active_channel = "Off"
        self.channel_powers = np.zeros(5,dtype=np.int8)
        self.exposure_ms = 10.0             # unit: ms
        self.galvo_step = 0.4               # unit: um
        self.scan_axis_calibration = 0.043  # unit: V / um
        self.galvo_neutral_volt = -.15      # unit: V
        self.scan_axis_range_um = 50.0      # unit: um
        self.camera_pixel_size_um = .115    # unit: um
        self.opm_tilt = 30                  # unit: degrees
        self.channel_states=[False,False,False,False,False]
        self.path_to_mm_config = None

    # start pymmcore-plus
    def __post_init__(self):
        self.mmc = RemoteMMCore()
        self.mmc.loadSystemConfiguration(self.path_to_mm_config)
        
    # set 2D acquistion thread worker
    def _set_worker2d(self,worker_2d):
        self.worker_2d = worker_2d
        self.worker_2d_started = False
        self.worker_2d_running = False

    # set 3D acquistion thread worker
    def _set_worker3d(self,worker_3d):
        self.worker_3d = worker_3d
        self.worker_3d_started = False
        self.worker_3d_running = False

    # set viewer
    def _set_viewer(self,viewer):
        self.viewer = viewer

    @thread_worker
    def _acquire_2d_data(self):
        while True:

            with RemoteMMCore() as mmc_thread:

                # get raw image size
                mmc_thread.snapImage()
                test_image = mmc_thread.getImage()
                ny,nx=test_image.shape
                raw_image_2d = np.zeros([self.n_channels,ny,nx],dtype=np.uint16)

                mmc_thread.snapImage()
                raw_image_2d[self.active_channel,:] = mmc_thread.getImage()
                time.sleep(.01)
                yield raw_image_2d

    @thread_worker
    def _acquire_3d_data(self):

        while True:
            
            #------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------

            # set up lasers
            channel_labels = ["405", "488", "561", "635", "730"]
            channel_states = self.channel_states
            channel_powers = self.channel_powers
            do_ind = [0, 1, 2, 3, 4] # digital output line corresponding to each channel

            # parse which channels are active
            active_channel_indices = [ind for ind, st in zip(do_ind, channel_states) if st]
            n_active_channels = len(active_channel_indices)
            
            print("%d active channels: " % n_active_channels, end="")
            for ind in active_channel_indices:
                print("%s " % channel_labels[ind], end="")
            print("")

            # exposure time
            exposure_ms = self.exposure_ms #unit: ms

            # scan axis range
            scan_axis_range_um = self.scan_axis_range_um # unit: microns
            
            # galvo voltage at neutral
            galvo_neutral_volt = self.galvo_neutral_volt # unit: volts

            # galvo scan setup
            scan_axis_calibration = self.scan_axis_calibration # unit: V / um

            # scan step size
            scan_axis_step_um = self.galvo_step  # unit: um

            # camera pixel size
            pixel_size_um = self.camera_pixel_size_um # unit: um

            # opm tilt angle
            opm_tilt = self.opm_tilt # unit: degrees

            # deskew parameters
            deskew_parameters = np.empty([3])
            deskew_parameters[0] = opm_tilt                 # (degrees)
            deskew_parameters[1] = scan_axis_step_um*100    # (nm)
            deskew_parameters[2] = pixel_size_um*100        # (nm)

            # display data
            display_flag = False

            n_timepoints = 1

            #------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------End setup of scan parameters----------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------

            # give camera time to change modes if necessary
            self.mmc.setConfig('Camera-Setup','ScanMode3')
            self.mmc.waitForConfig('Camera-Setup','ScanMode3')

            # set camera to internal trigger
            self.mmc.setConfig('Camera-TriggerSource','INTERNAL')
            self.mmc.waitForConfig('Camera-TriggerSource','INTERNAL')
            
            # set camera to internal trigger
            # give camera time to change modes if necessary
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER KIND[0]','EXPOSURE')
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER KIND[1]','EXPOSURE')
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER KIND[2]','EXPOSURE')
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER POLARITY[0]','POSITIVE')
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER POLARITY[1]','POSITIVE')
            self.mmc.setProperty('OrcaFusionBT','OUTPUT TRIGGER POLARITY[2]','POSITIVE')

            # set exposure time
            self.mmc.setExposure(exposure_ms)

            # determine image size
            self.mmc.snapImage()
            test_image = self.mmc.getImage()
            y_pixels,x_pixels = test_image.shape
            
            # turn all lasers on
            self.mmc.setConfig('Laser','Off')
            self.mmc.waitForConfig('Laser','Off')

            # set all laser to external triggering
            self.mmc.setConfig('Modulation-405','External-Digital')
            self.mmc.waitForConfig('Modulation-405','External-Digital')
            self.mmc.setConfig('Modulation-488','External-Digital')
            self.mmc.waitForConfig('Modulation-488','External-Digital')
            self.mmc.setConfig('Modulation-561','External-Digital')
            self.mmc.waitForConfig('Modulation-561','External-Digital')
            self.mmc.setConfig('Modulation-637','External-Digital')
            self.mmc.waitForConfig('Modulation-637','External-Digital')
            self.mmc.setConfig('Modulation-730','External-Digital')
            self.mmc.waitForConfig('Modulation-730','External-Digital')

            # turn all lasers on
            self.mmc.setConfig('Laser','AllOn')
            self.mmc.waitForConfig('Laser','AllOn')

            self.mmc.setProperty('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
            self.mmc.setProperty('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
            self.mmc.setProperty('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
            self.mmc.setProperty('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
            self.mmc.setProperty('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

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
            print('Time points:  ' + str(n_timepoints))

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

            raw_image_stack = np.zeros([n_active_channels,scan_steps,y_pixels,x_pixels]).astype(np.uint16)

            self.mmc.startSequenceAcquisition(n_active_channels*scan_steps,0,True)
            for z in range(scan_steps):
                for c in active_channel_indices:
                    while self.mmc.getRemainingImageCount()==0:
                        pass
                    current_image = self.mmc.popNextImage()
                    raw_image_stack[c,z,:]=current_image
            self.mmc.stopSequenceAcquisition()

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

            for i in range(do_ind[-1]):
                if active_channel_indices[i]:
                    deskewed_image = deskew(np.flipud(raw_image_stack[0,i,:]),*deskew_parameters)    
                yield deskewed_image.astype(np.uint16)

    # reset lasers back to software control
    def _reset_lasers(self):
        
        # turn all lasers off
        self.mmc.setConfig('Laser','Off')
        self.mmc.waitForConfig('Laser','Off')

        # set all lasers back to software control
        self.mmc.setConfig('Modulation-405','CW (constant power)')
        self.mmc.waitForConfig('Modulation-405','CW (constant power)')
        self.mmc.setConfig('Modulation-488','CW (constant power)')
        self.mmc.waitForConfig('Modulation-488','CW (constant power)')
        self.mmc.setConfig('Modulation-561','CW (constant power)')
        self.mmc.waitForConfig('Modulation-561','CW (constant power)')
        self.mmc.setConfig('Modulation-637','CW (constant power)')
        self.mmc.waitForConfig('Modulation-637','CW (constant power)')
        self.mmc.setConfig('Modulation-730','CW (constant power)')
        self.mmc.waitForConfig('Modulation-730','CW (constant power)')

    # reset galvo controller back to neutral voltage
    def _reset_galvo(self):

        # put the galvo back to neutral
        # first, set the galvo to the initial point if it is not already
        taskAO_last = daq.Task()
        taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
        taskAO_last.WriteAnalogScalarF64(True, -1, self.galvo_neutral_volt, None)
        taskAO_last.StopTask()
        taskAO_last.ClearTask()
    
    # update viewer layers
    def _update_layers(self,new_image):
        channel_names = ['405nm','488nm','561nm','635nm','730nm']
        colormaps = ['bop purple','bop blue','bop orange','red','grey']

        for i in range(self.n_channels):
            if self.channel_states[i]:
                channel_name = channel_names[i]
                try:
                    self.viewer.layers[channel_name].data = new_image[i,:]
                except:
                    self.viewer.add_image(new_image[i,:], name=channel_name, colormap=colormaps[i],contrast_limits=[100,.9*np.max(new_image[i,:])])
    
    # set exposure time
    @magicgui(
        auto_call=True,
        exposure_ms={"widget_type": "FloatSpinBox", "min": 1, "max": 500,'label': 'Camera exposure (ms)'},
        layout='horizontal'
    )
    def set_exposure(self, exposure_ms=10.0):
        self.exposure_ms=exposure_ms

        # if in 2D mode, update software exposure setting on the fly
        if self.worker_2d_running:
            with RemoteMMCore() as mmc_exp:
                mmc_exp.setExposure(self.exposure_ms)

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
        self.channel_powers=[power_405,power_488,power_561,power_635,power_730]

        # if in 2D mode, update software power setting on the fly
        if self.worker_2d_running:
            with RemoteMMCore() as mmc_laser:
                pass

    # change active laser
    @magicgui(
        auto_call=True,
        active_channels = {"widget_type": "Select", "choices": ["Off","405","488","561","635","730"], "allow_multiple": True, "label": "Active channels"}
    )
    def set_active_channel(self, active_channels):
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

        # if in 2d mode, update active channel on the fly
        if self.worker_2d_running:
            with RemoteMMCore() as mmc_channel:
                mmc_channel.set_config()
    
    # set lateral galvo footprint 
    @magicgui(
        auto_call=True,
        galvo_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 200, "label": 'Galvo sweep (um)'},
        layout='horizontal'
    )
    def set_galvo_sweep(self, galvo_footprint_um=50.0):
        self.scan_axis_range_um=galvo_footprint_um

    # set lateral galvo step size
    @magicgui(
        auto_call=True,
        galvo_step={"widget_type": "FloatSpinBox", "min": 0, "max": 1, "label": 'Galvo step (um)'},
        layout='horizontal'
    )
    def set_galvo_step(self, galvo_step=0.4):
        self.galvo_step=galvo_step

    # control continuous 2D imaging (software triggering)
    @magicgui(
        auto_call=True,
        live_mode_3D={"widget_type": "PushButton", "label": 'Start/Stop Live (2D)'},
        layout='horizontal'
    )
    def live_mode_2D(self,live_mode_2D=False):

        if not(self.worker_3d_running):
            if self.worker_2d_running:
                self.worker_2d.pause()
                self.worker_2d_running = False
            else:
                if not(self.worker_2d_started):
                    self.worker_2d_started = True
                    self.worker_2d_running = True
                    self.worker_2d.start()
                else:
                    self.worker_2d.resume()
                    self.worker_2d_running = True

    # control continuous 3D volume (hardware triggering)
    @magicgui(
        auto_call=True,
        live_mode_3D={"widget_type": "PushButton", "label": 'Start/Stop live (3D)'},
        layout='horizontal'
    )
    def live_mode_3D(self,live_mode_3D):

        if not(self.worker_2d_running):
            if self.worker_3d_running:
                self.worker_3d.pause()
                self.worker_3d_running = False
                self._reset_lasers()
                self._reset_galvo()
            else:
                if not(self.worker_3d_started):
                    self.worker_3d.start()
                    self.worker_3d_started = True
                    self.worker_3d_running = True
                else:
                    self.worker_3d.resume()
                    self.worker_3d_running = True

def main():

    # setup OPM GUI and Napari viewer
    instrument_control_widget = OpmControl()
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(instrument_control_widget,name='ASU Snouty-OPM control')
    instrument_control_widget._set_viewer(viewer)

    # setup 2D imaging thread worker
    worker_2d = instrument_control_widget._acquire_2d_data()
    worker_2d.yielded.connect(instrument_control_widget._update_layer)
    instrument_control_widget._set_worker_2d(worker_2d)

    # setup 3D imaging thread worker 
    worker_3d = instrument_control_widget._acquire_3d_data()
    worker_3d.yielded.connect(instrument_control_widget._update_layer)
    instrument_control_widget._set_worker_3d(worker_3d)

    # start Napari
    napari.run()

    # shutdown threads
    worker_2d.quit()
    worker_3d.quit()

    # shutdown instrument
    instrument_control_widget.set_laser_power(0,0,0,0,0)
    instrument_control_widget._reset_lasers()
    instrument_control_widget._reset_galvo()

if __name__ == "__main__":
    main()