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
from reconstruction.image_post_processing import deskew
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
        self.channel_states=[False,False,False,False,False]
        self.exposure_ms = 10.0             # unit: ms
        self.scan_axis_step_um = 0.4        # unit: um
        self.scan_axis_calibration = 0.043  # unit: V / um
        self.galvo_neutral_volt = -.27      # unit: V
        self.scan_axis_range_um = 50.0      # unit: um
        self.camera_pixel_size_um = .115    # unit: um
        self.opm_tilt = 30                  # unit: degrees
        self.ROI_uleft_corner_x = int(200)  # unit: camera pixels
        self.ROI_uleft_corner_y = int(896)  # unit: camera pixels
        self.ROI_width_x = int(1900)        # unit: camera pixels
        self.ROI_width_y = int(512)         # unit: camera pixels
        self.path_to_mm_config = Path('C:/Program Files/Micro-Manager-2.0gamma/temp_HamDCAM.cfg')

        self.channel_labels = ["405", "488", "561", "637", "730"]
        self.do_ind = [0, 1, 2, 3, 4]       # digital output line corresponding to each channel

        self.debug=False

        self.powers_changed = True
        self.channels_changed = True
        self.ROI_changed = True
        self.exposure_changed = True
        self.footprint_changed = True
        self.galvo_step_changed = True
        self.DAQ_running = False

    # start pymmcore-plus
    def __post_init__(self):
        self.mmc = RemoteMMCore()
        self.mmc.loadSystemConfiguration(str(self.path_to_mm_config))

    # set 2D acquistion thread worker
    def _set_worker_2d(self,worker_2d):
        self.worker_2d = worker_2d
        self.worker_2d_started = False
        self.worker_2d_running = False
        
    # set 3D acquistion thread worker
    def _set_worker_3d(self,worker_3d):
        self.worker_3d = worker_3d
        self.worker_3d_started = False
        self.worker_3d_running = False

    # set viewer
    def _set_viewer(self,viewer):
        self.viewer = viewer

    # update viewer layers
    def _update_layers(self,new_image):
        channel_names = ['405nm','488nm','561nm','637nm','730nm']
        colormaps = ['bop purple','bop blue','bop orange','red','grey']

        for c in self.active_channel_indices:
            channel_name = channel_names[c]
            try:
                self.viewer.layers[channel_name].data = new_image[c,:]
            except:
                self.viewer.add_image(new_image[c,:], name=channel_name, blending='additive', colormap=colormaps[c],contrast_limits=[100,.9*np.max(new_image[c,:])])

    @thread_worker
    def _acquire_2d_data(self):
        while True:

            # parse which channels are active
            self.active_channel_indices = [ind for ind, st in zip(self.do_ind, self.channel_states) if st]
            self.n_active_channels = len(self.active_channel_indices)
            if self.n_active_channels == 0:
                yield None
                
            if self.debug:
                print("%d active channels: " % self.n_active_channels, end="")
                for ind in self.active_channel_indices:
                    print("%s " % self.channel_labels[ind], end="")
                print("")

            if self.powers_changed:
                self._set_mmc_laser_power()
                self.powers_changed = False

            with RemoteMMCore() as mmc_2d:

                if self.ROI_changed:
                    current_ROI = mmc_2d.getROI()
                    if not(current_ROI[2]==2304) or not(current_ROI[3]==2304):
                        mmc_2d.clearROI()
                        mmc_2d.waitForDevice('OrcaFusionBT')
                    mmc_2d.setROI(int(self.ROI_uleft_corner_x),int(self.ROI_uleft_corner_y),int(self.ROI_width_x),int(self.ROI_width_y))
                    mmc_2d.waitForDevice('OrcaFusionBT')
                    self.ROI_changed = False

                # set exposure time
                if self.exposure_changed:
                    mmc_2d.setExposure(self.exposure_ms)
                    self.exposure_changed = False

                # get raw image size
                raw_image_2d = np.zeros([self.do_ind[-1],self.ROI_width_y,self.ROI_width_x],dtype=np.uint16)

                for c in self.active_channel_indices:
                    mmc_2d.snapImage()
                    raw_image_2d[c,:] = mmc_2d.getImage()
                    time.sleep(.01)
                    yield raw_image_2d


    @thread_worker
    def _acquire_3d_data(self):

        while True:
            
            with RemoteMMCore() as mmc_3d:

                #------------------------------------------------------------------------------------------------------------------------------------
                #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------
                # parse which channels are active
                self.active_channel_indices = [ind for ind, st in zip(self.do_ind, self.channel_states) if st]
                self.n_active_channels = len(self.active_channel_indices)
                    
                if self.debug:
                    print("%d active channels: " % self.n_active_channels, end="")
                    for ind in self.active_channel_indices:
                        print("%s " % self.channel_labels[ind], end="")
                    print("")

                n_timepoints = 1

                if self.ROI_changed:
                    current_ROI = mmc_3d.getROI()
                    if not(current_ROI[2]==2304) or not(current_ROI[3]==2304):
                        mmc_3d.clearROI()
                        mmc_3d.waitForDevice('OrcaFusionBT')
                    mmc_3d.setROI(int(self.ROI_uleft_corner_x),int(self.ROI_uleft_corner_y),int(self.ROI_width_x),int(self.ROI_width_y))
                    mmc_3d.waitForDevice('OrcaFusionBT')
                    self.ROI_changed = False

                # set exposure time
                if self.exposure_changed:
                    mmc_3d.setExposure(self.exposure_ms)
                    self.exposure_changed = False

                if self.powers_changed:
                    self._set_mmc_laser_power()
                    self.powers_changed = False
                
                if self.footprint_changed:
                    # determine sweep footprint
                    self.min_volt = -(self.scan_axis_range_um * self.scan_axis_calibration / 2.) + self.galvo_neutral_volt # unit: volts
                    self.scan_axis_step_volts = self.scan_axis_step_um * self.scan_axis_calibration # unit: V
                    self.scan_axis_range_volts = self.scan_axis_range_um * self.scan_axis_calibration # unit: V
                    self.scan_steps = np.rint(self.scan_axis_range_volts / self.scan_axis_step_volts).astype(np.int16) # galvo steps

                if self.channels_changed or self.footprint_changed or not(self.DAQ_running):
                    if self.DAQ_running:
                        self._stop_DAQ()
                    self._create_DAQ_arrays()
                    self._start_DAQ()
                    self.raw_image_stack = np.zeros([self.do_ind[-1],self.scan_steps,self.ROI_width_y,self.ROI_width_x]).astype(np.uint16)

                    # change step size from physical space (nm) to camera space (pixels)
                    pixel_step = self.scan_axis_step_um/self.camera_pixel_size_um    # (pixels)

                    # calculate the number of pixels scanned during stage scan 
                    scan_end = self.scan_steps * pixel_step  # (pixels)

                    # calculate properties for final image
                    final_ny = np.int64(np.ceil(scan_end+self.ROI_width_y*np.cos(self.opm_tilt*np.pi/180))) # (pixels)
                    final_nz = np.int64(np.ceil(self.ROI_width_y*np.sin(self.opm_tilt*np.pi/180)))          # (pixels)
                    final_nx = np.int64(self.ROI_width_x)                                                   # (pixels)

                    deskewed_image = np.zeros([self.do_ind[-1],final_nz,final_ny,final_nx]).astype(np.uint16)

                    self.channels_changed = False
                    self.footprint_changed = False
                
                if self.debug:
                    # output experiment info
                    print("Scan axis range: %.1f um = %0.3fV, Scan axis step: %.1f nm = %0.3fV , Number of galvo positions: %d" % 
                        (self.scan_axis_range_um, self.scan_axis_range_volts, self.scan_axis_step_um * 1000, self.scan_axis_step_volts, self.scan_steps))
                    print('Galvo neutral (Volt): ' + str(self.galvo_neutral_volt)+', Min voltage (volt): '+str(self.min_volt))
                    print('Time points:  ' + str(n_timepoints))

                #------------------------------------------------------------------------------------------------------------------------------------
                #----------------------------------------------End setup of scan parameters----------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------


                #------------------------------------------------------------------------------------------------------------------------------------
                #----------------------------------------------Start acquisition and deskew----------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------

                # run hardware triggered acquisition
                mmc_3d.startSequenceAcquisition(int(self.n_active_channels*self.scan_steps),0,True)
                for z in range(self.scan_steps):
                    for c in self.active_channel_indices:
                        while mmc_3d.getRemainingImageCount()==0:
                            pass
                        self.raw_image_stack[c,z,:] = mmc_3d.popNextImage()
                mmc_3d.stopSequenceAcquisition()

                # deskew parameters
                deskew_parameters = np.empty([3])
                deskew_parameters[0] = self.opm_tilt                 # (degrees)
                deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
                deskew_parameters[2] = self.camera_pixel_size_um*100        # (nm)

                for c in self.active_channel_indices:
                    deskewed_image[c,:] = deskew(np.flipud(self.raw_image_stack[c,:]),*deskew_parameters).astype(np.uint16)    
                yield deskewed_image

                #------------------------------------------------------------------------------------------------------------------------------------
                #-----------------------------------------------End acquisition and deskew-----------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------

    # laser to hardware control
    def _lasers_to_hardware(self):

        with RemoteMMCore() as mmc_lasers_hardware:

            # turn all lasers off
            mmc_lasers_hardware.setConfig('Laser','Off')
            mmc_lasers_hardware.waitForConfig('Laser','Off')

            # set all laser to external triggering
            mmc_lasers_hardware.setConfig('Modulation-405','External-Digital')
            mmc_lasers_hardware.waitForConfig('Modulation-405','External-Digital')
            mmc_lasers_hardware.setConfig('Modulation-488','External-Digital')
            mmc_lasers_hardware.waitForConfig('Modulation-488','External-Digital')
            mmc_lasers_hardware.setConfig('Modulation-561','External-Digital')
            mmc_lasers_hardware.waitForConfig('Modulation-561','External-Digital')
            mmc_lasers_hardware.setConfig('Modulation-637','External-Digital')
            mmc_lasers_hardware.waitForConfig('Modulation-637','External-Digital')
            mmc_lasers_hardware.setConfig('Modulation-730','External-Digital')
            mmc_lasers_hardware.waitForConfig('Modulation-730','External-Digital')

            # turn all lasers on
            mmc_lasers_hardware.setConfig('Laser','AllOn')
            mmc_lasers_hardware.waitForConfig('Laser','AllOn')

    # lasers to software control
    def _lasers_to_software(self):

        with RemoteMMCore() as mmc_lasers_software:
        
            # turn all lasers off
            mmc_lasers_software.setConfig('Laser','Off')
            mmc_lasers_software.waitForConfig('Laser','Off')

            # set all lasers back to software control
            mmc_lasers_software.setConfig('Modulation-405','CW (constant power)')
            mmc_lasers_software.waitForConfig('Modulation-405','CW (constant power)')
            mmc_lasers_software.setConfig('Modulation-488','CW (constant power)')
            mmc_lasers_software.waitForConfig('Modulation-488','CW (constant power)')
            mmc_lasers_software.setConfig('Modulation-561','CW (constant power)')
            mmc_lasers_software.waitForConfig('Modulation-561','CW (constant power)')
            mmc_lasers_software.setConfig('Modulation-637','CW (constant power)')
            mmc_lasers_software.waitForConfig('Modulation-637','CW (constant power)')
            mmc_lasers_software.setConfig('Modulation-730','CW (constant power)')
            mmc_lasers_software.waitForConfig('Modulation-730','CW (constant power)')

    # reset galvo controller back to neutral voltage
    def _reset_galvo(self):

        # put the galvo back to neutral
        # first, set the galvo to the initial point if it is not already
        taskAO_last = daq.Task()
        taskAO_last.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
        taskAO_last.WriteAnalogScalarF64(True, -1, self.galvo_neutral_volt, None)
        taskAO_last.StopTask()
        taskAO_last.ClearTask()
    
    def _set_mmc_laser_power(self):

        with RemoteMMCore() as mmc_laser_power:

            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 405-100C - PowerSetpoint (%)',float(self.channel_powers[0]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 488-150C - PowerSetpoint (%)',float(self.channel_powers[1]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser OBIS LS 561-150 - PowerSetpoint (%)',float(self.channel_powers[2]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 637-140C - PowerSetpoint (%)',float(self.channel_powers[3]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 730-30C - PowerSetpoint (%)',float(self.channel_powers[4]))

    def _create_DAQ_arrays(self):
        # setup DAQ
        nvoltage_steps = self.scan_steps
        # 2 time steps per frame, except for first frame plus one final frame to reset voltage
        #samples_per_ch = (nvoltage_steps * 2 - 1) + 1
        self.samples_per_ch = (nvoltage_steps * 2 * self.n_active_channels - 1) + 1
        self.DAQ_sample_rate_Hz = 10000
        #retriggerable = True
        num_DI_channels = 8

        # Generate values for DO
        self.dataDO = np.zeros((self.samples_per_ch, num_DI_channels), dtype=np.uint8)
        for ii, ind in enumerate(self.active_channel_indices):
            self.dataDO[2*ii::2*self.n_active_channels, ind] = 1
        self.dataDO[-1, :] = 0

        # generate voltage steps
        max_volt = self.min_volt + self.scan_axis_range_volts  # 2
        voltage_values = np.linspace(self.min_volt, max_volt, nvoltage_steps)

        # Generate values for AO
        waveform = np.zeros(self.samples_per_ch)
        # one less voltage value for first frame
        waveform[0:2*self.n_active_channels - 1] = voltage_values[0]

        if len(voltage_values) > 1:
            # (2 * # active channels) voltage values for all other frames
            waveform[2*self.n_active_channels - 1:-1] = np.kron(voltage_values[1:], np.ones(2 * self.n_active_channels))
        
        # set back to initial value at end
        waveform[-1] = voltage_values[0]

        self.waveform = waveform

    def _start_DAQ(self):
        try:    
            # ----- DIGITAL input -------
            self.taskDI = daq.Task()
            self.taskDI.CreateDIChan("/Dev1/PFI0", "", daq.DAQmx_Val_ChanForAllLines)
            
            ## Configure change detectin timing (from wave generator)
            self.taskDI.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
            self.taskDI.CfgChangeDetectionTiming("/Dev1/PFI0", "/Dev1/PFI0", daq.DAQmx_Val_ContSamps, 0)

            ## Set where the starting trigger 
            self.taskDI.CfgDigEdgeStartTrig("/Dev1/PFI0", daq.DAQmx_Val_Rising)
            
            ## Export DI signal to unused PFI pins, for clock and start
            self.taskDI.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI2")
            self.taskDI.ExportSignal(daq.DAQmx_Val_StartTrigger, "/Dev1/PFI1")
            
            # ----- DIGITAL output ------   
            self.taskDO = daq.Task()
            # TO DO: Write each laser line separately!
            self.taskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)

            ## Configure timing (from DI task) 
            self.taskDO.CfgSampClkTiming("/Dev1/PFI2", self.DAQ_sample_rate_Hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, self.samples_per_ch)
            
            ## Write the output waveform
            samples_per_ch_ct_digital = ct.c_int32()
            self.taskDO.WriteDigitalLines(self.samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByChannel, self.dataDO, ct.byref(samples_per_ch_ct_digital), None)

            # ------- ANALOG output -----------

            # first, set the galvo to the initial point if it is not already
            self.taskAO_first = daq.Task()
            self.taskAO_first.CreateAOVoltageChan("/Dev1/ao0", "", -6.0, 6.0, daq.DAQmx_Val_Volts, None)
            self.taskAO_first.WriteAnalogScalarF64(True, -1, self.waveform[0], None)
            self.taskAO_first.StopTask()
            self.taskAO_first.ClearTask()

            # now set up the task to ramp the galvo
            self.taskAO = daq.Task()
            self.taskAO.CreateAOVoltageChan("/Dev1/ao0", "", -6.0, 6.0, daq.DAQmx_Val_Volts, None)

            ## Configure timing (from DI task)
            self.taskAO.CfgSampClkTiming("/Dev1/PFI2", self.DAQ_sample_rate_Hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, self.samples_per_ch)
            
            ## Write the output waveform
            samples_per_ch_ct = ct.c_int32()
            self.taskAO.WriteAnalogF64(self.samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByScanNumber, self.waveform, ct.byref(samples_per_ch_ct), None)

            ## ------ Start both tasks ----------
            self.taskAO.StartTask()    
            self.taskDO.StartTask()    
            self.taskDI.StartTask()

            self.DAQ_running = True

        except daq.DAQError as err:
            print("DAQmx Error %s"%err)

    def _stop_DAQ(self):
            # stop DAQ
            try:
                ## Stop and clear both tasks
                self.taskDI.StopTask()
                self.taskDO.StopTask()
                self.taskAO.StopTask()
                self.taskDI.ClearTask()
                self.taskAO.ClearTask()
                self.taskDO.ClearTask()

                self.DAQ_running = False
            except daq.DAQError as err:
                print("DAQmx Error %s"%err)

    def _setup_camera(self):
        with RemoteMMCore() as mmc_camera_setup:

            # give camera time to change modes if necessary
            mmc_camera_setup.setConfig('Camera-Setup','ScanMode3')
            mmc_camera_setup.waitForConfig('Camera-Setup','ScanMode3')

            # set camera to internal trigger
            mmc_camera_setup.setConfig('Camera-TriggerSource','INTERNAL')
            mmc_camera_setup.waitForConfig('Camera-TriggerSource','INTERNAL')
            
            # set camera to internal trigger
            # give camera time to change modes if necessary
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER KIND[0]','EXPOSURE')
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER KIND[1]','EXPOSURE')
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER KIND[2]','EXPOSURE')
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER POLARITY[0]','POSITIVE')
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER POLARITY[1]','POSITIVE')
            mmc_camera_setup.setProperty('OrcaFusionBT',r'OUTPUT TRIGGER POLARITY[2]','POSITIVE')

    # startup instrument
    def _startup(self):
        self._set_mmc_laser_power()
        self._lasers_to_hardware()
        self._reset_galvo()
        self._setup_camera()

    # shutdown instrument
    def _shutdown(self):
        self._set_mmc_laser_power()
        self._lasers_to_software()
        if self.DAQ_running:
            self._stop_DAQ()
        self._reset_galvo()

    # set exposure time
    @magicgui(
        auto_call=True,
        exposure_ms={"widget_type": "FloatSpinBox", "min": 1, "max": 500,'label': 'Camera exposure (ms)'},
        layout='horizontal'
    )
    def set_exposure(self, exposure_ms=10.0):
        if not(exposure_ms == self.exposure_ms):
            self.exposure_ms=exposure_ms
            self.exposure_changed = True
        else:
            self.exposure_changed = False

    # set camera crop
    @magicgui(
        auto_call=True,
        uleft_corner_x={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI center (non-tilt)'},
        uleft_corner_y={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI center (tilt)'},
        width_x={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI width (non-tilt)'},
        width_y={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI height (tilt)'},
        layout='vertical'
    )
    def set_ROI(self, uleft_corner_x=200,uleft_corner_y=896,width_x=1800,width_y=512):
        
        if not(int(uleft_corner_x)==self.ROI_uleft_corner_x) or not(int(uleft_corner_y)==self.ROI_uleft_corner_y) or not(int(width_x)==self.ROI_width_x) or not(int(width_y)==self.ROI_width_y):
            self.ROI_uleft_corner_x=int(uleft_corner_x)
            self.ROI_uleft_corner_y=int(uleft_corner_y)
            self.ROI_width_x=int(width_x)
            self.ROI_width_y=int(width_y)
            self.ROI_changed = True
        else:
            self.ROI_changed = False

    # set laser power(s)
    @magicgui(
        auto_call=True,
        power_405={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '405nm power (%)'},
        power_488={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '488nm power (%)'},
        power_561={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '561nm power (%)'},
        power_637={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '637nm power (%)'},
        power_730={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '730nm power (%)'},
        layout='vertical'
    )
    def set_laser_power(self, power_405=0.0, power_488=0.0, power_561=0.0, power_637=0.0, power_730=0.0,):
        channel_powers = [power_405,power_488,power_561,power_637,power_730]

        if not(channel_powers[:] == self.channel_powers[:]):
            self.channel_powers=channel_powers
            self.powers_changed = True
        else:
            self.powers_changed = False
        
    # set active laser(s)
    @magicgui(
        auto_call=True,
        active_channels = {"widget_type": "Select", "choices": ["Off","405","488","561","637","730"], "allow_multiple": True, "label": "Active channels"}
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
            elif channel == '637':
                states[3]='True'
            elif channel == '730':
                states[4]='True'

        if not(states==self.channel_states):
            self.channel_states=states
            self.channels_changed = True
        else:
            self.channels_changed = False

    # set lateral galvo footprint 
    @magicgui(
        auto_call=True,
        galvo_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 200, "label": 'Galvo sweep (um)'},
        layout='horizontal'
    )
    def set_galvo_sweep(self, galvo_footprint_um=50.0):

        if not(galvo_footprint_um==self.scan_axis_range_um):
            self.galvo_footprint_um=galvo_footprint_um
            self.footprint_changed = True
        else:
            self.footprint_changed = False

    # set lateral galvo step size
    @magicgui(
        auto_call=True,
        galvo_step={"widget_type": "FloatSpinBox", "min": 0, "max": 1, "label": 'Galvo step (um)'},
        layout='horizontal'
    )
    def set_galvo_step(self, galvo_step=0.4):

        if not(galvo_step==self.galvo_step):
            self.galvo_step=galvo_step
            self.galvo_step_changed = True
        else:
            self.galvo_step_changed = False

    
    # control continuous 2D imaging (software triggering)
    @magicgui(
        auto_call=True,
        live_mode_2D={"widget_type": "PushButton", "label": 'Start/Stop Live (2D)'},
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
                self._stop_DAQ()
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
    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    instrument_control_widget._startup()

    viewer = napari.Viewer()

    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    instrument_control_widget._set_viewer(viewer)

    # setup 2D imaging thread worker
    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    worker_2d = instrument_control_widget._acquire_2d_data()
    worker_2d.yielded.connect(instrument_control_widget._update_layers)
    instrument_control_widget._set_worker_2d(worker_2d)
    
    # setup 3D imaging thread worker 
    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    worker_3d = instrument_control_widget._acquire_3d_data()
    worker_3d.yielded.connect(instrument_control_widget._update_layers)
    instrument_control_widget._set_worker_3d(worker_3d)

    viewer.window.add_dock_widget(instrument_control_widget,name='ASU Snouty-OPM control')

    # start Napari
    napari.run()

    # shutdown threads
    worker_2d.quit()
    worker_3d.quit()

    # shutdown instrument
    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    instrument_control_widget._shutdown()

if __name__ == "__main__":
    main()