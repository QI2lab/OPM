from pymmcore_plus import CMMCorePlus
from magicclass import magicclass, MagicTemplate
from magicgui import magicgui
from magicgui.tqdm import trange
from napari.qt.threading import thread_worker
import easygui

from pathlib import Path
import numpy as np
import time
import zarr

from src.hardware.OPMNIDAQ import OPMNIDAQ
from src.utils.data_io import write_metadata
from src.utils.image_post_processing import deskew
from datetime import datetime

# OPM control UI element            
@magicclass(labels=False)
class OPMMirrorScan(MagicTemplate):

    # initialize
    def __init__(self, mmc):
        # MicroManagerCore
        self.mmc = mmc
        # OPM parameters
        self.active_channel = "Off"
        self.channel_powers = np.zeros(5,dtype=np.int8)
        self.channel_states=[False,False,False,False,False]
        self.exposure_ms = 10.0                 # unit: ms
        self.scan_axis_step_um = 0.4            # unit: um
        self.scan_axis_calibration = 0.043      # unit: V / um
        self.galvo_neutral_volt = -.27          # unit: V
        self.scan_mirror_footprint_um = 50.0      # unit: um
        self.camera_pixel_size_um = .115        # unit: um
        self.opm_tilt = 30                      # unit: degrees

        # camera parameters
        self.camera_name = 'OrcaFusionBT'   # camera name in MM config
        self.ROI_uleft_corner_x = int(200)  # unit: camera pixels
        self.ROI_uleft_corner_y = int(896)  # unit: camera pixels
        self.ROI_width_x = int(1900)        # unit: camera pixels
        self.ROI_width_y = int(512)         # unit: camera pixels

        self.save_path = Path('D:/')

        self.channel_labels = ["405", "488", "561", "635", "730"]
        self.do_ind = [0, 1, 2, 3, 4]       # digital output line corresponding to each channel

        self.debug=False

        # flags for instrument setup
        self.powers_changed = True
        self.channels_changed = True
        self.ROI_changed = True
        self.exposure_changed = True
        self.footprint_changed = True
        self.DAQ_running = False
        self.save_path_setup = False

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

    def _create_3d_t_worker(self):
        worker_3d_t = self._acquire_3d_t_data()
        self._set_worker_3d_t(worker_3d_t)
        

    # set 3D timelapse acquistion thread worker
    def _set_worker_3d_t(self,worker_3d_t):
        self.worker_3d_t = worker_3d_t
        self.worker_3d_t_running = False

    # set viewer
    def _set_viewer(self,viewer):
        self.viewer = viewer

    # create and save metadata
    def _save_metadata(self):
        scan_param_data = [{'root_name': str("OPM_data"),
                            'scan_type': 'galvo',
                            'theta': self.opm_tilt, 
                            'exposure_ms': self.exposure_ms,
                            'scan_step': self.scan_axis_step_um, 
                            'pixel_size': self.camera_pixel_size_um,
                            'galvo_scan_range_um': self.scan_mirror_footprint_um,
                            'galvo_volts_per_um': self.scan_axis_calibration, 
                            'num_t': int(self.n_timepoints),
                            'time_delay': float(self.wait_time),
                            'num_y': 1, 
                            'num_z': 1,
                            'num_ch': int(self.n_active_channels),
                            'scan_axis_positions': int(self.scan_steps),
                            'y_pixels': self.ROI_width_y,
                            'x_pixels': self.ROI_width_x,
                            '405_active': self.channel_states[0],
                            '488_active': self.channel_states[1],
                            '561_active': self.channel_states[2],
                            '635_active': self.channel_states[3],
                            '730_active': self.channel_states[4],
                            '405_power': self.channel_powers[0],
                            '488_power': self.channel_powers[1],
                            '561_power': self.channel_powers[2],
                            '635_power': self.channel_powers[3],
                            '730_power': self.channel_powers[4],
                            }]
        
        write_metadata(scan_param_data[0], self.output_dir_path / Path('scan_metadata.csv'))

    # update viewer layers
    def _update_layers(self,values):
        current_channel = values[0]
        new_image = values[1]
        channel_names = ['405nm','488nm','561nm','635nm','730nm']
        colormaps = ['bop purple','bop blue','bop orange','red','grey']

        channel_name = channel_names[current_channel]
        colormap = colormaps[current_channel]
        try:
            self.viewer.layers[channel_name].data = new_image
        except:
            self.viewer.add_image(new_image, name=channel_name, blending='additive', colormap=colormap,contrast_limits=[110,.9*np.max(new_image)])

    @thread_worker
    def _acquire_2d_data(self):
        while True:

            # parse which channels are active
            active_channel_indices = [ind for ind, st in zip(self.do_ind, self.channel_states) if st]
            n_active_channels = len(active_channel_indices)
            if n_active_channels == 0:
                yield None
                
            if self.debug:
                print("%d active channels: " % n_active_channels, end="")
                for ind in active_channel_indices:
                    print("%s " % self.channel_labels[ind], end="")
                print("")

            if self.powers_changed:
                self._set_mmc_laser_power()
                self.powers_changed = False

            if self.channels_changed:
                if self.DAQ_running:
                    self.opmdaq.stop_waveform_playback()
                    self.DAQ_running = False
                    self.opmdaq.reset_scan_mirror()
                self.opmdaq.set_scan_type('stage')
                self.opmdaq.set_channels_to_use(self.channel_states)
                self.opmdaq.set_interleave_mode(True)
                self.opmdaq.generate_waveforms()
                self.opmdaq.start_waveform_playback()
                self.DAQ_running=True
                self.channels_changed = False
            
            if not(self.DAQ_running):
                self.opmdaq.start_waveform_playback()
                self.DAQ_running=True

            
            if self.ROI_changed:

                self._crop_camera()
                self.ROI_changed = False

            # set exposure time
            if self.exposure_changed:
                self.mmc.setExposure(self.exposure_ms)
                self.exposure_changed = False

            for c in active_channel_indices:
                self.mmc.snapImage()
                raw_image_2d = self.mmc.getImage()
                time.sleep(.05)
                yield c, raw_image_2d

    @thread_worker
    def _acquire_3d_data(self):

        while True:

            #------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------
            # parse which channels are active
            active_channel_indices = [ind for ind, st in zip(self.do_ind, self.channel_states) if st]
            n_active_channels = len(active_channel_indices)
                
            if self.debug:
                print("%d active channels: " % n_active_channels, end="")
                for ind in active_channel_indices:
                    print("%s " % self.channel_labels[ind], end="")
                print("")

            n_timepoints = 1

            if self.ROI_changed:
                self._crop_camera()
                self.ROI_changed = False

            # set exposure time
            if self.exposure_changed:
                self.mmc.setExposure(self.exposure_ms)
                self.exposure_changed = False

            if self.powers_changed:
                self._set_mmc_laser_power()
                self.powers_changed = False
            
            if self.channels_changed or self.footprint_changed or not(self.DAQ_running):
                if self.DAQ_running:
                    self.opmdaq.stop_waveform_playback()
                    self.DAQ_running = False
                self.opmdaq.set_scan_type('mirror')
                self.opmdaq.set_channels_to_use(self.channel_states)
                self.opmdaq.set_interleave_mode(True)
                scan_steps = self.opmdaq.set_scan_mirror_range(self.scan_axis_step_um,self.scan_mirror_footprint_um)
                self.opmdaq.generate_waveforms()
                self.channels_changed = False
                self.footprint_changed = False

            raw_image_stack = np.zeros([self.do_ind[-1],scan_steps,self.ROI_width_y,self.ROI_width_x]).astype(np.uint16)
            
            if self.debug:
                # output experiment info
                print("Scan axis range: %.1f um, Scan axis step: %.1f nm, Number of galvo positions: %d" % 
                    (self.scan_mirror_footprint_um,  self.scan_axis_step_um * 1000, scan_steps))
                print('Time points:  ' + str(n_timepoints))

            #------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------End setup of scan parameters----------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------


            #------------------------------------------------------------------------------------------------------------------------------------
            #----------------------------------------------Start acquisition and deskew----------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------

            self.opmdaq.start_waveform_playback()
            self.DAQ_running = True
            # run hardware triggered acquisition
            self.mmc.startSequenceAcquisition(int(n_active_channels*scan_steps),0,True)
            for z in range(scan_steps):
                for c in active_channel_indices:
                    while self.mmc.getRemainingImageCount()==0:
                        pass
                    raw_image_stack[c,z,:] = self.mmc.popNextImage()
            self.mmc.stopSequenceAcquisition()
            self.opmdaq.stop_waveform_playback()
            self.DAQ_running = False

            # deskew parameters
            deskew_parameters = np.empty([3])
            deskew_parameters[0] = self.opm_tilt                 # (degrees)
            deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
            deskew_parameters[2] = self.camera_pixel_size_um*100 # (nm)

            for c in active_channel_indices:
                deskewed_image = deskew(np.flipud(raw_image_stack[c,:]),*deskew_parameters).astype(np.uint16)  
                yield c, deskewed_image

            del raw_image_stack

            #------------------------------------------------------------------------------------------------------------------------------------
            #-----------------------------------------------End acquisition and deskew-----------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------

    @thread_worker
    def _acquire_3d_t_data(self):

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------
        # parse which channels are active
        active_channel_indices = [ind for ind, st in zip(self.do_ind, self.channel_states) if st]
        self.n_active_channels = len(active_channel_indices)
            
        if self.debug:
            print("%d active channels: " % self.n_active_channels, end="")
            for ind in active_channel_indices:
                print("%s " % self.channel_labels[ind], end="")
            print("")

        if self.ROI_changed:
            self._crop_camera()
            self.ROI_changed = False

        # set exposure time
        if self.exposure_changed:
            self.mmc.setExposure(self.exposure_ms)
            self.exposure_changed = False

        if self.powers_changed:
            self._set_mmc_laser_power()
            self.powers_changed = False
        
        if self.channels_changed or self.footprint_changed or not(self.DAQ_running):
            if self.DAQ_running:
                self.opmdaq.stop_waveform_playback()
                self.DAQ_running = False
            self.opmdaq.set_scan_type('mirror')
            self.opmdaq.set_channels_to_use(self.channel_states)
            self.opmdaq.set_interleave_mode(True)
            self.scan_steps = self.opmdaq.set_scan_mirror_range(self.scan_axis_step_um,self.scan_mirror_footprint_um)
            self.opmdaq.generate_waveforms()
            self.channels_changed = False
            self.footprint_changed = False

        # create directory for timelapse
        time_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        self.output_dir_path = self.save_path / Path('timelapse_'+time_string)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)


        # create name for zarr directory
        zarr_output_path = self.output_dir_path / Path('OPM_data.zarr')

        # create and open zarr file
        opm_data = zarr.open(str(zarr_output_path), mode="w", shape=(self.n_timepoints, self.n_active_channels, self.scan_steps, self.ROI_width_y, self.ROI_width_x), chunks=(1, 1, 1, self.ROI_width_y, self.ROI_width_x),compressor=None, dtype=np.uint16)

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------End setup of scan parameters----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------


        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------Start acquisition---------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # set circular buffer to be large
        self.mmc.clearCircularBuffer()
        circ_buffer_mb = 96000
        self.mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

        # run hardware triggered acquisition
        if self.wait_time == 0:
            self.opmdaq.start_waveform_playback()
            self.DAQ_running = True
            self.mmc.startSequenceAcquisition(int(self.n_timepoints*self.n_active_channels*self.scan_steps),0,True)
            for t in trange(self.n_timepoints,desc="t", position=0):
                for z in trange(self.scan_steps,desc="z", position=1, leave=False):
                    for c in range(self.n_active_channels):
                        while self.mmc.getRemainingImageCount()==0:
                            pass
                        opm_data[t, c, z, :, :]  = self.mmc.popNextImage()
            self.mmc.stopSequenceAcquisition()
            self.opmdaq.stop_waveform_playback()
            self.DAQ_running = False
        else:
            for t in trange(self.n_timepoints,desc="t", position=0):
                self.opmdaq.start_waveform_playback()
                self.DAQ_running = True
                self.mmc.startSequenceAcquisition(int(self.n_active_channels*self.scan_steps),0,True)
                for z in trange(self.scan_steps,desc="z", position=1, leave=False):
                    for c in range(self.n_active_channels):
                        while self.mmc.getRemainingImageCount()==0:
                            pass
                        opm_data[t, c, z, :, :]  = self.mmc.popNextImage()
                self.mmc.stopSequenceAcquisition()
                self.opmdaq.stop_waveform_playback()
                self.DAQ_running = False
                time.sleep(self.wait_time)
                
        # construct metadata and save
        self._save_metadata()

        #------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------End acquisition-------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # set circular buffer to be small 
        self.mmc.clearCircularBuffer()
        circ_buffer_mb = 4000
        self.mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

    def _crop_camera(self):
        """
        Crop camera to GUI values

        :return None:
        """

        current_ROI = self.mmc.getROI()
        if not(current_ROI[2]==2304) or not(current_ROI[3]==2304):
            self.mmc.clearROI()
            self.mmc.waitForDevice(self.camera_name)
        self.mmc.setROI(int(self.ROI_uleft_corner_x),int(self.ROI_uleft_corner_y),int(self.ROI_width_x),int(self.ROI_width_y))
        self.mmc.waitForDevice(self.camera_name)

    def _lasers_to_hardware(self):
        """
        Change lasers to hardware control

        :return None:
        """

        # turn all lasers off
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

    def _lasers_to_software(self):
        """
        Change lasers to software control

        :return None:
        """

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

    def _set_mmc_laser_power(self):
        """
        Change laser power

        :return None:
        """
        
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 405-100C - PowerSetpoint (%)',float(self.channel_powers[0]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 488-150C - PowerSetpoint (%)',float(self.channel_powers[1]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser OBIS LS 561-150 - PowerSetpoint (%)',float(self.channel_powers[2]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 637-140C - PowerSetpoint (%)',float(self.channel_powers[3]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 730-30C - PowerSetpoint (%)',float(self.channel_powers[4]))

    def _setup_camera(self):
        """
        Setup camera readout and triggering for OPM

        :return None:
        """

        # give camera time to change modes if necessary
        self.mmc.setConfig('Camera-Setup','ScanMode3')
        self.mmc.waitForConfig('Camera-Setup','ScanMode3')

        # set camera to internal trigger
        self.mmc.setConfig('Camera-TriggerType','NORMAL')
        self.mmc.waitForConfig('Camera-TriggerType','NORMAL')
        trigger_value = self.mmc.getProperty(self.camera_name,'Trigger')
        while not(trigger_value == 'NORMAL'):
            self.mmc.setConfig('Camera-TriggerType','NORMAL')
            self.mmc.waitForConfig('Camera-TriggerType','NORMAL')
            time.sleep(2)
            trigger_value = self.mmc.getProperty(self.camera_name,'Trigger')
            
        # give camera time to change modes if necessary
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[0]','EXPOSURE')
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[1]','EXPOSURE')
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[2]','EXPOSURE')
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[0]','POSITIVE')
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[1]','POSITIVE')
        self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[2]','POSITIVE')

    def _enforce_DCAM_internal_trigger(self):
        """
        Enforce camera being in trigger = INTERNAL mode

        :return None:
        """

        # set camera to START mode upon input trigger
        self.mmc.setConfig('Camera-TriggerSource','INTERNAL')
        self.mmc.waitForConfig('Camera-TriggerSource','INTERNAL')

        # check if camera actually changed
        # we find that camera doesn't always go back to START mode and need to check it
        trigger_value = self.mmc.getProperty(self.camera_name,'TRIGGER SOURCE')
        while not(trigger_value == 'INTERNAL'):
            self.mmc.setConfig('Camera-TriggerSource','INTERNAL')
            self.mmc.waitForConfig('Camera-TriggerSource','INTERNAL')
            trigger_value = self.mmc.getProperty(self.camera_name,'TRIGGER SOURCE')

            
    def _startup(self):
        """
        Startup OPM instrument in neutral state for all hardware

        :return None:
        """

        # set lasers to 0% power and hardware control
        self._set_mmc_laser_power()
        self._lasers_to_hardware()

        # set camera to OPM specific setup
        self._crop_camera()
        self._setup_camera()
        self._enforce_DCAM_internal_trigger()

        # connect to DAQ
        self.opmdaq = OPMNIDAQ(scan_mirror_neutral=self.galvo_neutral_volt)
        # reset scan mirror position to neutral
        self.opmdaq.reset_scan_mirror()

    def _shutdown(self):
        """
        Shutdown OPM instrument in neutral state for all hardware

        :return None:
        """
        
        # set lasers to 0% power and software control
        self._set_mmc_laser_power()
        self._lasers_to_software()

        # shutdown DAQ
        if self.DAQ_running:
            self.opmdaq.stop_waveform_playback()
        self.opmdaq.reset_scan_mirror()

    @magicgui(
        auto_call=False,
        exposure_ms={"widget_type": "FloatSpinBox", "min": 1, "max": 500,'label': 'Camera exposure (ms)'},
        layout='horizontal',
        call_button='Update exposure'
    )
    def set_exposure(self, exposure_ms=10.0):
        """
        Magicgui element to get camera exposure time

        :param exposure_ms: float
            camera exposure time
        :return None:
        """

        if not(exposure_ms == self.exposure_ms):
            self.exposure_ms=exposure_ms
            self.exposure_changed = True
        else:
            self.exposure_changed = False

    @magicgui(
        auto_call=False,
        uleft_corner_x={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI center (non-tilt)'},
        uleft_corner_y={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI center (tilt)'},
        width_x={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI width (non-tilt)'},
        width_y={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI height (tilt)'},
        layout='vertical', 
        call_button="Update crop"
    )
    def set_ROI(self, uleft_corner_x=200,uleft_corner_y=896,width_x=1800,width_y=512):
        """
        Magicgui element to get camera ROI

        :param uleft_corner_x: int
            upper left ROI x pixel
        :param uleft_corner_y: int
            upper left ROI y pixel
        :param width_x: int
            ROI width in pixels
        :param width_y: int
            ROI height in pixels = TILTED DIRECTION
        :return None:
        """
        
        if not(int(uleft_corner_x)==self.ROI_uleft_corner_x) or not(int(uleft_corner_y)==self.ROI_uleft_corner_y) or not(int(width_x)==self.ROI_width_x) or not(int(width_y)==self.ROI_width_y):
            self.ROI_uleft_corner_x=int(uleft_corner_x)
            self.ROI_uleft_corner_y=int(uleft_corner_y)
            self.ROI_width_x=int(width_x)
            self.ROI_width_y=int(width_y)
            self.ROI_changed = True
        else:
            self.ROI_changed = False

    @magicgui(
        auto_call=False,
        power_405={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '405nm power (%)'},
        power_488={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '488nm power (%)'},
        power_561={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '561nm power (%)'},
        power_635={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '635nm power (%)'},
        power_730={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '730nm power (%)'},
        layout='vertical',
        call_button='Update powers'
    )
    def set_laser_power(self, power_405=0.0, power_488=0.0, power_561=0.0, power_635=0.0, power_730=0.0):
        """
        Magicgui element to get laser powers (0-100%)

        :param power_405: float
            405 nm laser power
        :param power_488: float
            488 nm laser power
        :param power_561: float
            561 nm laser power
        :param power_635: float
            635 nm laser power
        :param power_730: float
            730 nm laser power
        :return None:
        """

        channel_powers = [power_405,power_488,power_561,power_635,power_730]

        if not(np.all(channel_powers == self.channel_powers)):
            self.channel_powers=channel_powers
            self.powers_changed = True
        else:
            self.powers_changed = False
        
    @magicgui(
        auto_call=True,
        active_channels = {"widget_type": "Select", "choices": ["Off","405","488","561","635","730"], "allow_multiple": True, "label": "Active channels"}
    )
    def set_active_channel(self, active_channels):
        """
        Magicgui element to set active lasers

        :param active_channels: list
            list of booleans, one for each laser channel
        :return None:
        """

        states = [False,False,False,False,False]
        for channel in active_channels:
            if channel == 'Off':
                states = [False,False,False,False,False]
                break
            if channel == '405':
                states[0]=True
            elif channel == '488':
                states[1]=True
            elif channel == '561':
                states[2]=True
            elif channel == '635':
                states[3]=True
            elif channel == '730':
                states[4]=True

        if not(np.all(states == self.channel_states)):
            self.channel_states=states
            self.channels_changed = True
        else:
            self.channels_changed = False

    @magicgui(
        auto_call=False,
        scan_mirror_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 225, "label": 'Mirror sweep (um)'},
        layout='horizontal',
        call_button='Update scan range'
    )
    def set_galvo_sweep(self, scan_mirror_footprint_um=50.0):
        """
        Magicgui element to set scan footprint

        :param scan_mirror_footprint_um: float
            size of scan mirror sweep in microns
        :return None:
        """

        if not(scan_mirror_footprint_um==self.scan_mirror_footprint_um):
            self.scan_mirror_footprint_um=scan_mirror_footprint_um
            self.footprint_changed = True
        else:
            self.footprint_changed = False

    # control continuous 2D imaging (software triggering)
    @magicgui(
        auto_call=True,
        live_mode_2D={"widget_type": "PushButton", "label": 'Start/Stop Live (2D)'},
        layout='horizontal'
    )
    def live_mode_2D(self,live_mode_2D=False):

        if (np.any(self.channel_states)):
            if not(self.worker_3d_running) and not(self.worker_3d_t_running):
                if self.worker_2d_running:
                    self.worker_2d.pause()
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running=False
                    self.worker_2d_running = False
                else:
                    if not(self.worker_2d_started):
                        self.worker_2d_started = True
                        self.worker_2d_running = True
                        self.worker_2d.start()
                    else:
                        self.worker_2d.resume()
                        self.worker_2d_running = True
            else:
                if self.worker_3d_running:
                    raise Exception('Stop live 3D acquisition first.')
                elif self.worker_iterative_running:
                    raise Exception('Iterative acquisition in process.')
                else:
                    raise Exception('Unknown error.')
        else:
            raise Exception('Set at least one active channel before starting.')
    

    # control continuous 3D volume (hardware triggering)
    @magicgui(
        auto_call=True,
        live_mode_3D={"widget_type": "PushButton", "label": 'Start/Stop live (3D)'},
        layout='horizontal'
    )
    def live_mode_3D(self,live_mode_3D):

        if (np.any(self.channel_states)):
            if not(self.worker_2d_running) and not(self.worker_3d_t_running):
                self.galvo_scan = True
                if self.worker_3d_running:
                    self.worker_3d.pause()
                    self.worker_3d_running = False
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running = False
                    self.opmdaq.reset_scan_mirror()
                else:
                    if not(self.worker_3d_started):
                        self.worker_3d.start()
                        self.worker_3d_started = True
                        self.worker_3d_running = True
                    else:
                        self.worker_3d.resume()
                        self.worker_3d_running = True
            else:
                if self.worker_2d_running:
                    raise Exception('Stop live 2D acquisition first.')
                elif self.worker_iterative_running:
                    raise Exception('Iterative acquisition in process.')
                else:
                    raise Exception('Unknown error.')
        else:
            raise Exception('Set at least one active channel before starting.')

    # set timelapse parameters
    @magicgui(
        auto_call=True,
        n_timepoints={"widget_type": "SpinBox", "min": 0, "max": 10000, "label": 'Timepoints to acquire'},
        wait_time={"widget_type": "FloatSpinBox", "min": 0, "max": 240, "label": 'Delay between timepoints (s)'},
        layout='horizontal'
    )
    def set_timepoints(self, n_timepoints=1,wait_time=0):
        self.n_timepoints = n_timepoints
        self.wait_time = wait_time
        self.timelapse_setup = True

    # set filepath for saving data
    @magicgui(
        auto_call=False,
        save_path={"widget_type": "FileEdit","mode": "d", "label": 'Save path:'},
        layout='horizontal', 
        call_button="Set"
    )
    def set_save_path(self, save_path='d:/'):
        self.save_path = Path(save_path)
        self.save_path_setup = True

    # control timelapse 3D volume (hardware triggering)
    @magicgui(
        auto_call=True,
        timelapse_mode_3D={"widget_type": "PushButton", "label": 'Start/Stop live (3D)'},
        layout='horizontal'
    )
    def timelapse_mode_3D(self,timelapse_mode_3D):
        if not(self.worker_2d_running) and not(self.worker_3d_running):
            if (self.save_path_setup and self.timelapse_setup):
                self.worker_3d_t.start()
                self.worker_3d_t.returned.connect(self._create_3d_t_worker)
                self.worker_3d_t_running = True
            else:
                raise Exception('Setup save path and timelapse first.')
        else:
            raise Exception('Stop active live mode first.')

    
    @magicgui(call_button='Save images')
    def save_displayed_images(self):
        """
        Save all the displayed images in a single zarr image on disk.
        """
        nb_ch = len(self.viewer.layers)
        if nb_ch == 0:
            print("There is no displayed image to save")
        else:
            print("Saving images")
            if not self.save_path_setup:
                save_path = easygui.diropenbox('Path to save images')
                if save_path is None:
                    print("No folder selected, abort saving images")
                    return
                self.save_path = Path(save_path)
                self.save_path_setup = True
                print("Save images in folder", self.save_path.as_posix())

            # make the multichannel image stack
            im_shape = list(self.viewer.layers[0].data.shape)
            if len(im_shape) <= 3:
                full_shape = [nb_ch] + im_shape
                chunk_shape = [1] + im_shape
            else:
                # there is time too
                full_shape = im_shape.copy()
                full_shape.insert(nb_ch, 1)
                chunk_shape = [1, 1] + im_shape
            # create directory for saved images
            time_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
            self.output_dir_path = self.save_path / Path('export_'+time_string)
            self.output_dir_path.mkdir(parents=True, exist_ok=True)
            zarr_output_path = self.output_dir_path / Path('OPM_data.zarr')
            opm_data = zarr.open(str(zarr_output_path), mode="w", shape=full_shape, chunks=chunk_shape, compressor=None, dtype=np.uint16)
            if len(im_shape) <= 3:
                for c in range(nb_ch):
                    opm_data[c] = self.viewer.layers[0].data
            else:
                for t in range(im_shape[0]):
                    for c in nb_ch:
                        opm_data[t, c, :, :, :] = self.viewer.layers[0].data
            print('Image saved')


    @magicgui(call_button='Process image')
    def process_loaded_image(self):
        """
        Split channels and deskew the first available image.
        """
        if len(self.viewer.layers) == 0:
            print("There is no displayed image to process")
        else:
            print("Processing image")
            # # deskew parameters
            # deskew_parameters = np.empty([3])
            # deskew_parameters[0] = self.opm_tilt                 # (degrees)
            # deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
            # deskew_parameters[2] = self.camera_pixel_size_um*100 # (nm)

            # for c in active_channel_indices:
            #     deskewed_image = deskew(np.flipud(raw_image_stack[c,:]),*deskew_parameters).astype(np.uint16)  
