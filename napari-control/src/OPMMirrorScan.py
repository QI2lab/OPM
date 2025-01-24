from pymmcore_plus import CMMCorePlus
from magicclass import magicclass, MagicTemplate
from magicgui import magicgui
from magicgui.tqdm import trange
from napari.qt.threading import thread_worker
#from superqt.utils import ensure_main_thread

from pathlib import Path
import numpy as np
import time
import zarr

from src.hardware.OPMNIDAQ import OPMNIDAQ
from src.hardware.PicardShutter import PicardShutter
from src.utils.autofocus_remote_unit import manage_O3_focus
from src.utils.data_io import write_metadata
from src.utils.image_post_processing import deskew
from datetime import datetime

# OPM control UI element            
@magicclass(labels=False)
class OPMMirrorScan(MagicTemplate):
    """MagicTemplate class to control OPM mirror scan."""

    # initialize
    def __init__(self):
        # OPM parameters
        self.active_channel = "Off"
        self.channel_powers = np.zeros(5,dtype=np.int8)
        self.channel_states=[False,False,False,False,False]
        self.exposure_ms = 10.0                 # unit: ms
        self.scan_axis_step_um = 0.4            # unit: um
        self.scan_axis_calibration = 0.043      # unit: V / um updated 2023.10.30
        self.galvo_neutral_volt = 0.            # unit: V
        self.scan_mirror_footprint_um = 50.0    # unit: um
        self.camera_pixel_size_um = .115        # unit: um
        self.opm_tilt = 30                      # unit: degrees

        # camera parameters
        self.camera_name = 'OrcaFusionBT'   # camera name in MM config
        self.ROI_center_x = int(1123)
        self.ROI_center_y = int(1172)-128   # the (-128) is to offset the area of best focus from known alignment point
        self.ROI_width_x = int(1900)        # unit: camera pixels
        self.ROI_width_y = int(512)         # unit: camera pixels
        self.ROI_corner_x = int(self.ROI_center_x -  self.ROI_width_x//2)
        self.ROI_corner_y = int(self.ROI_center_y -  self.ROI_width_y//2)

        # O3 piezo stage name
        self.O3_stage_name='MCL NanoDrive Z Stage'

        # shutter ID
        self.shutter_id = 712

        # default save path
        self.save_path = Path('D:/')

        self.channel_labels = ["405", "488", "561", "635", "730"]
        self.do_ind = [0, 1, 2, 3, 4]       # digital output line corresponding to each channel
        self.laser_blanking_value = True

        self.debug=False

        # flags for instrument setup
        self.powers_changed = True
        self.channels_changed = True
        self.ROI_changed = True
        self.exposure_changed = True
        self.footprint_changed = True
        self.scan_step_changed = True
        self.DAQ_running = False
        self.save_path_setup = False
        self.timelapse_setup = False

        # create mmcore singleton
        self.mmc = CMMCorePlus.instance()

        # set mmcore circular buffer
        self.mmc.clearCircularBuffer()
        circ_buffer_mb = 16000
        self.mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

    def _set_worker_2d(self,worker_2d):
        """Set 2D live-mode thread worker.
        
        Parameters
        ----------
        worker_2d: thread_worker
            Thread worker for 2D live-mode acquisition.
        """

        self.worker_2d = worker_2d
        self.worker_2d_started = False
        self.worker_2d_running = False
        
    def _set_worker_3d(self,worker_3d):
        """Set 3D live-mode thread worker.
        
        Parameters
        ----------
        worker_3d: thread_worker
            Thread worker for 3D live-mode acquisition.
        """

        self.worker_3d = worker_3d
        self.worker_3d_started = False
        self.worker_3d_running = False

    def _set_ao_worker_3d(self,ao_worker_3d):
        """Set 3D adaptive optics optimization thread worker.
        
        Parameters
        ----------
        ao_worker_3d: thread_worker
            Thread worker for 3D live-mode acquisition.
        """

        self.ao_worker_3d = ao_worker_3d
        self.ao_worker_3d_started = False
        self.ao_worker_3d_running = False

    def _create_3d_t_worker(self):
        """Create 3D timelapse acquistion thread worker.
        
        This thread is only created when user requests a timelapse acq.
        """

        worker_3d_t = self._acquire_3d_t_data()
        self._set_worker_3d_t(worker_3d_t)

    def _set_worker_3d_t(self,worker_3d_t):
        """Set 3D timelapse acquistion thread worker.
        
        Parameters
        ----------
        worker_3d_t: thread_worker
            Thread worker for 3D timelapse acquisition.
        """

        self.worker_3d_t = worker_3d_t
        self.worker_3d_t_running = False

    def _set_viewer(self,viewer):
        """Set napari viewer.
        
        Parameters
        ----------
        viewer: napari.viewer.Viewer
            The napari viewer instance.
        """
        self.viewer = viewer

    def _save_metadata(self):
        """Save metadata to CSV file."""

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

    def _update_layers(self,values):
        """Update napari viewer.
        
        Parameters
        ----------
        values: list
            Channels and images to update in the viewer.
        """

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
        """Live-mode: 2D acquisition without deskewing."""

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

            if self.channels_changed or self.scan_step_changed:
                if self.DAQ_running:
                    self.opmdaq.stop_waveform_playback()
                    self.DAQ_running = False
                    self.opmdaq.reset_scan_mirror()
                self.opmdaq.set_laser_blanking(self.laser_blanking_value)
                self.opmdaq.set_scan_type('2D')
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

    
    def _execute_3d_sweep(self):
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
        
        if self.channels_changed or self.footprint_changed or not(self.DAQ_running) or self.scan_step_changed:
            if self.DAQ_running:
                self.opmdaq.stop_waveform_playback()
                self.DAQ_running = False
            self.opmdaq.set_laser_blanking(self.laser_blanking_value)
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
        #----------------------------------------------------Start acquisition---------------------------------------------------------------
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

        return active_channel_indices, raw_image_stack

        #------------------------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------End acquisition----------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------


    @thread_worker
    def _acquire_3d_data(self):
        """Live-mode: 3D acquisition and deskewing."""

        while True:

            # execute sweep and return data
            active_channel_indices, raw_image_stack = self._execute_3d_sweep()

            # deskew parameters
            deskew_parameters = np.empty([3])
            deskew_parameters[0] = self.opm_tilt                 # (degrees)
            deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
            deskew_parameters[2] = self.camera_pixel_size_um*100 # (nm)

            for c in active_channel_indices:
                deskewed_image = deskew(raw_image_stack[c,:],*deskew_parameters).astype(np.uint16)  
                yield c, deskewed_image

            del raw_image_stack

    @thread_worker
    def _optimize_AO_3d(self):
        """Live-mode: optimize AO."""

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------Begin setup of AO opt parameters------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        """Setup AO parameters here."""

        #------------------------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------End setup of AO opt parameters-------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------


        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------Begin AO optimization-----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        """Setup loops for sensorless AO optimizaiton."""

        """perturb mirror for given mode and delta"""

        """acquire 3D data and max project"""
        # execute sweep and return data
        active_channel_indices, raw_image_stack = self._execute_3d_sweep()

        # deskew data
        # deskew parameters
        deskew_parameters = np.empty([3])
        deskew_parameters[0] = self.opm_tilt                 # (degrees)
        deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
        deskew_parameters[2] = self.camera_pixel_size_um*100 # (nm)

        max_z_deskewed_images = []
        for c in active_channel_indices:
            deskewed_image = deskew(raw_image_stack[c,:],*deskew_parameters).astype(np.uint16)
            max_z_deskewed_images.append(np.max(deskewed_image,axis=0))

            yield c, deskewed_image

        del raw_image_stack
        max_z_deskewed_images = np.asarray(max_z_deskewed_images)

        """STEVEN: The max_z_deskewed_images contains all of the channels requested. To start, I think you should only optimize on one channel, 
        so you can squeeze the channel dimension and go from there...
        
        `max_z_deskewed_image = np.squeeze(max_z_deskewed_images)`
        """
    
        """Calculate metric."""


        """After looping through all mirror pertubations for this mode, decide if mirror is updated"""

        """Loop back to top and do the next mode until all modes are done"""

        """Loop back to top and do the next iteration"""

        #------------------------------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------End AO optimization--------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

    @thread_worker
    def _acquire_3d_t_data(self):
        """Acquisition-mode: 3D + time acquisition to disk with no deskewing."""

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
        
        if self.channels_changed or self.footprint_changed or not(self.DAQ_running) or self.scan_step_changed:
            if self.DAQ_running:
                self.opmdaq.stop_waveform_playback()
                self.DAQ_running = False
            self.opmdaq.set_laser_blanking(self.laser_blanking_value)
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
        opm_data = zarr.open(str(zarr_output_path), mode="w", shape=(self.n_timepoints, self.n_active_channels, self.scan_steps, self.ROI_width_y, self.ROI_width_x), chunks=(1, self.n_active_channels, self.scan_steps, self.ROI_width_y, self.ROI_width_x),compressor=None, dtype=np.uint16)

        # construct metadata and save
        self._save_metadata()
        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------End setup of scan parameters----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------


        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------Start acquisition---------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # turn off Z motor
        exp_zstage_name = self.mmc.getFocusDevice()
        self.mmc.setProperty(exp_zstage_name,'MotorOnOff','Off')

        # set circular buffer to be large
        # self.mmc.clearCircularBuffer()
        # circ_buffer_mb = 16000
        # self.mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

        turned_off = False
        image_counter = 0
        # run hardware triggered acquisition
        if self.wait_time == 0:
            temp_data = np.zeros((self.n_active_channels, self.scan_steps,self.ROI_width_y, self.ROI_width_x),dtype=np.uint16)
            self.mmc.setExposure(self.exposure_ms)
            self.opmdaq.start_waveform_playback()
            self.DAQ_running = True
            self.mmc.startSequenceAcquisition(int(self.n_timepoints*self.n_active_channels*self.scan_steps),0,True)
            for t in trange(self.n_timepoints,desc="t", position=0):
                for z in trange(self.scan_steps,desc="z", position=1, leave=False):
                    for c in range(self.n_active_channels):
                        while self.mmc.getRemainingImageCount()==0:
                            pass
                        temp_data[c,z,:] = self.mmc.popNextImage()
                        image_counter +=1 
                        #print(self.mmc.getRemainingImageCount() + image_counter,int(self.n_timepoints*self.n_active_channels*self.scan_steps))
                        if (self.mmc.getRemainingImageCount() + image_counter) >= int(self.n_timepoints*self.n_active_channels*self.scan_steps) and not(turned_off):
                            self.opmdaq.stop_waveform_playback()
                            self.DAQ_running = False
                            turned_off = True
                opm_data[t, :]  = temp_data
            self.mmc.stopSequenceAcquisition()
            if not(turned_off):
                self.opmdaq.stop_waveform_playback()
                self.DAQ_running = False
        else:
            af_counter = 0
            
            self.current_O3_stage = manage_O3_focus(self.mmc,self.shutter_controller,self.O3_stage_name,verbose=True)
            self.mmc.setExposure(self.exposure_ms)
            temp_data = np.zeros((self.n_active_channels, self.scan_steps,self.ROI_width_y, self.ROI_width_x),dtype=np.uint16)
            for t in trange(self.n_timepoints,desc="t", position=0):
                self.mmc.setExposure(self.exposure_ms)
                self.opmdaq.start_waveform_playback()
                self.DAQ_running = True
                self.mmc.startSequenceAcquisition(int(self.n_timepoints*self.n_active_channels*self.scan_steps),0,True)
                for z in trange(self.scan_steps,desc="z", position=1, leave=False):
                    for c in range(self.n_active_channels):
                        while self.mmc.getRemainingImageCount()==0:
                            pass
                        temp_data[c,z,:] = self.mmc.popNextImage()
                        image_counter +=1 
                        if (self.mmc.getRemainingImageCount() + image_counter) >= int(self.n_timepoints*self.n_active_channels*self.scan_steps) and not(turned_off):
                            self.opmdaq.stop_waveform_playback()
                            self.DAQ_running = False
                            turned_off = True
                opm_data[t, :]  = temp_data
                self.mmc.stopSequenceAcquisition()
                if not(turned_off):
                    self.opmdaq.stop_waveform_playback()
                    self.DAQ_running = False
                if af_counter == 0:
                    t_start = time.perf_counter()
                    self.current_O3_stage = manage_O3_focus(self.mmc,self.shutter_controller,self.O3_stage_name,verbose=True)
                    self.mmc.setExposure(self.exposure_ms)
                    t_end = time.perf_counter()
                    t_elapsed = t_end - t_start
                    time.sleep(self.wait_time-t_elapsed*2)
                    self.current_O3_stage = manage_O3_focus(self.mmc,self.shutter_controller,self.O3_stage_name,verbose=True)
                    self.mmc.setExposure(self.exposure_ms)
                    af_counter = 0
                else:
                    time.sleep(self.wait_time)
                    af_counter = af_counter + 1

        # turn on Z motors
        exp_zstage_name = self.mmc.getFocusDevice()
        self.mmc.setProperty(exp_zstage_name,'MotorOnOff','On')

        #------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------End acquisition-------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        # set circular buffer to be small 
        #self.mmc.clearCircularBuffer()
        #circ_buffer_mb = 000
        #self.mmc.setCircularBufferMemoryFootprint(int(circ_buffer_mb))
        # self.channel_powers=[0,0,0,0,0]
        # self._set_mmc_laser_power()

    def _crop_camera(self):
        """Crop camera to GUI values."""

        current_ROI = self.mmc.getROI()
        if not(current_ROI[2]==2304) or not(current_ROI[3]==2304):
            self.mmc.clearROI()
            self.mmc.waitForDevice(self.camera_name)
        
        self.mmc.setROI(int(self.ROI_corner_x),
                        int(self.ROI_corner_y),
                        int(self.ROI_width_x),
                        int(self.ROI_width_y))
        self.mmc.waitForDevice(self.camera_name)

    def _lasers_to_hardware(self):
        """Change lasers to hardware control. """

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
        """Change lasers to software control."""

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
        """Change laser power."""
        
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 405-100C - PowerSetpoint (%)',float(self.channel_powers[0]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 488-150C - PowerSetpoint (%)',float(self.channel_powers[1]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser OBIS LS 561-150 - PowerSetpoint (%)',float(self.channel_powers[2]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 637-140C - PowerSetpoint (%)',float(self.channel_powers[3]))
        self.mmc.setProperty(r'Coherent-Scientific Remote',r'Laser 730-30C - PowerSetpoint (%)',float(self.channel_powers[4]))

    def _setup_camera(self):
        """Setup camera readout and triggering for OPM."""

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
        """Enforce camera being in trigger = INTERNAL mode."""

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
        """Startup OPM instrument in neutral state for all hardware."""

        # set lasers to 0% power and hardware control
        self._set_mmc_laser_power()
        self._lasers_to_hardware()

        # set camera to OPM specific setup
        self._crop_camera()
        self._setup_camera()
        self._enforce_DCAM_internal_trigger()

        # connect to DAQ
        self.opmdaq = OPMNIDAQ()
        # reset scan mirror position to neutral
        self.opmdaq.reset_scan_mirror()
        self.opmdaq.set_laser_blanking(self.laser_blanking)

        # connect to Picard shutter
        self.shutter_controller = PicardShutter(shutter_id=self.shutter_id,verbose=False)
        self.shutter_controller.closeShutter()
        self.shutter_state = 0

        self.mmc.setProperty(self.mmc.getFocusDevice(),'MotorOnOff','On')

    def _shutdown(self):
        """Shutdown OPM instrument in neutral state for all hardware."""
        
        # set lasers to 0% power and software control
        self._set_mmc_laser_power()
        self._lasers_to_software()

        # shutdown DAQ
        if self.DAQ_running:
            self.opmdaq.stop_waveform_playback()
        self.opmdaq.reset_scan_mirror()

        self.shutter_controller.shutDown()

    @magicgui(
        auto_call=False,
        exposure_ms={"widget_type": "FloatSpinBox", "min": 0.1, "max": 500,'label': 'Camera exposure (ms)'},
        layout='horizontal',
        call_button='Update exposure'
    )
    def set_exposure(self, exposure_ms=10.0):
        """Magicgui element to set camera exposure time.

        Parameters
        ----------
        exposure_ms: float
            camera exposure time
        """

        if not(exposure_ms == self.exposure_ms):
            self.exposure_ms=exposure_ms
            self.exposure_changed = True
        else:
            self.exposure_changed = False

    @magicgui(
        auto_call=False,
        width_x={"widget_type": "SpinBox", "min": 0, "max": 2304,'label': 'ROI width (non-tilt)'},
        width_y={"widget_type": "SpinBox", "min": 0, "max": 1024,'label': 'ROI height (tilt)'},
        layout='vertical', 
        call_button="Update crop"
    )
    def set_ROI(self, width_x=1900,width_y=512):
        """Magicgui element to set camera ROI.

        Parameters
        ----------
        uleft_corner_x: int
            upper left ROI x pixel
        uleft_corner_y: int
            upper left ROI y pixel
        width_x: int
            ROI width in pixels
        width_y: int
            ROI height in pixels = TILTED DIRECTION
        """
       
        if not(int(width_x)==self.ROI_width_x) or not(int(width_y)==self.ROI_width_y):
            self.ROI_corner_x=int(self.ROI_center_x-width_x//2)
            self.ROI_corner_y=int(self.ROI_center_y-width_y//2)
            self.ROI_width_x=int(width_x)
            self.ROI_width_y=int(width_y)
            self.ROI_changed = True
        else:
            self.ROI_changed = False

    @magicgui(
        auto_call=False,
        scan_step={"widget_type": "FloatSpinBox", "min": 0, "max": 2.0,'label': 'Scan step (um)'},
        layout='vertical', 
        call_button="Update scan step"
    )
    def set_scan_step(self, scan_step=0.4):
        """Magicgui element to set scan step

        Parameters
        ----------
        scan_step: float
            scan step size in microns
        """

        if not(scan_step == self.scan_axis_step_um):
            self.scan_axis_step_um = scan_step
            self.scan_step_changed = True
        else:
            self.scan_step_changed = False

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
        """Magicgui element to set relative laser powers (0-100%).

        Parameters
        ----------
        power_405: float
            405 nm laser power
        power_488: float
            488 nm laser power
        power_561: float
            561 nm laser power
        power_635: float
            635 nm laser power
        power_730: float
            730 nm laser power
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
        """Magicgui element to set active lasers.

        Parameters
        ----------
        active_channels: list
            list of booleans, one for each laser channel
        """

        states = [False,False,False,False,False]
        for channel in active_channels:
            if channel == 'Off':
                states[0]=True
            elif channel == '405':
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
        auto_call=True,
        laser_blanking={"widget_type": "CheckBox", "label": 'Laser blanking'},
        layout='horizontal'
    )
    def laser_blanking(self,laser_blanking = True):
        """Magicguic element to laser blanking state.
        
        Parameters
        ----------
        laser_blanking: bool, default True
            True = blanking, False = no blanking
        """

        if not(self.laser_blanking_value == laser_blanking):
            self.laser_blanking_value = laser_blanking
            self.opmdaq.set_laser_blanking(self.laser_blanking_value)

    @magicgui(
        auto_call=False,
        scan_mirror_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 250, "label": 'Mirror sweep (um)'},
        layout='horizontal',
        call_button='Update scan range'
    )
    def set_galvo_sweep(self, scan_mirror_footprint_um=25.0):
        """Magicgui element to set scan footprint.

        Parameters
        ----------
        scan_mirror_footprint_um: float, default 25.0
            size of the scan mirror footprint in microns
        """

        if not(scan_mirror_footprint_um==self.scan_mirror_footprint_um):
            self.scan_mirror_footprint_um=scan_mirror_footprint_um
            self.footprint_changed = True
        else:
            self.footprint_changed = False

    @magicgui(
        auto_call=True,
        live_mode_2D={"widget_type": "PushButton", "label": 'Start/Stop Live (2D)'},
        layout='horizontal'
    )
    def live_mode_2D(self,live_mode_2D=False):
        """Magicgui element to control live 2D imaging."""

        if (np.any(self.channel_states)):
            if not(self.worker_3d_running) and not(self.worker_3d_t_running):
                if self.worker_2d_running:
                    self.worker_2d.pause()
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running=False
                    self.worker_2d_running = False
                    # self.channel_powers=[0,0,0,0,0]
                    # self._set_mmc_laser_power()
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
    
    @magicgui(
        auto_call=True,
        live_mode_3D={"widget_type": "PushButton", "label": 'Start/Stop live (3D)'},
        layout='horizontal'
    )
    def live_mode_3D(self,live_mode_3D):
        """Magicgui element to start/stop live-mode 3D imaging.
        
        This function has to wait for an image to be yielded before it can stop the thread. 
        It can be slow to respond if you are taking a large galvo sweep with multiple colors.
        """

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


    @magicgui(
        auto_call=True,
        ao_opt_3D={"widget_type": "PushButton", "label": 'Start AO opt. (3D)'},
        layout='horizontal'
    )
    def ao_opt_3D(self,ao_opt_3D):
        """Magicgui element to start/stop AO optimization.
        
        This function has to wait for an image to be yielded before it can stop the thread. 
        It can be slow to respond if you are taking a large galvo sweep with multiple colors.
        """

        if (np.any(self.channel_states)):
            if not(self.worker_2d_running) and not (self.worker_3d_running) and not(self.worker_3d_t_running):
                self.galvo_scan = True
                if self.ao_worker_3d_running:
                    self.ao_worker_3d.pause()
                    self.ao_worker_3d_running = False
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running = False
                    self.opmdaq.reset_scan_mirror()
                else:
                    if not(self.ao_worker_3d_started):
                        self.ao_worker_3d.start()
                        self.ao_worker_3d_started = True
                        self.ao_worker_3d_running = True
                    else:
                        self.ao_worker_3d.resume()
                        self.ao_worker_3d_running = True
            else:
                if self.worker_2d_running:
                    raise Exception('Stop live 2D acquisition first.')
                elif self.worker_3d_running:
                    raise Exception('Stop live 3D acquisition first.')
                elif self.worker_3d_t_running:
                    raise Exception('Iterative acquisition in process.')
                else:
                    raise Exception('Unknown error.')
        else:
            raise Exception('Set at least one active channel before starting.')


    @magicgui(
        auto_call=True,
        n_timepoints={"widget_type": "SpinBox", "min": 0, "max": 10000, "label": 'Timepoints to acquire'},
        wait_time={"widget_type": "FloatSpinBox", "min": 0, "max": 720, "label": 'Delay between timepoints (s)'},
        layout='horizontal'
    )
    def set_timepoints(self, n_timepoints=400,wait_time=0):
        self.n_timepoints = n_timepoints
        self.wait_time = wait_time
        self.timelapse_setup = True
        """Set timelapse parameters.
        
        Parameters
        ----------
        n_timepoints: int, default 400
            number of timepoints to acquire
        wait_time: float, default 0
            time delay between timepoints in seconds. 0 is continuous imaging.
        """

    # set filepath for saving data
    @magicgui(
        auto_call=False,
        save_path={"widget_type": "FileEdit","mode": "d", "label": 'Save path:'},
        layout='horizontal', 
        call_button="Set"
    )
    def set_save_path(self, save_path=""):
        self.save_path = Path(save_path)
        self.save_path_setup = True
        """Magicgui element to set the filepath for saving data.
        
        Parameters
        ----------
        save_path: str, default ""
            path to save data
        """

    @magicgui(
        auto_call=True,
        timelapse_mode_3D={"widget_type": "PushButton", "label": 'Start acquistion'},
        layout='horizontal'
    )
    def timelapse_mode_3D(self,timelapse_mode_3D):
        """Magicui element to start 3D timelapse acquisition.
        
        
        This function currently cannot be stopped once started.
        """

        if not(self.worker_2d_running) and not(self.worker_3d_running):
            if (self.save_path_setup and self.timelapse_setup):
                self.worker_3d_t.start()
                self.worker_3d_t.returned.connect(self._create_3d_t_worker)
                self.worker_3d_t_running = True
            else:
                raise Exception('Setup save path and timelapse first.')
        else:
            raise Exception('Stop active live mode first.')

    @magicgui(
        auto_call=True,
        shutter_change={"widget_type": "PushButton", "label": 'Toggle alignment laser shutter.'},
        layout='horizontal'
    )
    def shutter_change(self,shutter_change):
        """Magicgui element to toggle the alignment laser shutter.
        
        Parameters
        ----------
        shutter_change: bool
            True = on, False = off
        """
            if self.shutter_state == 0:
                self.shutter_controller.openShutter()
                self.shutter_state = 1
            else:
                self.shutter_controller.closeShutter()
                self.shutter_state = 0
    @magicgui(
        auto_call=True,
        autofocus_O2O3={"widget_type": "PushButton", "label": 'Autofocus O2-O3'},
        layout='horizontal'
    )
    def autofocus_O2O3(self,autofocus_O2O3):
        """Magicgui element to autofocus the O2-O3 stage.
        
        Parameters
        ----------
        autofocus_O2O3: bool
            True = run autofocus, False = do not run autofocus
        """
        if not(self.worker_2d_running) and not(self.worker_3d_running) and not(self.worker_3d_t_running):
            if self.DAQ_running:
                self.opmdaq.stop_waveform_playback()
            self.current_O3_stage = manage_O3_focus(self.mmc,self.shutter_controller,self.O3_stage_name,verbose=True)
        else:
            raise Exception('Stop active live mode first.')