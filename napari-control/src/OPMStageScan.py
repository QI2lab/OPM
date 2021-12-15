# napari imports
from magicclass import magicclass, set_design
from magicgui import magicgui
from magicgui.tqdm import trange
from napari.qt.threading import thread_worker

#general python imports
from pathlib import Path
import numpy as np
from datetime import datetime
import time

# data i/o imports
import zarr
import npy2bdv

# hardware control imports
from pymmcore_plus import RemoteMMCore
from src.hardware.OPMNIDAQ import OPMNIDAQ
import src.hardware.ASI as ASIstage
from src.hardware.HamiltonMVP import HamiltonMVP
from src.hardware.APump import APump

# ASU OPM specific functions
from src.utils.fluidics_control import run_fluidic_program
from src.utils.data_io import write_metadata
from src.utils.image_post_processing import deskew
from src.OPMIterative import OPMIterative

# OPM control UI element            
@magicclass(labels=False)
@set_design(text="ASU Snouty-OPM control")
class OPMStageScan:

    # initialize
    def __init__(self, OPMIterative: OPMIterative):
        # OPM parameters
        self.active_channel = "Off"
        self.channel_powers = np.zeros(5,dtype=np.int8)
        self.channel_states=[False,False,False,False,False]
        self.exposure_ms = 10.0                 # unit: ms
        self.scan_axis_step_um = 0.4            # unit: um
        self.scan_axis_calibration = 0.043      # unit: V / um
        self.galvo_neutral_volt = -.15          # unit: V
        self.scan_mirror_footprint_um = 50.0      # unit: um
        self.camera_pixel_size_um = .115        # unit: um
        self.opm_tilt = 30                      # unit: degrees

        # camera parameters
        self.camera_name = 'OrcaFusionBT'   # camera name in MM config
        self.ROI_uleft_corner_x = int(200)  # unit: camera pixels
        self.ROI_uleft_corner_y = int(896)  # unit: camera pixels
        self.ROI_width_x = int(1900)        # unit: camera pixels
        self.ROI_width_y = int(512)         # unit: camera pixels

        # fluidics parameters
        self.path_to_fluidics_program = None
        self.iterative_mode = 0             # 0: flush fluidics, 1: single stage-scan w/o fluidics, 2: iterative stage-scan w/ fluidics
        self.pump_COM_port = 'COM5'
        self.valve_COM_port = 'COM6'
        self.pump_parameters = {'pump_com_port': self.pump_COM_port,
                                'pump_ID': 30,
                                'verbose': True,
                                'simulate_pump': False,
                                'serial_verbose': False,
                                'flip_flow_direction': False}

        # MM parameters
        self.x_stage_name = 'XYStage:XY:31'
        self.z_stage_name = 'ZStage:M:37'
        self.channel_labels = ["405", "488", "561", "635", "730"]
        self.do_ind = [0, 1, 2, 3, 4]       # digital output line corresponding to each channel

        # scan parameters
        self.save_path = Path('D:/')        
        self.excess_stage_steps = 20       # excess stage steps to allow stage to come up to speed

        # flags for instrument setup
        self.powers_changed = True
        self.channels_changed = True
        self.ROI_changed = True
        self.exposure_changed = True
        self.footprint_changed = True
        self.DAQ_running = False
        self.iterative_setup = False

        # debug flag
        self.debug=False

        self.OPMIterative = OPMIterative

    # set 2D acquistion thread worker
    def _set_worker_2d(self,worker_2d):
        """
        Set worker for continuous 2D imaging

        :param worker_2d: thread_worker
            Napari thread worker
        :return None:
        """

        self.worker_2d = worker_2d
        self.worker_2d_started = False
        self.worker_2d_running = False
        
    # set 3D acquistion thread worker
    def _set_worker_3d(self,worker_3d):
        """
        Set worker for continuous 3D imaging

        :param worker_3d: thread_worker
            Napari thread worker
        :return None:
        """

        self.worker_3d = worker_3d
        self.worker_3d_started = False
        self.worker_3d_running = False

    def _create_worker_iterative(self):
        """
        Create worker for iterative 3D imaging

        :return None:
        """

        worker_iterative = self._acquire_iterative_data()
        self._set_worker_iterative(worker_iterative)
        self.worker_iterative_running = False

    # set 3D timelapse acquistion thread worker
    def _set_worker_iterative(self,worker_iterative):
        """
        Set worker for iterative 3D imaging

        :param worker_3d_t: thread_worker
            Napari thread worker
        :return None:
        """
        self.worker_iterative = worker_iterative

    # set viewer
    def _set_viewer(self,viewer):
        """
        Set Napari viewer

        :param viewer: Viewer
            Napari viewer
        :return None:
        """

        self.viewer = viewer

    # update viewer layers
    def _update_layers(self,values):
        """
        Update Napari viewer layers

        :param values: tuple
            yielded data containing channel and image to update
        :return None:
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

    def _save_round_metadata(self,r_idx):
        """
        Construct round metadata dictionary and save

        :param r_idx: int
            round index
        :return None:
        """

        scan_param_data = [{'root_name': str("OPM_stage_data"),
                            'scan_type': 'stage',
                            'theta': float(self.opm_tilt), 
                            'exposure_ms': float(self.exposure_ms),
                            'pixel_size': float(self.camera_pixel_size_um),
                            'scan_axis_start': float(self.scan_axis_start_um),
                            'scan_axis_end': float(self.scan_axis_end_um),
                            'scan_axis_step': float(self.scan_axis_step_um), 
                            'tile_axis_start': float(self.tile_axis_start_um),
                            'tile_axis_end': float(self.tile_axis_end_um),
                            'tile_axis_step': float(self.tile_axis_step_um),
                            'height_axis_start': float(self.height_axis_start_um),
                            'height_axis_end': float(self.height_axis_end_um),
                            'height_axis_step': float(self.height_axis_step_um),
                            'r_idx': int(r_idx),
                            'num_r': int(self.n_iterative_rounds),
                            'num_y': int(self.n_xy_tiles), 
                            'num_z': int(self.n_z_tiles),
                            'num_ch': int(self.n_active_channels),
                            'scan_axis_positions': int(self.scan_steps),
                            'scan_axis_speed': float(self.scan_axis_speed),
                            'y_pixels': int(self.y_pixels),
                            'x_pixels': int(self.x_pixels),
                            '405_active': bool(self.channel_states[0]),
                            '488_active': bool(self.channel_states[1]),
                            '561_active': bool(self.channel_states[2]),
                            '635_active': bool(self.channel_states[3]),
                            '730_active': bool(self.channel_states[4]),
                            '405_power': float(self.channel_powers[0]),
                            '488_power': float(self.channel_powers[1]),
                            '561_power': float(self.channel_powers[2]),
                            '635_power': float(self.channel_powers[3]),
                            '730_power': float(self.channel_powers[4]),
                            }]
        
        write_metadata(scan_param_data[0], self.metadata_dir_path / Path('scan_'+str(r_idx)+'_metadata.csv'))

    def _save_stage_positions(self,r_idx,tile_idx,current_stage_data):
        """
        Construct stage position metadata dictionary and save

        :param r_idx: int
            round index
        :param tile_idx: int
            tile index
        :param current_stage_data: dict
            dictionary of stage positions
        :return None:
        """

        write_metadata(current_stage_data[0], self.metadata_dir_path / Path('stage_'+str(r_idx)+'_'+str(tile_idx)+'_metadata.csv'))


    def _save_full_metadata(self):
        """
        Save full metadata dictionaries

        :return None:
        """

        write_metadata(self.codebook[0], self.metadata_dir_path / Path('full_codebook_metadata.csv'))
        write_metadata(self.scan_settings[0], self.metadata_dir_path / Path('full_scan_settings_metadata.csv'))

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
                self.powers_changed = False

            with RemoteMMCore() as mmc_2d:

                if self.ROI_changed:

                    self._crop_camera()
                    self.ROI_changed = False

                # set exposure time
                if self.exposure_changed:
                    mmc_2d.setExposure(self.exposure_ms)
                    self.exposure_changed = False

                for c in active_channel_indices:
                    mmc_2d.snapImage()
                    raw_image_2d = mmc_2d.getImage()
                    time.sleep(.05)
                    yield c, raw_image_2d

    @thread_worker
    def _acquire_3d_data(self):
        while True:
            with RemoteMMCore() as mmc_3d:

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
                    mmc_3d.setExposure(self.exposure_ms)
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
                    self.opmdaq.start_waveform_playback()
                    self.DAQ_running = True
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

                # run hardware triggered acquisition
                mmc_3d.startSequenceAcquisition(int(n_active_channels*scan_steps),0,True)
                for z in range(scan_steps):
                    for c in active_channel_indices:
                        while mmc_3d.getRemainingImageCount()==0:
                            pass
                        raw_image_stack[c,z,:] = mmc_3d.popNextImage()
                mmc_3d.stopSequenceAcquisition()

                # deskew parameters
                deskew_parameters = np.empty([3])
                deskew_parameters[0] = self.opm_tilt                 # (degrees)
                deskew_parameters[1] = self.scan_axis_step_um*100    # (nm)
                deskew_parameters[2] = self.camera_pixel_size_um*100 # (nm)

                for c in active_channel_indices:
                    deskewed_image = deskew(np.flipud(raw_image_stack[c,:]),*deskew_parameters).astype(np.uint16)
                    time.sleep(.05)    
                    yield c, deskewed_image

                del raw_image_stack

                #------------------------------------------------------------------------------------------------------------------------------------
                #-----------------------------------------------End acquisition and deskew-----------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------

    @thread_worker
    def _acquire_iterative_data(self):

        # unwrap settings from iterative setup widget

        exposure_ms = self.scan_settings['exposure_ms']
        scan_axis_start_um = self.scan_settings['scan_axis_start_um']
        scan_axis_end_um =  self.scan_settings['scan_axis_end_um']
        scan_axis_step_um = self.scan_settings['scan_axis_step_um']
        tile_axis_start_um = self.scan_settings['tile_axis_start_um']
        tile_axis_end_um = self.scan_settings['tile_axis_end_um']
        tile_axis_step_um =self.scan_settings['tile_axis_step_um']
        height_axis_start_um = self.scan_settings['height_axis_start_um']
        height_axis_end_um = self.scan_settings['height_axis_end_um']
        height_axis_step_um = self.scan_settings['height_axis_step_um']
        n_iterative_rounds = self.scan_settings['n_iterative_rounds']
        nuclei_round = self.scan_settings['nuclei_round']
        num_xy_tiles = self.scan_settings['num_xy_tiles']
        num_z_tiles = self.scan_settings['num_z_tiles']
        n_active_channels_readout = self.scan_settings['n_active_channels_readout']
        n_active_channels_nuclei = self.scan_settings['n_active_channels_nuclei']
        scan_axis_positions = self.scan_settings['scan_axis_positions']
        scan_axis_speed_readout = self.scan_settings['scan_axis_speed_readout']
        scan_axis_speed_nuclei = self.scan_settings['scan_axis_speed_nuclei']
        y_pixels = self.scan_settings['y_pixels']
        x_pixels = self.scan_settings['x_pixels']
        channel_states_readout = [
            self.scan_settings['405_active_readout'],
            self.scan_settings['488_active_readout'],
            self.scan_settings['561_active_readout'],
            self.scan_settings['635_active_readout'],
            self.scan_settings['730_active_readout']
            ]
        channel_powers_readout = [
            self.scan_settings['405_power_readout'],
            self.scan_settings['488_power_readout'],
            self.scan_settings['561_power_readout'],
            self.scan_settings['635_power_readout'],
            self.scan_settings['730_power_readout']
            ]
        channel_states_nuclei = [
            self.scan_settings['405_active_nuclei'],
            self.scan_settings['488_active_nuclei'],
            self.scan_settings['561_active_nuclei'],
            self.scan_settings['635_active_nuclei'],
            self.scan_settings['730_active_nuclei']
            ]
        channel_powers_nuclei = [
            self.scan_settings['405_power_nuclei'],
            self.scan_settings['488_power_nuclei'],
            self.scan_settings['561_power_nuclei'],
            self.scan_settings['635_power_nuclei'],
            self.scan_settings['730_power_nuclei']
            ]

        # scan parameters that do not change between readout & nuclei rounds
        self.exposure_ms = exposure_ms
        self.pixel_size = self.camera_pixel_size_um
        self.scan_axis_start_um = scan_axis_start_um
        self.scan_axis_end_um = scan_axis_end_um
        self.scan_axis_step_um = scan_axis_step_um
        self.tile_axis_start_um = tile_axis_start_um
        self.tile_axis_end_um = tile_axis_end_um
        self.tile_axis_step_um = tile_axis_step_um
        self.height_axis_start_um = height_axis_start_um
        self.height_axis_end_um  = height_axis_end_um
        self.height_axis_step_um  = height_axis_step_um
        self.n_iterative_rounds = n_iterative_rounds
        self.n_xy_tiles = num_xy_tiles
        self.n_z_tiles = num_z_tiles
        self.y_pixels = y_pixels
        self.x_pixels = x_pixels
        self.scan_steps = scan_axis_positions
        
        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------
        
        # create directory for iterative imaging
        time_string = datetime.now().strftime("%Y_%m_%d-%I_%M_%S")
        output_dir_path = self.save_path / Path('iterative_'+time_string)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # create BDV H5 for registration of fiducial
        bdv_output_dir_path = output_dir_path / Path('fiducial_data')
        bdv_output_dir_path.mkdir(parents=True, exist_ok=True)
        bdv_output_path = bdv_output_dir_path / Path('fiducial_bdv.h5')
        bdv_writer = npy2bdv.BdvWriter(str(bdv_output_path), 
                                        nchannels=1, 
                                        ntiles=self.n_tiles, 
                                        subsamp=((1,1,1),),
                                        blockdim=((1, 128, 128),),
                                        compression='None')

        # create blank affine transformation to use for stage translation in BDV H5 XML
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                                (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                                (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

        # create metadata directory in output directory
        self.metadata_dir_path = output_dir_path / Path('metadata')
        self.metadata_dir_path.mkdir(parents=True, exist_ok=True)

        # create zarr data directory in output directory
        zarr_dir_path = output_dir_path / Path('raw_data')
        zarr_dir_path.mkdir(parents=True, exist_ok=True)

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------End setup of scan parameters----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------


        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------Start acquisition---------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------
        
        with RemoteMMCore() as mmc_iterative:

            # set circular buffer to be large
            mmc_iterative.clearCircularBuffer()
            circ_buffer_mb = 64000
            mmc_iterative.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

            ASIstage.set_1d_stage_scan(mmc_iterative)
            ASIstage.check_if_busy(mmc_iterative)
            ASIstage.set_1d_stage_scan_area(mmc_iterative,self.scan_axis_start_um/1000.,self.scan_axis_end_um/1000.)
            ASIstage.check_if_busy(mmc_iterative)
            ASIstage.setup_start_trigger_output(mmc_iterative)
            ASIstage.check_if_busy(mmc_iterative)

            for r_idx in trange(self.n_rounds,desc='round',position=0,leave=True):

                if r_idx==0:

                    # scan parameters that change between readout and nuclei rounds
                    self.n_active_channels = n_active_channels_readout
                    self.scan_steps = scan_axis_positions
                    self.scan_axis_speed = scan_axis_speed_readout
                    self.channel_states = channel_states_readout
                    self.channel_powers = channel_powers_readout

                    # setup instrument for readout rounds
                    self._crop_camera()
                    mmc_iterative.setExposure(self.exposure_ms)
                    self._set_mmc_laser_power()
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running = False
                        self.opmdaq.reset_scan_mirror()
                    self.opmdaq.set_scan_type('stage')
                    self.opmdaq.set_channels_to_use(self.channel_states)
                    self.opmdaq.set_interleave_mode(True)
                    self.opmdaq.generate_waveforms()

                elif r_idx == (nuclei_round - 1):

                    # scan parameters that change between readout and nuclei rounds
                    self.n_active_channels = n_active_channels_nuclei
                    self.scan_axis_speed = scan_axis_speed_nuclei
                    self.channel_states = channel_states_nuclei
                    self.channel_powers = channel_powers_nuclei

                    # setup instrument for nuclei round
                    self._set_mmc_laser_power()
                    if self.DAQ_running:
                        self.opmdaq.stop_waveform_playback()
                        self.DAQ_running = False
                        self.opmdaq.reset_scan_mirror()
                    self.opmdaq.set_scan_type('stage')
                    self.opmdaq.set_channels_to_use(self.channel_states)
                    self.opmdaq.set_interleave_mode(True)
                    self.opmdaq.generate_waveforms()

                ASIstage.set_axis_speed(mmc_iterative,'X',0.1)
                ASIstage.check_if_busy(mmc_iterative)
                ASIstage.set_axis_speed(mmc_iterative,'Y',0.1)
                ASIstage.check_if_busy(mmc_iterative)

                if (r_idx>0):
                    # run fluidics program for this round
                    success_fluidics = False
                    success_fluidics = run_fluidic_program(r_idx, self.df_program, self.valve_controller, self.pump_controller)
                else:
                    success_fluidics = True

                if (success_fluidics):
                    # create Zarr for this round
                    zarr_output_path = self.zarr_dir_path / Path('OPM_stage_data'+str(r_idx)+'.zarr')

                    # create and open zarr file
                    opm_round_data = zarr.open(
                        str(zarr_output_path), 
                        mode="w", 
                        shape=(self.n_xy_tiles, self.n_z_tiles, self.n_active_channels, self.stage_steps, self.ROI_width_y, self.ROI_width_x), 
                        chunks=(1, 1, 1, 1, self.ROI_width_y, self.ROI_width_x),
                        compressor=None,
                        dimension_separator="/",
                        dtype=np.uint16)

                    bdv_tile_idx = 0

                    for xy_idx in trange(self.n_xy_tiles,desc="xy tile",position=1,leave=False):
                        for z_idx in trange(self.n_z_tiles,desc="z tile",position=2,leave=False):
                    
                            # move stage
                            ASIstage.set_xy_position(mmc_iterative,stage_x,stage_y)
                            ASIstage.check_if_busy(mmc_iterative)
                            ASIstage.set_z_axis_mode(mmc_iterative,stage_z)
                            ASIstage.check_if_busy(mmc_iterative)

                            # grab actual stage position
                            stage_x, stage_y, stage_z = ASIstage.get_xyz_position(mmc_iterative)
                            ASIstage.check_if_busy(mmc_iterative)

                            # set scan stage speed
                            ASIstage.set_axis_speed(mmc_iterative,'X',self.scan_axis_speed)
                            ASIstage.check_if_busy(mmc_iterative)
        
                            # create current stage position
                            current_stage_data = [{'stage_x': float(stage_x), 
                                                'stage_y': float(stage_y), 
                                                'stage_z': float(stage_z)}]

                            # create affine xform for stage position
                            affine_matrix = unit_matrix
                            affine_matrix[1,3] = (stage_z)/(self.camera_pixel_size_um)  # x-translation 
                            affine_matrix[0,3] = (stage_y)/(self.camera_pixel_size_um)  # y-translation
                            affine_matrix[2,3] = (stage_x)/(self.camera_pixel_size_um)  # z-translation

                            # Create virtual stack within BDV H5 to place fused z planes into
                            bdv_writer.append_view(stack=None,  
                                                virtual_stack_dim=(self.stage_steps,self.ROI_width_y,self.ROI_width_x),
                                                time=r_idx, 
                                                channel=0, 
                                                tile=bdv_tile_idx,
                                                voxel_size_xyz=(self.camera_pixel_size_um, self.camera_pixel_size_um, self.scan_axis_step_um),
                                                voxel_units='um',
                                                calibration = (1,1,np.abs(self.stage_step_size_um/self.camera_pixel_size_um)),
                                                m_affine=affine_matrix,
                                                name_affine = 'tile '+str(bdv_tile_idx)+' translation')

                            # enforce camera in START trigger mode
                            self._enforce_DCAM_external_trigger()

                            # start DAQ
                            self.opmdaq.start_waveform_playback()

                            # run hardware triggered acquisition
                            mmc_iterative.startSequenceAcquisition(int(self.n_active_channels*self.stage_steps),0,True)
                            ASIstage.start_1d_stage_scan(mmc_iterative)
                            # collect and discard excess images while stage is coming up to speed
                            for excess_idx in range(self.excess_stage_steps):
                                for c_idx in range(self.n_active_channels):
                                    excess_image = mmc_iterative.popNextImage()

                            # collect interleaved data
                            # place all data into Zarr
                            # also place fidicual images into BDV for stitching
                            for s_idx in trange(self.stage_steps,desc="scan", position=3, leave=False):
                                for c_idx in range(self.n_active_channels):
                                    while mmc_iterative.getRemainingImageCount()==0:
                                        pass
                                    image = mmc_iterative.popNextImage()
                                    opm_round_data[xy_idx, z_idx, c_idx, s_idx, :, :] =image
                                    if (c_idx==0 and r_idx<(self.n_rounds-1)) or (c_idx==1 and r_idx==(self.n_rounds-1)):
                                        #write the fiducial channel into bigstitcher
                                        bdv_writer.append_plane(
                                            plane=image, 
                                            z=s_idx, 
                                            time=r_idx, 
                                            channel=0)
                            mmc_iterative.stopSequenceAcquisition()

                            # stop DAQ
                            self.opmdaq.stop_waveform_playback()

                            bdv_tile_idx = bdv_tile_idx + 1
                            
                            self._save_stage_positions(r_idx,xy_idx,z_idx,current_stage_data)

                    # del reference to Zarr file
                    opm_round_data = None

                    # construct round metadata and save
                    self._save_round_metadata(r_idx)

            #------------------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------End acquisition-------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------------------------

            if (success_fluidics):
                # clean up DAQ
                self.opmdaq.reset_scan_mirror()

                # set circular buffer to be small 
                mmc_iterative.clearCircularBuffer()
                circ_buffer_mb = 4000
                mmc_iterative.setCircularBufferMemoryFootprint(int(circ_buffer_mb))

                # create down-sampled views with compression
                bdv_writer.create_pyramids(subsamp=((4, 8, 8)),
                                blockdim=((8, 128, 128)),
                                compression='lzf')

                # write and close BDV H5 xml file
                bdv_writer.write_xml()
                bdv_writer.close()

                # write full metadata
                self._save_full_metadata()
            else:
                # write and close BDV H5 xml file
                bdv_writer.write_xml()
                bdv_writer.close()
                raise Exception('Error in fluidics. Acquisition failed.')

    def _crop_camera(self):
        """
        Crop camera to GUI values

        :return None:
        """

        with RemoteMMCore() as mmc_crop_camera:
            current_ROI = mmc_crop_camera.getROI()
            if not(current_ROI[2]==2304) or not(current_ROI[3]==2304):
                mmc_crop_camera.clearROI()
                mmc_crop_camera.waitForDevice(self.camera_name)
            mmc_crop_camera.setROI(int(self.ROI_uleft_corner_x),int(self.ROI_uleft_corner_y),int(self.ROI_width_x),int(self.ROI_width_y))
            mmc_crop_camera.waitForDevice(self.camera_name)

    def _lasers_to_hardware(self):
        """
        Change lasers to hardware control

        :return None:
        """

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

    def _lasers_to_software(self):
        """
        Change lasers to software control

        :return None:
        """

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

    def _set_mmc_laser_power(self):
        """
        Change laser power

        :return None:
        """
        with RemoteMMCore() as mmc_laser_power:
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 405-100C - PowerSetpoint (%)',float(self.channel_powers[0]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 488-150C - PowerSetpoint (%)',float(self.channel_powers[1]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser OBIS LS 561-150 - PowerSetpoint (%)',float(self.channel_powers[2]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 637-140C - PowerSetpoint (%)',float(self.channel_powers[3]))
            mmc_laser_power.setProperty(r'Coherent-Scientific Remote',r'Laser 730-30C - PowerSetpoint (%)',float(self.channel_powers[4]))

    def _setup_camera(self):
        """
        Setup camera readout and triggering for OPM

        :return None:
        """

        with RemoteMMCore() as mmc_camera_setup:

            # give camera time to change modes if necessary
            mmc_camera_setup.setConfig('Camera-Setup','ScanMode3')
            mmc_camera_setup.waitForConfig('Camera-Setup','ScanMode3')

            # set camera to internal trigger
            mmc_camera_setup.setConfig('Camera-TriggerSource','INTERNAL')
            mmc_camera_setup.waitForConfig('Camera-TriggerSource','INTERNAL')
            
            # set camera to internal trigger
            # give camera time to change modes if necessary
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[0]','EXPOSURE')
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[1]','EXPOSURE')
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[2]','EXPOSURE')
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[0]','POSITIVE')
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[1]','POSITIVE')
            mmc_camera_setup.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[2]','POSITIVE')

    def _enforce_DCAM_external_trigger(self):
        """
        Enforce camera being in external trigger = START mode

        :return None:
        """

        with RemoteMMCore() as mmc_camera_trigger:

            # set camera to START mode upon input trigger
            mmc_camera_trigger.setConfig('Camera-TriggerType','START')
            mmc_camera_trigger.waitForConfig('Camera-TriggerType','START')

            # check if camera actually changed
            # we find that camera doesn't always go back to START mode and need to check it
            
    def _startup(self):
        """
        Startup OPM instrument in neutral state for all hardware

        :return None:
        """

        # set lasers to 0% power and hardware control
        self._set_mmc_laser_power()
        self._lasers_to_hardware()

        # set camera to OPM specific setup
        self._setup_camera()
        #self._enforce_DCAM_external_trigger()

        '''
        # connect to pump
        self.pump_controller = APump(self.pump_parameters)
        # set pump to remote control
        self.pump_controller.enableRemoteControl(True)

        # connect to valves
        self.valve_controller = HamiltonMVP(com_port=self.valve_COM_port)
        # initialize valves
        self.valve_controller.autoAddress()
        '''

        # connect to DAQ
        self.opmdaq = OPMNIDAQ()
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
        scan_mirror_footprint_um={"widget_type": "FloatSpinBox", "min": 5, "max": 200, "label": 'Mirror sweep (um)'},
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
            if not(self.worker_3d_running) and not(self.worker_iterative_running):
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
            if not(self.worker_2d_running) and not(self.worker_iterative_running):
                self.galvo_scan = True
                if self.worker_3d_running:
                    self.worker_3d.pause()
                    self.worker_3d_running = False
                    self.opmdaq.stop_waveform_playback()
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

    def _set_iterative_configuration(self,values):
        if len(values) > 0:
            self.codebook = values[0]
            self.df_fluidics = values[1]
            self.scan_settings = values[2]
            self.valve_controller = values[3]
            self.pump_controller = values[4]
            self.iterative_setup = True

    # control stage scan (hardware triggering)
    @magicgui(
        auto_call=True,
        stagescan_mode_3D={"widget_type": "PushButton", "label": 'Start iterative scan'},
        layout='horizontal'
    )
    def stagescan_mode_3D(self,stagescan_mode_3D):
        if not(self.worker_2d_running) and not(self.worker_3d_running):
            if (self.iterative_setup and self.save_path_setup):
                self.galvo_scan = False
                self.worker_iterative.start()
                self.worker_iterative_running = True
                self.worker_iterative.returned.connect(self._create_worker_iterative)
            else:
                raise Exception('Set configuration and save path first.')
        else:
            raise Exception('Stop active live mode first.')