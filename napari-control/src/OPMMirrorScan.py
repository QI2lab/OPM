#!/usr/bin/python
"""MagicTemplate class to control OPM mirror scan.

Last updated: 2025.01.24 by Douglas Shepherd
"""

import h5py
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

from src.hardware.AOMirror import AOMirror
from src.hardware.OPMNIDAQ import OPMNIDAQ
from src.hardware.PicardShutter import PicardShutter
from src.utils.sensorless_ao import (metric_brightness,metric_gauss2d,metric_shannon_dct,quadratic_fit,
                                     save_optimization_results,plot_metric_progress,plot_zernike_coeffs)
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
        self.scan_axis_calibration = 0.0433      # unit: V / um updated 2025.01.24
        self.galvo_neutral_volt = 0.            # unit: V
        self.scan_mirror_footprint_um = 50.0    # unit: um
        self.camera_pixel_size_um = .115        # unit: um
        self.opm_tilt = 30                      # unit: degrees

        # camera parameters
        self.camera_name = 'OrcaFusionBT'   # camera name in MM config
        self.ROI_center_x = int(1178)
        self.ROI_center_y = int(1046)# -128   # the (-128) is to offset the area of best focus from known alignment point
        self.ROI_width_x = int(1900)        # unit: camera pixels
        self.ROI_width_y = int(512)         # unit: camera pixels
        self.ROI_corner_x = int(self.ROI_center_x -  self.ROI_width_x//2)
        self.ROI_corner_y = int(self.ROI_center_y -  self.ROI_width_y//2)

        # O3 piezo stage name. Needs to match the name in MM config.
        self.O3_stage_name='MCL NanoDrive Z Stage'

        # shutter ID. Obtained from Picard software.
        self.shutter_id = 712 # verified 2025.01.24

        # ao mirror setup
        wfc_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WaveFrontCorrector_mirao52-e_0329.dat")
        wfc_correction_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\correction_data_backup_starter.aoc")
        haso_config_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\Configuration Files\WFS_HASO4_VIS_7635.dat")
        wfc_flat_file_path = Path(r"C:\Users\qi2lab\Documents\github\opm_ao\OUT_FILES\20250122_tilted_gauss2d_laser_actuator_positions.wcs")

        # Adaptive optics parameters
        # ao_mirror puts the mirror in the flat_position state to start.
        self.ao_mirror = AOMirror(wfc_config_file_path = wfc_config_file_path,
                                  haso_config_file_path = haso_config_file_path,
                                  interaction_matrix_file_path = wfc_correction_file_path,
                                  flat_positions_file_path = wfc_flat_file_path,
                                  coeff_file_path = None,
                                  n_modes = 32,
                                  modes_to_ignore = [])
    
        # default save path
        self.save_path = Path('D:/')

        # Channel and laser setup
        # TODO: Generate list of laser names for changing properties, make sure it matches the channel_labels.
        self.channel_labels = ["405", "488", "561", "637", "730"]
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
        channel_names = ['405nm','488nm','561nm','637nm','730nm']
        colormaps = ['bop purple','bop blue','bop orange','red','grey']

        channel_name = channel_names[current_channel]
        colormap = colormaps[current_channel]
        try:
            self.viewer.layers[channel_name].data = new_image
        except Exception:
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
        initial_zern_modes = self.ao_mirror.current_coeffs.copy() # coeff before optmization
        iteration_zern_modes = initial_zern_modes.copy() # 
        active_zern_modes = initial_zern_modes.copy() # modified coeffs to be or are applied to mirror
        optimized_zern_modes = initial_zern_modes.copy() # final coeffs after running iterations
        metric_type = "shannon_dct"
        psf_radius_px = 1.5
        modes_to_optimize=[7,14,23,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31]
        n_iter=3
        n_steps=3
        init_range=.500
        alpha=.5
        compare_to_zero_delta = False
        verbose=True
        save_results = True
        display_images = True
        crop_size = None
        if save_results:
            optimal_metrics = []
            optimal_coefficients = []
            mode_images = []
            iteration_images = []
        
        #------------------------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------End setup of AO opt parameters-------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        #------------------------------------------------------------------------------------------------------------------------------------
        #----------------------------------------------------Begin AO optimization-----------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------------

        """Setup loops for sensorless AO optimizaiton."""
        delta_range=init_range
        if verbose:
            print(f"Starting A.O. optimization using {metric_type} metric")
        for k in range(n_iter): 
            # measure the starting metric for this iteration...
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

            del raw_image_stack
            max_z_deskewed_images = np.asarray(max_z_deskewed_images)
            max_z_deskewed_image = np.squeeze(max_z_deskewed_images)
            yield c, max_z_deskewed_image
            
            # Calculate the starting metric, future pertubations must improve from here.
            if metric_type=="brightness":
                starting_metric = metric_brightness(image=max_z_deskewed_image)
            elif metric_type=="gauss2d":
                starting_metric = metric_gauss2d(image=max_z_deskewed_image)
            elif metric_type=="shannon_dct":
                starting_metric = metric_shannon_dct(image=max_z_deskewed_image,
                                                     psf_radius_px=psf_radius_px,
                                                     crop_size=crop_size)
            
            # if k==0: 
            # SJS: We should be able to update the optimal metric at the start of every iteration. 
            #      The mirror modes should be the same since the end of the last iteration.
            #      Also updating at the start of every iteration prevent us from not progressing if an anomolous high metric occurs in the previous iteration.
            optimal_metric = starting_metric 
            if k==0:       
                if save_results:
                    iteration_images.append(max_z_deskewed_image)
                    
            for mode in modes_to_optimize:
                if verbose:
                    print(f"AO iteration: {k+1} / {n_iter}")
                    print(f"  Perturbing mirror mode: {mode+1} / {modes_to_optimize[-1]+1}")
                """perturb mirror for given mode and delta"""
                 # Grab the current starting modes for this iteration
                iteration_zern_modes = self.ao_mirror.current_coeffs.copy()
                deltas = np.linspace(-delta_range, delta_range, n_steps)
                metrics = []
                for delta in deltas:
                    active_zern_modes = iteration_zern_modes.copy()
                    active_zern_modes[mode] += delta
                    success = self.ao_mirror.set_modal_coefficients(active_zern_modes)
                    if not(success):
                        print("    Setting mirror coefficients failed!")
                        metric = 0
                        max_z_deskewed_image = np.zeros_like(max_z_deskewed_image)
                        
                        if display_images:
                            yield c, max_z_deskewed_image
                        if save_results:
                            mode_images.append(max_z_deskewed_image)
                            
                    else:
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

                        del raw_image_stack
                        max_z_deskewed_images = np.asarray(max_z_deskewed_images)
                        max_z_deskewed_image = np.squeeze(max_z_deskewed_images)
                        
                        if display_images:
                            yield c, max_z_deskewed_image
                        if save_results:
                            mode_images.append(max_z_deskewed_image)
                            
                        """Calculate metric."""
                        if metric_type=="brightness":
                            metric = metric_brightness(image=max_z_deskewed_image)
                        elif metric_type=="gauss2d":
                            metric = metric_gauss2d(image=max_z_deskewed_image)
                        elif metric_type=="shannon_dct":
                            metric = metric_shannon_dct(image=max_z_deskewed_image,
                                                        psf_radius_px=psf_radius_px,
                                                        crop_size=crop_size)
                            
                        if metric==np.nan:
                            print("Metric is NAN, setting to 0")
                            metric = float(np.nan_to_num(metric))
                        if verbose:
                            print(f"      Metric = {metric:.4f}")
                        
                    metrics.append(metric)
                
                """After looping through all mirror pertubations for this mode, decide if mirror is updated"""
                # Quadratic fit to determine optimal delta
                try:
                    popt = quadratic_fit(deltas, metrics)
                    a, b, c = popt
                    
                    # reduced the rejected amplitude of a.
                    is_increasing = all(x < y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    is_decreasing = all(x > y for x, y in zip(np.asarray(metrics), np.asarray(metrics)[1:]))
                    if is_increasing or is_decreasing:
                        print("      Test metrics are monotonic and linear, fit rejected. ")
                        raise Exception
                    elif a >=0:
                        print("      Test metrics have a positive curvature, fit rejected.")
                        raise Exception
                    
                    optimal_delta = -b / (2 * a)
                    if verbose:
                        print(f"    Quadratic fit result for optimal delta: {optimal_delta:.4f}")
    
                    if (optimal_delta>delta_range) or (optimal_delta<-delta_range):
                        print(f"      Optimal delta is outside of delta_range: {-b / (2 * a):.3f}")
                        raise Exception
                            
                except Exception:
                    optimal_delta = 0
                    if verbose:
                        print(f"        Exception in fit occurred, optimal delta = {optimal_delta:.4f}")
        
                coeff_opt = iteration_zern_modes[mode] + optimal_delta
                
                # test the new coeff to make sure it improves the overall metric.
                active_zern_modes[mode] = coeff_opt

                # verify mirror successfully loads requested state
                success = self.ao_mirror.set_modal_coefficients(active_zern_modes)
                if not(success):
                    if verbose:
                        print("    Setting mirror positions failed, using current mode coefficient.")
                    coeff_to_keep = iteration_zern_modes[mode]
                else:
                    # Measure the metric using the coeff to keep mirror state
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

                    del raw_image_stack
                    max_z_deskewed_images = np.asarray(max_z_deskewed_images)
                    max_z_deskewed_image = np.squeeze(max_z_deskewed_images)
                    
                    if display_images:
                        yield c, max_z_deskewed_image
                        
                    """Calculate metric."""
                    if metric_type=="brightness":
                        metric = metric_brightness(image=max_z_deskewed_image)
                    elif metric_type=="gauss2d":
                        metric = metric_gauss2d(image=max_z_deskewed_image)
                    elif metric_type=="shannon_dct":
                        metric = metric_shannon_dct(image=max_z_deskewed_image,
                                                    psf_radius_px=psf_radius_px,
                                                    crop_size=crop_size)
                        
                    if metric==np.nan:
                        print("    Metric is NAN, setting to 0")
                        metric = float(np.nan_to_num(metric))
                    
                    # When using the brightness metric, bleaching can cause the opt metric to be unattainable. 
                    # This flag changes the algorithm to compare it to the delta=0 measurement
                    if compare_to_zero_delta:
                        if metric>=metrics[len(metrics)//2]:
                            coeff_to_keep = coeff_opt
                            optimal_metric = metric
                            if verbose:
                                print(f"    Updating mirror with new optmimal mode coeff.: {coeff_to_keep:.4f} with metric: {metric:.4f}")
                        else:
                            # if not keep the current mode coeff
                            if verbose:
                                print("    Metric did not increase, using previous iteration's mode coeff.",
                                    f"\n      optimal metric: {optimal_metric:.6f}",
                                    f"\n      rejected metric: {metric:.6f}")
                            coeff_to_keep = iteration_zern_modes[mode]
                    else:
                        if metric>=optimal_metric:
                            coeff_to_keep = coeff_opt
                            optimal_metric = metric
                            if verbose:
                                print(f"      Updating mirror with new optmimal mode coeff.: {coeff_to_keep:.4f} with metric: {metric:.4f}")
                        else:
                            # if not keep the current mode coeff
                            if verbose:
                                print("    Metric not improved using previous iteration's mode",
                                    f"\n     optimal metric: {optimal_metric:.6f}",
                                    f"\n     rejected metric: {metric:.6f}")
                            coeff_to_keep = iteration_zern_modes[mode]
                
            
                if save_results:
                    optimal_metrics.append(optimal_metric)
                    
                # update mirror with the coeff to keep
                active_zern_modes[mode] = coeff_to_keep
                _ = self.ao_mirror.set_modal_coefficients(active_zern_modes)
                """Loop back to top and do the next mode until all modes are done"""
          
            # Update the iteration_zern_modes with final mirror state
            iteration_zern_modes = self.ao_mirror.current_coeffs.copy()
            if verbose:
                print(f"  Zernike modes at the end of iteration:\n{iteration_zern_modes}")
                
            # Reduce the sweep range for finer sampling around new optimal coefficient amplitude
            delta_range *= alpha
            if verbose:
                print(f"  Reduced sweep range to {delta_range:.4f}",
                      f"  Current metric: {metric:.4f}")
            
            if save_results: 
                optimal_coefficients.append(self.ao_mirror.current_coeffs.copy())
                iteration_images.append(max_z_deskewed_image)
                
            """Loop back to top and do the next iteration"""
                
        optimized_zern_modes = self.ao_mirror.current_coeffs.copy()          
        if verbose:
            print(f"Starting Zernike mode amplitude:\n{initial_zern_modes}",
                f"\nFinal optimized Zernike mode amplitude:\n{optimized_zern_modes}")
        
        # apply optimized Zernike mode coefficients to the mirror
        _ = self.ao_mirror.set_modal_coefficients(optimized_zern_modes)
        
        if self.save_path.exists():
            save_wfc_path = self.save_path / Path("wfc_optimized_mirror_positions.wcs")
            self.ao_mirror.save_mirror_state(save_wfc_path)
            
            # Create save paths for results and metrics / coeffs summary figures.
            results_save_path = self.save_path / Path("ao_optimization.h5")
            optimal_metrics_path = self.save_path / Path("ao_optimal_metrics.png")
            optimal_coeffs_path = self.save_path / Path("ao_coefficients.png")
            
            # Save ao results
            optimal_metrics = np.reshape(optimal_metrics, [n_iter, len(modes_to_optimize)])
            save_optimization_results(np.array(iteration_images), np.array(mode_images), optimal_coefficients, 
                                      optimal_metrics, modes_to_optimize, results_save_path)
            plot_metric_progress(optimal_metrics, modes_to_optimize, np.array(AOMirror.mode_names), optimal_metrics_path, False)
            plot_zernike_coeffs(optimal_coefficients, np.array(AOMirror.mode_names), optimal_coeffs_path, False)
            
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
        for ch_str in self.channel_labels:
            self.mmc.setConfig(f'Modulation-{ch_str}','External-Digital')
            self.mmc.waitForConfig(f'Modulation-{ch_str}','External-Digital')

        # turn all lasers on
        self.mmc.setConfig('Laser','AllOn')
        self.mmc.waitForConfig('Laser','AllOn')

    
    def _lasers_to_software(self):
        """Change lasers to software control."""

        # turn all lasers off
        self.mmc.setConfig('Laser','Off')
        self.mmc.waitForConfig('Laser','Off')

        # set all lasers back to software control
        for ch_str in self.channel_labels:
            self.mmc.setConfig(f'Modulation-{ch_str}','CW (constant power)')
            self.mmc.waitForConfig(f'Modulation-{ch_str}','CW (constant power)')


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
        for trig_idx in range(3):
            self.mmc.setProperty(self.camera_name,f'OUTPUT TRIGGER KIND[{trig_idx}]','EXPOSURE')
            self.mmc.setProperty(self.camera_name,f'OUTPUT TRIGGER POLARITY[{trig_idx}]','POSITIVE')
            
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[0]','EXPOSURE')
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[1]','EXPOSURE')
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER KIND[2]','EXPOSURE')
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[0]','POSITIVE')
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[1]','POSITIVE')
        # self.mmc.setProperty(self.camera_name,r'OUTPUT TRIGGER POLARITY[2]','POSITIVE')


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
        power_637={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '637nm power (%)'},
        power_730={"widget_type": "FloatSpinBox", "min": 0, "max": 100, "label": '730nm power (%)'},
        layout='vertical',
        call_button='Update powers'
    )
    def set_laser_power(self, power_405=0.0, power_488=0.0, power_561=0.0, power_637=0.0, power_730=0.0):
        """Magicgui element to set relative laser powers (0-100%).

        Parameters
        ----------
        power_405: float
            405 nm laser power
        power_488: float
            488 nm laser power
        power_561: float
            561 nm laser power
        power_637: float
            637 nm laser power
        power_730: float
            730 nm laser power
        """

        channel_powers = [power_405,power_488,power_561,power_637,power_730]

        if not(np.all(channel_powers == self.channel_powers)):
            self.channel_powers=channel_powers
            self.powers_changed = True
        else:
            self.powers_changed = False
        
    
    @magicgui(
        auto_call=True,
        active_channels = {"widget_type": "Select", "choices": ["Off","405","488","561","637","730"], "allow_multiple": True, "label": "Active channels"}
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
            elif channel == '637':
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