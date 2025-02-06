#!/usr/bin/python
"""
Instrument interface class to run NIDAQ with camera as master for OPM using PyDAQMx 

Authors:
Peter Brown
Franky Djutanta
Steven Sheppard
Douglas Shepherd

Change Log:
2025/02/25: Projection mode
2021/12/11: Initial version
----------------------------------------------------------------------------------------
"""

"""
Example projection code
nidaq.set_channels_to_use([True, False, False, False, False])
nidaq.exposure = 0.10
scan_mirror_sweep_um = 40
proj_mirror_sweep_um = scan_mirror_sweep_um  #+ (30 * np.cos(30 * np.pi / 180))
proj_mirror_calibration =  .0052
nidaq.set_scan_mirror_range(0.4, scan_mirror_sweep_um)
voltage = proj_mirror_sweep_um * proj_mirror_calibration 
nidaq.proj_mirror_min_volt = -voltage/2
nidaq.proj_mirror_max_volt = voltage/2
nidaq.generate_waveforms()
nidaq.prepare_waveform_playback()
nidaq.start_waveform_playback()
"""

import PyDAQmx as daq
import ctypes as ct
import numpy as np
from typing import Sequence, List

class OPMNIDAQ:
    """Class to control NIDAQ."""

    def __init__(self,verbose: bool=False):
        # Define acquisition parameters.
        self.scan_type = 'mirror'
        self.do_ind = [0,1,2,3,4]
        self.active_channels_indices = None
        self.n_active_channels = 0
        self.exposure = .05

        # Define waveform generation parameters for projection mode
        self.daq_sample_rate_hz = 10000
        self.num_do_channels = len(self.do_ind)
        self.do_waveform = [False] * len(self.do_ind)
        self.ao_waveform = [np.zeros(1), np.zeros(1)]
                
        # Configure hardware pin addresses.
        self.dev_name = "Dev1"
        self.channel_addresses = {"do_channels":["/Dev1/port0/line0", # 405
                                                 "/Dev1/port0/line1", # 473
                                                 "/Dev1/port0/line2", # 532
                                                 "/Dev1/port0/line3", # 561
                                                 "/Dev1/port0/line4"],# 638
                                  "ao_mirrors":["/Dev1/ao0",  # image scanning galvo
                                                "/Dev1/ao1"], # projection scanning galvo
                                  "di_camera_trigger":"/Dev1/PFI0",
                                  "di_start_trigger":"/Dev1/PFI1",
                                  "di_change_trigger":"/Dev1/PFI2",
                                  "di_start_ao_trigger":"/Dev1/PFI3"}
        
        self.address_channel_do = ["/Dev1/port0/line0", # 405
                                   "/Dev1/port0/line1", # 473
                                   "/Dev1/port0/line2", # 532
                                   "/Dev1/port0/line3", # 561
                                   "/Dev1/port0/line4"] # 638
        self.address_ao_mirrors = ["/Dev1/ao0", # image scanning galvo
                                   "/Dev1/ao1"] # projection scanning galvo
        self.channel_di_trigger_from_camera = "/Dev1/PFI0" # camera trig port 0
        self.channel_di_start_trigger = "/Dev1/PFI1" # Empty PFI pin
        self.channel_di_change_trigger = "/Dev1/PFI2" # Empty PFI pin
        self.channel_ao_start_trigger = "/Dev1/PFI3" # Route channel_do_trigger
        
        # Define image scanning galvo mirror parameters.
        self.ao_neutral_positions = [0.0, 0.0]
        self.scan_mirror_calibration = 0.043
        
        # Define projection galvo mirror parameters.
        # TODO: covert from pixel to voltage using calibration, grab ROI values.
        self.proj_mirror_min_volt = 0.0
        self.proj_mirror_max_volt = 0.0
        self.proj_mirror_calibration =  .0052

        # Define laser blanking option
        self.laser_blanking=True
        
        # task handles
        self._task_do = None
        self._task_ao = None
        self._task_di = None
        
        
        
    def reset(self):
        """Reset the device."""

        daq.DAQmxResetDevice(self.dev_name)
        self.reset_ao_channels()
        self.reset_do_channels()
   
   
    def set_acquisition_params(self,
                               scan_type: str = None,
                               channel_states: List[bool] = None,
                               image_scan_step_size_um: float = None,
                               image_scan_sweep_um: float = None,
                               laser_blanking: bool = None,
                               exposure: float = None,
                               proj_mirror_min_volt: float = None,
                               proj_mirror_max_volt: float = None,
                               proj_mirror_calibration: float = None):
        """
        """
        if scan_type:
            self.scan_type=scan_type
        if channel_states:
            self.active_channel_indices = [ind for ind, st in zip(self.do_ind, channel_states) if st]
            self.n_active_channels = len(self.active_channel_indices)
        if image_scan_step_size_um and image_scan_sweep_um:
            # determine sweep footprint
            self.scan_mirror_min_volt = -(image_scan_step_size_um * self.scan_mirror_calibration / 2.) + self.ao_neutral_positions[0] # unit: volts
            self.scan_axis_step_volts = image_scan_step_size_um * self.scan_mirror_calibration # unit: V
            self.scan_axis_range_volts = image_scan_sweep_um * self.scan_mirror_calibration # unit: V
            self.image_scan_steps = np.rint(self.scan_axis_range_volts / self.scan_axis_step_volts).astype(np.int16) # galvo steps
            return self.image_scan_steps
        if laser_blanking:
            self.laser_blanking = laser_blanking
        if exposure:
            self.exposure = exposure
        if proj_mirror_min_volt:
            self.proj_mirror_min_volt = proj_mirror_min_volt
        if proj_mirror_max_volt:
            self.proj_mirror_max_volt = proj_mirror_max_volt
        if proj_mirror_calibration:
            self.proj_mirror_calibration = proj_mirror_calibration
        
        
    def set_scan_type(self,scan_type: str):
        self.scan_type = scan_type
        

    def set_channels_to_use(self,channel_states: Sequence):
        """Set the active channels to use for acquisition.
        
        Parameters
        ----------
        channel_states : Sequence
            A list of boolean values indicating which channels to use.
        """
        self.active_channels_indices = [ind for ind, st in zip(self.do_ind, channel_states) if st]
        self.n_active_channels = len(self.active_channels_indices)
        
    def set_scan_mirror_range(self,scan_mirror_step_size_um: float, scan_mirror_sweep_um: float):
        """Set the range of the scanning mirror in microns.
        
        Parameters
        ----------
        scan_mirror_step_size_um : float
            The step size of the scanning mirror in microns.
        scan_mirror_sweep_um : float
            The range of the scanning mirror in microns.
        
        Returns
        -------
        image_scan_steps : int
            The number of steps in the scan mirror sweep.
        """

        # determine scan mirror voltage range
        self.scan_mirror_min_volt = -(scan_mirror_sweep_um * self.scan_mirror_calibration / 2.) + self.scan_mirror_neutral # unit: volts
        self.scan_axis_step_volts = scan_mirror_step_size_um * self.scan_mirror_calibration # unit: V
        self.scan_axis_range_volts = scan_mirror_sweep_um * self.scan_mirror_calibration # unit: V
        self.image_scan_steps = np.rint(self.scan_axis_range_volts / self.scan_axis_step_volts).astype(np.int16) # galvo steps
        return self.image_scan_steps

    def set_proj_mirror_range(self,proj_mirror_sweep_um: float):
        """Set the range of the projection mirror in microns.
        
        Parameters
        ----------
        proj_mirror_sweep_um : float
            The range of the projection mirror in microns.
        """

        # determine projection mirror voltage range
        voltage = proj_mirror_sweep_um * self.proj_mirror_calibration
        self.proj_mirror_min_volt = -voltage/2
        self.proj_mirror_max_volt = voltage/2

    def reset_ao_channels(self):
        """Stops any waveforms and deletes tasks, set analog lines to the mirror's neutral positions
        """
        self.clear_tasks()
        
        ao_waveform = np.column_stack((np.full(2, self.ao_neutral_positions[0]),
                                       np.full(2, self.self.ao_neutral_positions[1])))

        samples_per_ch_ct = ct.c_int32()
        with daq.Task("ResetAO") as _task:
            _task.CreateAOVoltageChan(self.address_ao_mirrors[0], 
                                      "reset_ao0", 
                                      -6.0, 6.0, daq.DAQmx_Val_Volts, None)
            _task.CreateAOVoltageChan(self.address_ao_mirrors[1], 
                                      "reset_ao1",
                                      -1.0, 1.0, daq.DAQmx_Val_Volts, None)
            _task.WriteAnalogF64(1, 
                                 True, 
                                 1, 
                                 daq.DAQmx_Val_GroupByScanNumber, 
                                 ao_waveform, 
                                 ct.byref(samples_per_ch_ct), None)
            _task.StopTask()
            _task.ClearTask()
      
       
    def reset_do_channels(self):
        """Stops any running waveforms and deletes task handlers. Then sets DO lines to 0.
        """
        self.clear_tasks()
        
        with daq.Task("ResetDO") as _task:
            _task.CreateDOChan(", ".join(self.address_channel_do), 
                               "reset_do", 
                               daq.DAQmx_Val_ChanForAllLines)
            _task.WriteDigitalLines(1, True, 1.0, daq.DAQmx_Val_GroupByChannel, 
                                    np.zeros((1, len(self.address_channel_do)), dtype=np.uint8),
                                    None, None)    
            _task.StopTask()
            _task.ClearTask()
            
            
    def generate_waveforms(self):
        """Generate waveforms necessary to capture 1 'volume'.
           - Waveforms run after receiving change detection from camera trigger.
           - 'volume' for a 2d scan is a single image mirror position.
           - 'volume' for a projection scan is a linear ramp for both the image and projection mirrors.
           - 'volume' for a mirror scan is n_scan_step frames x n_do channels x camera_roi
        """        

        if self.scan_type == 'mirror':
            """Fire active lasers, advance image scanning galvo in a linear ramp,
               hold the projection galvo in it's neutral position
            """
            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            # Collect one frame for each scan position
            n_voltage_steps = self.image_scan_steps
            self.samples_per_do_ch = 2*n_voltage_steps*self.n_active_channels
            
            # Generate values for DO
            do_waveform = np.zeros((self.samples_per_do_ch, self.num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self.active_channels_indices):
                # Turn laser on in order for each image position
                if self.laser_blanking:
                    do_waveform[2*ii::2*self.n_active_channels, ind] = 1
                else:
                    do_waveform[:,int(ind)] = 1
            
            if self.laser_blanking:
                do_waveform[-1, :] = 0
                
            #-----------------------------------------------------#
            # Create ao waveform, scan the image mirror voltage, keep the projection mirror neutral            
            # This array is written for both AO channels
            ao_waveform = np.zeros((self.samples_per_do_ch, 2))
            
            # Generate image scanning mirror voltage steps
            max_volt = self.scan_mirror_min_volt + self.scan_axis_range_volts
            scan_mirror_volts = np.linspace(self.scan_mirror_min_volt, max_volt, n_voltage_steps)
            
            # Set the last time point (when exp is off) to the first mirror positions.
            ao_waveform[0:2*self.n_active_channels - 1, 0] = scan_mirror_volts[0]

            if len(scan_mirror_volts) > 1:
                # (2 * # active channels) voltage values for all other frames
                ao_waveform[2*self.n_active_channels - 1:-1, 0] = np.kron(scan_mirror_volts[1:], np.ones(2 * self.n_active_channels))
            
            # set back to initial value at end
            ao_waveform[-1] = scan_mirror_volts[0]
        
        elif self.scan_type == "projection":
            """Perform projection scan.
            
               - Fire active lasers
               - Synchronize image scanning mirror and projection mirror to rolling shutter.
               - Capture one frame per active channel.
               - Start camera in light sheet mode, apply linear ramp to each mirror
            """
            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            self.samples_per_do_ch = (2*self.n_active_channels - 1) + 1
 
            # Generate values for DO
            do_waveform = np.zeros((self.samples_per_do_ch, self.num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self.active_channels_indices):
                if self.laser_blanking:
                    do_waveform[2*ii::2*self.n_active_channels, ind] = 1
                else:
                    do_waveform[:,int(ind)] = 1
            
            if self.laser_blanking:
                do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, scan the image mirror and projection mirror voltages.
            # This array is written for both AO channels and runs at the camera di rising edge
            n_voltage_steps = int(self.exposure * self.daq_sample_rate_hz)
            return_samples = 5
            self.samples_per_ao_ch = n_voltage_steps + return_samples
            ao_waveform = np.zeros((self.samples_per_ao_ch, 2))
            
            # Generate projection mirror linear ramp
            # TODO: Set using edges of the ROI, and calibration volts per px
            if self.verbose:
                print(self.proj_mirror_min_volt)
                print(self.proj_mirror_max_volt)
            proj_mirror_volts = np.linspace(self.proj_mirror_min_volt, self.proj_mirror_max_volt, n_voltage_steps)
            
            # Generate image scanning mirror voltage steps
            scan_mirror_max_volts = self.scan_mirror_min_volt + self.scan_axis_range_volts
            scan_mirror_volts = np.linspace(self.scan_mirror_min_volt, scan_mirror_max_volts, n_voltage_steps)
            
            # Set the last time point (when exp is off) to the first mirror positions.
            ao_waveform[:-1, 0] = scan_mirror_volts
            ao_waveform[:-1, 1] = proj_mirror_volts

            # set back to initial value at end
            ao_waveform[-return_samples:, 0] = scan_mirror_volts[0]
            ao_waveform[-return_samples:, 1] = proj_mirror_volts[0]
            
        elif self.scan_type == 'stage':
            """Only fire the active channel lasers,keep the mirrors in their neutral positions"""

            #-----------------------------------------------------#
            # setup digital trigger buffer on DAQ
            self.samples_per_do_ch = 2 * int(self.n_active_channels)

            # create DAQ pattern for laser strobing controlled via rolling shutter
            do_waveform = np.zeros((self.samples_per_do_ch, self.num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self.active_channels_indices):
                if self.laser_blanking:
                    do_waveform[2*ii::2*int(self.n_active_channels), int(ind)] = 1
                else:
                    do_waveform[:,int(ind)] = 1

            if self.laser_blanking:
                do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, keeping the mirrors in their neutral positions
            # In stage scan mode, only the first time point gets set.
            ao_waveform = np.zeros((1, 2))
            ao_waveform[:, 0] = self.ao_neutral_positions[0]
            ao_waveform[:, 1] = self.ao_neutral_positions[1]
            
        elif self.scan_type == '2D':
            """Only fire the active channel lasers, keep the mirrors in their neutral positions
            """
            #-----------------------------------------------------#
            # The DO channel changes with changes in camera's trigger output,
            # There are 2 time steps per frame, except for first frame plus one final frame to reset voltage
            self.samples_per_do_ch = (2*self.n_active_channels - 1) + 1
 
            # Generate values for DO
            do_waveform = np.zeros((self.samples_per_do_ch, self.num_do_channels), dtype=np.uint8)
            for ii, ind in enumerate(self.active_channels_indices):
                if self.laser_blanking:
                    do_waveform[2*ii::2*self.n_active_channels, ind] = 1
                else:
                    do_waveform[:,int(ind)] = 1
            
            if self.laser_blanking:
                do_waveform[-1, :] = 0
            
            #-----------------------------------------------------#
            # Create ao waveform, keeping the mirrors in their neutral positions
            # In 2D mode, the first time point gets set.
            ao_waveform = np.zeros((1, 2))
            ao_waveform[:, 0] = self.ao_neutral_positions[0]
            ao_waveform[:, 1] = self.ao_neutral_positions[1]
            
        # Update daq waveforms
        self.do_waveform = do_waveform
        self.ao_waveform = ao_waveform
            
    def prepare_waveform_playback(self):
        """Create DAQ tasks for synchronizing camera output triggers to lasers and galvo mirrors."""
        self.stop_waveform_playback()
        try:
            #-------------------------------------------------#
            # Create DI trigger from camera task
            # This should only be done at startup or after a reset occurs
            if self._task_di == None:
                self._task_di = daq.Task("TaskDI")
                self._task_di.CreateDIChan(self.channel_di_trigger_from_camera,
                                        "DI_CameraTrigger", 
                                        daq.DAQmx_Val_ChanForAllLines)
                
                # Configure change detection timing (from wave generator)
                self._task_di.CfgInputBuffer(0)    # must be enforced for change-detection timing, i.e no buffer
                self._task_di.CfgChangeDetectionTiming(self.channel_di_trigger_from_camera, 
                                                    self.channel_di_trigger_from_camera, 
                                                    daq.DAQmx_Val_ContSamps, 
                                                    0)

                # Set where the starting trigger 
                self._task_di.CfgDigEdgeStartTrig(self.channel_di_trigger_from_camera, daq.DAQmx_Val_Rising)
                
                # Export DI signal to unused PFI pins, for clock and start
                self._task_di.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, self.channel_di_change_trigger)
                self._task_di.ExportSignal(daq.DAQmx_Val_StartTrigger, self.channel_di_start_trigger)
        
            #-------------------------------------------------#
            # Create DO laser control tasks
            if self._task_do == None:
                self._task_do = daq.Task("TaskDO")
                self._task_do.CreateDOChan(", ".join(self.address_channel_do), 
                                        "DO_LaserControl", 
                                        daq.DAQmx_Val_ChanForAllLines)
                
                # Configure change timing from camera trigger task
                self._task_do.CfgSampClkTiming(self.channel_di_change_trigger, 
                                            self.daq_sample_rate_hz,
                                            daq.DAQmx_Val_Rising,
                                            daq.DAQmx_Val_ContSamps, 
                                            self.samples_per_do_ch)
            
            # Write the output waveform
            samples_per_ch_ct_digital = ct.c_int32()
            self._task_do.WriteDigitalLines(self.samples_per_do_ch, 
                                            False, 
                                            10.0, 
                                            daq.DAQmx_Val_GroupByChannel, 
                                            self.do_waveform, 
                                            ct.byref(samples_per_ch_ct_digital), 
                                            None)

            #-------------------------------------------------#
            # Create AO tasks, dependent on acquisition scan mode
            # first, set the scan and projection galvo to the initial point if it is not already
            samples_per_ch_ct = ct.c_int32()
            with daq.Task("TaskInitAO") as _task:
                # Create a 2d array that sets the initial AO voltage to the start of the scan.
                initial_ao_waveform = np.column_stack((np.full(2, self.ao_waveform[0,0]),
                                                       np.full(2, self.ao_waveform[0,1])))
                _task.CreateAOVoltageChan(self.address_ao_mirrors[0], 
                                             "initialize_ao0", 
                                             -6.0, 6.0, daq.DAQmx_Val_Volts, None)
                _task.CreateAOVoltageChan(self.address_ao_mirrors[1], 
                                             "initialize_ao1",
                                             -1.0, 1.0, daq.DAQmx_Val_Volts, None)
                _task.WriteAnalogF64(1, True, 1, daq.DAQmx_Val_GroupByScanNumber, 
                                        initial_ao_waveform, ct.byref(samples_per_ch_ct), None)
                _task.StopTask()
                _task.ClearTask()
                
            # Create and configure timing for AO tasks
            if self._task_ao == None:
                self._task_ao = daq.Task("TaskAO")
                self._task_ao.CreateAOVoltageChan(self.address_ao_mirrors[0], 
                                                "AO_ImageScanning", 
                                                -6.0, 6.0, daq.DAQmx_Val_Volts, None)
                self._task_ao.CreateAOVoltageChan(self.address_ao_mirrors[1], 
                                                "AO_ProjectionScanning", 
                                                -1.0, 1.0, daq.DAQmx_Val_Volts, None)

                if self.scan_type=='mirror':
                    # Configure timing to change on di change trigger, matches do_waveform shape
                    self._task_ao.CfgSampClkTiming(self.channel_di_change_trigger,
                                                self.daq_sample_rate_hz, 
                                                daq.DAQmx_Val_Rising, 
                                                daq.DAQmx_Val_ContSamps,
                                                self.samples_per_do_ch)
                elif self.scan_type=="projection":
                    # Configure to run on internal clock, ao_waveform has shape dictated by camera exposure
                    self._task_ao.CfgSampClkTiming("",
                                                    self.daq_sample_rate_hz,  # Define how fast the samples are output
                                                    daq.DAQmx_Val_Rising,
                                                    daq.DAQmx_Val_FiniteSamps,  # Output finite number of samples
                                                    self.ao_waveform.shape[0])  # Total samples

                    # Configure AO to start on the rising edge of DI signal
                    self._task_ao.CfgDigEdgeStartTrig(self.channel_di_trigger_from_camera,
                                                    daq.DAQmx_Val_Rising)
                                    
                    # Make the task retriggerable, for every exposure
                    self._task_ao.SetStartTrigRetriggerable(True)
                   
            # Write waveforms
            if self.scan_type=='mirror':
                # Write the output waveform
                self._task_ao.WriteAnalogF64(self.samples_per_do_ch,
                                             False, 10.0, daq.DAQmx_Val_GroupByScanNumber, 
                                             self.ao_waveform, ct.byref(samples_per_ch_ct), None)
            elif self.scan_type=="projection":                
                # Write the output waveform
                samples_per_ch_ct = ct.c_int32()
                self._task_ao.WriteAnalogF64(self.samples_per_ao_ch,
                                             False, 10.0, daq.DAQmx_Val_GroupByScanNumber, 
                                             self.ao_waveform, ct.byref(samples_per_ch_ct), None)                
        except daq.DAQError as err:
            print("DAQmx Error %s"%err)
     
     
    def start_waveform_playback(self):
        """Starts any tasks that exist.

        """
        try:
            for _task in [self._task_di, self._task_do, self._task_ao]:
                if _task:
                    _task.StartTask()
        except daq.DAQError as err:
            print("DAQmx Error %s"%err)


    def stop_waveform_playback(self):
        """Stop any tasks that exist."""
        try:
            for _task in [self._task_do, self._task_ao]:
                if _task:
                    _task.StopTask()
        
        except daq.DAQError as err:
            print("DAQmx Error %s"%err)
      
        
    def clear_tasks(self):
        """Stop, Clear and remove task handlers
        """
        try:
            for _task in [self._task_di, self._task_do, self._task_ao]:
                if hasattr(self, _task):
                    task = getattr(self, _task)
                    if task is not None:
                        task.StopTask()
                        task.ClearTask()
                        setattr(self, _task, None) 
        
        except daq.DAQError as err:
            print("DAQmx Error %s"%err)
    
    
    def __del__(self):
        """Set DO to 0s, AO to neutral positions, clear tasks
        """
        self.reset_ao_channels()
        self.reset_do_channels()  