
#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
Basic class to run NIDAQ with camera as master for OPM using PyDAQMx 
----------------------------------------------------------------------------------------
Peter Brown
Franky Djutanta
Douglas Shepherd
12/11/2021
douglas.shepherd@asu.edu
----------------------------------------------------------------------------------------
'''

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
import PyDAQmx as daq
import ctypes as ct
import numpy as np

class OPMNIDAQ:

    def __init__(self,scan_mirror_neutral=0.0,scan_mirror_calibration=0.043):

        self.scan_type = 'mirror'
        self.interleave_lasers = True
        self.do_ind = [0,1,2,3,4]
        self.active_channels_indices = None
        self.n_active_channels = 0

        self.DAQ_sample_rate_Hz = 10000
        self.num_DI_channels = 8
        self.dataDO = None
        self.waveform = None
        self.channelAO = "/Dev1/ao0"
        self.min_AO_voltage = -7.0
        self.max_AO_voltage = 7.0
        self.channelDO = "/Dev1/port0/line0:7"
        self.channelDI_trigger_from_camera = "/Dev1/PFI0"
        self.channelDI_start_trigger = "/Dev1/PFI1"
        self.channelDI_change_trigger = "/Dev1/PFI2"

        self.scan_mirror_neutral = scan_mirror_neutral
        self.scan_mirror_calibration = scan_mirror_calibration
        self.laser_blanking=False
        
    def set_laser_blanking(self,laser_blanking):
        self.laser_blanking=laser_blanking
    
    def set_scan_type(self,scan_type):
        self.scan_type = scan_type

    def reset_scan_mirror(self):
        self.taskAO = daq.Task()
        self.taskAO.CreateAOVoltageChan("/Dev1/ao0","",-6.0,6.0,daq.DAQmx_Val_Volts,None)
        self.taskAO.WriteAnalogScalarF64(True, -1, self.scan_mirror_neutral, None)
        self.taskAO.StartTask()
        self.taskAO.StopTask()
        self.taskAO.ClearTask()
    
    def set_scan_mirror_range(self,scan_mirrror_step_size_um,scan_mirror_sweep_um):
        # determine sweep footprint
        self.min_volt = -(scan_mirror_sweep_um * self.scan_mirror_calibration / 2.) + self.scan_mirror_neutral # unit: volts
        self.scan_axis_step_volts = scan_mirrror_step_size_um * self.scan_mirror_calibration # unit: V
        self.scan_axis_range_volts = scan_mirror_sweep_um * self.scan_mirror_calibration # unit: V
        self.scan_steps = np.rint(self.scan_axis_range_volts / self.scan_axis_step_volts).astype(np.int16) # galvo steps

        return self.scan_steps

    def set_interleave_mode(self,interleave_lasers):
        self.interleave_lasers = interleave_lasers
    
    def set_channels_to_use(self,channel_states):
        self.active_channel_indices = [ind for ind, st in zip(self.do_ind, channel_states) if st]
        self.n_active_channels = len(self.active_channel_indices)
    
    def generate_waveforms(self):
        if self.scan_type == 'mirror':
            # setup DAQ
            nvoltage_steps = self.scan_steps
            # 2 time steps per frame, except for first frame plus one final frame to reset voltage
            #samples_per_ch = (nvoltage_steps * 2 - 1) + 1
            self.samples_per_ch = (nvoltage_steps * 2 * self.n_active_channels - 1) + 1
 
            # Generate values for DO
            dataDO = np.zeros((self.samples_per_ch, self.num_DI_channels), dtype=np.uint8)
        
            for ii, ind in enumerate(self.active_channel_indices):
                if self.laser_blanking:
                    dataDO[2*ii::2*self.n_active_channels, ind] = 1
                else:
                    dataDO[:,int(ind)] = 1
            
            # dataDO[-1, int(ind)] = 0

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

            self.dataDO = dataDO
            self.waveform = waveform
        elif self.scan_type == 'stage':
            # setup digital trigger buffer on DAQ
            self.samples_per_ch = 2 * int(self.n_active_channels)

            # create DAQ pattern for laser strobing controlled via rolling shutter
            dataDO = np.zeros((self.samples_per_ch, self.num_DI_channels), dtype=np.uint8)
            for ii, ind in enumerate(self.active_channel_indices):
                if self.laser_blanking:
                    dataDO[2*ii::2*int(self.n_active_channels), int(ind)] = 1
                else:
                    dataDO[:,int(ind)] = 1

            if self.laser_blanking:
                dataDO[-1, :] = 0
            
            self.dataDO = dataDO
            self.waveform = None

        elif self.scan_type == '2D':
            # setup DAQ
            nvoltage_steps = 1
            # 2 time steps per frame, except for first frame plus one final frame to reset voltage
            #samples_per_ch = (nvoltage_steps * 2 - 1) + 1
            self.samples_per_ch = (nvoltage_steps * 2 * self.n_active_channels - 1) + 1
 
            # Generate values for DO
            dataDO = np.zeros((self.samples_per_ch, self.num_DI_channels), dtype=np.uint8)
        
            for ii, ind in enumerate(self.active_channel_indices):
                if self.laser_blanking:
                    dataDO[2*ii::2*self.n_active_channels, ind] = 1
                else:
                    dataDO[:,int(ind)] = 1
            
            if self.laser_blanking:
                dataDO[-1, :] = 0

            self.dataDO = dataDO
            self.waveform = None

    def start_waveform_playback(self):
        try:    
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

            ## Configure timing (from DI task) 
            self.taskDO.CfgSampClkTiming(self.channelDI_change_trigger, self.DAQ_sample_rate_Hz, daq.DAQmx_Val_Rising, daq.DAQmx_Val_ContSamps, self.samples_per_ch)
            
            ## Write the output waveform
            samples_per_ch_ct_digital = ct.c_int32()
            self.taskDO.WriteDigitalLines(self.samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByChannel, self.dataDO, ct.byref(samples_per_ch_ct_digital), None)

            if self.scan_type == 'mirror':
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
                
                # start analog tasks
                self.taskAO.StartTask()
            
            # start digital tasks
            self.taskDO.StartTask()    
            self.taskDI.StartTask()

        except daq.DAQError as err:
            print("DAQmx Error %s"%err)

    def stop_waveform_playback(self):
        try:
            self.taskDI.StopTask()
            self.taskDO.StopTask()
            if self.scan_type == 'mirror':
                self.taskAO.StopTask()

            self.taskDI.ClearTask()
            self.taskDO.ClearTask()
            if self.scan_type == 'mirror':
                self.taskAO.ClearTask()

            self.TaskDO = daq.Task()
            self.TaskDO.CreateDOChan("/Dev1/port0/line0:7", "", daq.DAQmx_Val_ChanForAllLines)
            array = np.zeros((self.samples_per_ch, self.num_DI_channels), dtype=np.uint8)
            self.TaskDO.WriteDigitalLines(1,1,10.0,daq.DAQmx_Val_GroupByChannel,array,None,None)
            self.TaskDO.StopTask()
            self.TaskDO.ClearTask()

            self.reset_scan_mirror()

        except daq.DAQError as err:
            print("DAQmx Error %s"%err)