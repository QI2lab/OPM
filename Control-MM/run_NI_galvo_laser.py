import PyDAQmx as daq
import numpy as np
import ctypes as ct

samples_per_ch = 21 # must be odd number
DAQ_sample_rate_Hz = 10000

# Generate values for DO
dataDO = np.zeros((samples_per_ch,8),dtype=np.uint8)
dataDO[::2,:] = 1
dataDO[-1, :] = 0
print(dataDO)

# Generate values for AO
min_volt = 1
max_volt = 5
wf = np.zeros(samples_per_ch)
wf_temp = np.linspace(min_volt,max_volt,samples_per_ch//2)
wf[1::2] = wf_temp # start : end : step
wf[2::2] = wf_temp
wf[-1] = 0

try:    
    # ----- DIGITAL input -------
    taskDI = daq.Task()
    taskDI.CreateDIChan("/Dev1/PFI0","",daq.DAQmx_Val_ChanForAllLines)
    taskDI.CfgInputBuffer(0)
    
    #Configure change detectin timing (from wave generator)   
    taskDI.CfgChangeDetectionTiming("/Dev1/PFI0","/Dev1/PFI0",daq.DAQmx_Val_ContSamps,0)
    
    #Set where the starting trigger 
    taskDI.CfgDigEdgeStartTrig("/Dev1/PFI0",daq.DAQmx_Val_Rising)
    
    taskDI.ExportSignal(daq.DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI2")
    taskDI.ExportSignal(daq.DAQmx_Val_StartTrigger, "/Dev1/PFI1")
    
    # ----- DIGITAL output ------   
    taskDO = daq.Task()
    taskDO.CreateDOChan("/Dev1/port0/line0:7","",daq.DAQmx_Val_ChanForAllLines)

    ## Stop any task
    taskDO.StopTask()
    
    ## Configure timing (from DI task) 
    taskDO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_FiniteSamps,samples_per_ch)
    
    ## Set where the starting trigger 
    taskDO.CfgDigEdgeStartTrig("/Dev1/PFI1",daq.DAQmx_Val_Rising)
    
    ## Write the output waveform
    taskDO.WriteDigitalLines(samples_per_ch,False,10.0,daq.DAQmx_Val_GroupByChannel,dataDO,None,None)

    # ------- ANALOG output -----------
    taskAO = daq.Task()
    taskAO.CreateAOVoltageChan("/Dev1/ao0","",-10.0,10.0,daq.DAQmx_Val_Volts,None)
    
    ## Stop any task
    taskAO.StopTask()
    
    ## Configure timing (from DI task)
    taskAO.CfgSampClkTiming("/Dev1/PFI2",DAQ_sample_rate_Hz,daq.DAQmx_Val_Rising,daq.DAQmx_Val_FiniteSamps,samples_per_ch)
   
    ## Set where the starting trigger 
    taskAO.CfgDigEdgeStartTrig("/Dev1/PFI1",daq.DAQmx_Val_Rising)
    
    ## Write the output waveform
    samples_per_ch_ct = ct.c_int32()
    taskAO.WriteAnalogF64(samples_per_ch, False, 10.0, daq.DAQmx_Val_GroupByScanNumber,wf,ct.byref(samples_per_ch_ct),None)

    ## ------ Start both tasks ----------
    taskDI.StartTask()
    taskAO.StartTask()    
    taskDO.StartTask()    
    taskAO.WaitUntilTaskDone(-1) # wait until signal is sent
    taskDO.WaitUntilTaskDone(-1) # wait until signal is sent
   
    ## Stop and clear both tasks
    taskDI.StopTask()
    taskAO.StopTask()
    taskDO.StopTask()
    taskDI.ClearTask()
    taskAO.ClearTask()
    taskDO.ClearTask()
    
except daq.DAQError as err:
    print("DAQmx Error %s"%err)