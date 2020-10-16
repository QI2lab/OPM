#!/usr/bin/env python

'''
OPM stage control

Shepherd 10/20
'''

# imports
from pycromanager import Bridge, Acquisition
import numpy as np
from pathlib import Path
import time

def hook_fn(event,bridge,event_queue):
    
    core=bridge.get_core()

    command='1SCAN'
    core.set_property('TigerCommHub','SerialCommand',command)

    return event

def main():

    bridge = Bridge()
    core = bridge.get_core()

    # lasers to use
    # 0 -> inactive
    # 1 -> active

    state_405 = 0
    state_488 = 1
    state_561 = 0
    state_635 = 0
    state_730 = 0

    # laser powers (0 -> 100%)

    power_405 = 0
    power_488 = 0
    power_561 = 0
    power_635 = 0
    power_730 = 0

    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # FOV parameters
    ROI = [1024, 0, 256, 2048] #unit: pixels

    # camera exposure
    exposure_ms = 10 #unit: ms

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_asi = 280000 #unit: 1/10 um
    scan_axis_end_asi = 290000 #unit: 1/10 um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_asi = 146000 #unit: 1/10 um
    tile_axis_end_asi = 145000 #unit: 1/10 um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_asi = 690000 #unit: 1/10 um
    height_axis_end_asi = 695000 #unit: 1/10 um

    save_directory = Path('F:/20201016/')

    # scan axis setup
    scan_axis_step_um = 0.2  # unit: um
    scan_axis_step_mm = scan_axis_step_um / 1000.
    scan_axis_start_mm = scan_axis_start_asi / 10000.
    scan_axis_end_mm = scan_axis_end_asi / 10000.
    scan_axis_range_asi = np.abs(scan_axis_end_asi-scan_axis_start_asi)  # unit: 1/10 um
    scan_axis_range_mm = scan_axis_range_asi / 10000 #unit: mm
    exposure_s = exposure_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / exposure_s,2) #unit: mm/s
    number_of_images = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)

    # tile axis setup
    tile_axis_overlap=0.2
    tile_axis_range_asi = np.abs(tile_axis_end_asi - tile_axis_start_asi) #unit: 1/10 um
    tile_axis_range_mm = tile_axis_range_asi / 10000 #unit: mm
    tile_axis_ROI = ROI[3]-ROI[1] #unit: pixel
    tile_step_um = (tile_axis_ROI*pixel_size_um) * (1-tile_axis_overlap) #unit: um
    tile_step_mm = (tile_axis_ROI*pixel_size_um) * (1-tile_axis_overlap) * .001 #unit: mm
    tile_step_asi = (tile_axis_ROI*pixel_size_um) * (1-tile_axis_overlap) * 10.0 #unit: 1/10 um

    # height axis setup
    # this is more complicated, since we have an oblique light sheet
    # the height of the scan is the length of the ROI in the tilted direction * sin(tilt angle)
    # however, it may be better to hardcode displacement based on measurements of the light sheet Rayleigh length
    # for now, go with overlap calculation
    height_axis_overlap=0.2
    height_axis_range_asi = np.abs(height_axis_end_asi-height_axis_start_asi)
    height_axis_ROI = ROI[3]-ROI[1]*pixel_size_um #unit: um
    height_axis_step_um = (height_axis_ROI*pixel_size_um*np.sin(30.*(np.pi/180.)))*(1-height_axis_overlap) # unit: um
    height_axis_step_mm = (height_axis_ROI*pixel_size_um*np.sin(30.*(np.pi/180.)))*(1-height_axis_overlap) * .001 #unit: mm
    height_axis_step_asi = (height_axis_ROI*pixel_size_um*np.sin(30.*(np.pi/180.)))*(1-height_axis_overlap) * 10.0 #unit: 1/10 um


    # dummy variables for now to test stage
    tile_axis_positions_asi=[150000,152000]
    height_axis_positions_asi=[695000]

    # Setup PLC card to give start trigger
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    addrOutputBNC3 = 35
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC3)
    core.set_property(plcName, propCellConfig, addrStageSync)

    # allow for repeated commands to Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set tile axis speed for all moves
    command = 'SPEED Y=.5'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # move tile axis to initial position
    # expects 1/10 um
    command = 'MOVE Y='+str(tile_axis_start_asi)
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure stage has finished move
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis speed for large move to initial position
    command = 'SPEED X=.5'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # move scan scan stage to initial position
    # expects 1/10 um
    command = 'MOVE X='+str(scan_axis_start_asi)
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure stage has finished move
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis speed to correct speed for continuous stage scan
    # expects mm/s
    command = 'SPEED X='+str(scan_axis_speed)
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis to true 1D scan with no backlash
    command = '1SCAN X? Y=0 Z=9 F=0'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set range for scan axis
    # expects mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=50'
    core.set_property('TigerCommHub','SerialCommand',command)
    answer = core.get_property('TigerCommHub','SerialCommand')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)
        
    # turn off repeated commands to Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')
    answer = core.get_property('TigerCommHub','SerialCommand')

    # set camera into 16bit readout mode
    # give camera time to change modes if necessary
    core.set_property('Camera','ReadoutRate','100MHz 16bit')
    time.sleep(5)

    # set camera into low noise readout mode
    # give camera time to change modes if necessary
    core.set_property('Camera','Gain','2-CMS')
    time.sleep(5)

    # set camera to trigger first mode
    # give camera time to change modes if necessary
    core.set_property('Camera','TriggerMode','Trigger first')
    time.sleep(5)

    # crop FOV
    core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # set all lasers to off and user defined power
    core.set_config('Obis-State-405','Off')
    core.wait_for_config('Obis-State-405','Off')
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])

    core.set_config('Obis-State-488','Off')
    core.wait_for_config('Obis-State-488','Off')
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])

    core.set_config('Obis-State-561','Off')
    core.wait_for_config('Obis-State-561','Off')
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])

    core.set_config('Obis-State-637','Off')
    core.wait_for_config('Obis-State-637','Off')
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])

    core.set_config('Obis-State-730','Off')
    core.wait_for_config('Obis-State-730','Off')
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

     # create events to hold all of the scan axis images during constant speed stage scan
    # we call this 'z' here, even though it is actually oblique images acquired by moving scan axis (x) in our system
    events = []
    for z in range(number_of_images):
        events.append({'axes': {'z': z}})

    # loop through all tile axis positions
    for y in range(len(tile_axis_positions_asi)):   

        # loop through all height axis positions at current tile axis position
        for zstage in range(len(height_axis_positions_asi)):

            # loop through all channels at current height axis and tile axis positions
            for c in range(len(channel_states)):
                if channel_states[c] == 1:

                    #set laser
                    if (c==0):
                        core.set_config("Obis-State-405","On")
                        core.wait_for_config("Obis-State-405","On")
                    elif (c==1):
                        core.set_config("Obis-State-488","On")
                        core.wait_for_config("Obis-State-488","On")
                    elif (c==2):
                        core.set_config("Obis-State-561","On")
                        core.wait_for_config("Obis-State-561","On")
                    elif (c==3):
                        core.set_config("Obis-State-637","On")
                        core.wait_for_config("Obis-State-637","On")
                    elif (c==4):
                        core.set_config("Obis-State-730","On")
                        core.wait_for_config("Obis-State-730","On")

                    # setup file name
                    save_name = 'scan_'+'y_'+str(y).zfill(4)+'x_'+str(zstage).zfill(4)+'c_'+str(c).zfill(2)

                    # run acquisition
                    # TO DO: properly handle an error here if camera driver fails to return expected number of images.
                    with Acquisition(directory=save_directory, name=save_name, 
                                     post_camera_hook_fn=hook_fn,show_display=False, max_multi_res_index=0) as acq:
                        acq.acquire(events,keep_shutter_open=True)

                    core=bridge.get_core()

                    # allow for repeated commands to Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

                    # ensure scan axis has returned to initial position
                    ready='B'
                    while(ready!='N'):
                        command = 'STATUS'
                        core.set_property('TigerCommHub','SerialCommand',command)
                        ready = core.get_property('TigerCommHub','SerialResponse')
                        time.sleep(1)

                    # turn off repeated commands to Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

                    if (c==0):
                        core.set_config("Obis-State-405","Off")
                        core.wait_for_config("Obis-State-405","Off")
                    elif (c==1):
                        core.set_config("Obis-State-488","Off")
                        core.wait_for_config("Obis-State-488","Off")
                    elif (c==2):
                        core.set_config("Obis-State-561","Off")
                        core.wait_for_config("Obis-State-561","Off")
                    elif (c==3):
                        core.set_config("Obis-State-637","Off")
                        core.wait_for_config("Obis-State-637","Off")
                    elif (c==4):
                        core.set_config("Obis-State-730","Off")
                        core.wait_for_config("Obis-State-730","Off")

            # allow for repeated commands to Tiger
            core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

            # move height axis to new position
            command = 'MOV M='+str(height_axis_positions_asi[zstage])
            core.set_property('TigerCommHub','SerialCommand',command)
            answer = core.get_property('TigerCommHub','SerialCommand')

            # make sure stage has finished move
            ready='B'
            while(ready!='N'):
                command = 'STATUS'
                core.set_property('TigerCommHub','SerialCommand',command)
                ready = core.get_property('TigerCommHub','SerialResponse')
                time.sleep(.500)

            # turn off repeated commands to Tiger
            core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

        # allow for repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

        # move tile axis to new position
        command = 'MOV Y='+str(tile_axis_positions_asi[y])
        core.set_property('TigerCommHub','SerialCommand',command)
        answer = core.get_property('TigerCommHub','SerialCommand')

        # make sure stage has finished move
        ready='B'
        while(ready!='N'):
            command = 'STATUS'
            core.set_property('TigerCommHub','SerialCommand',command)
            ready = core.get_property('TigerCommHub','SerialResponse')
            time.sleep(.500)

        # turn off repeated commands to Tiger
        core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

        #-----------------------------------------------------------------------------------------------------------
        # cycle camera mode after one tile to avoid issues with Photometrics driver not returning expected
        # number of images. This always occurs after a acquiring large number of images in a single run (n~5 million)
        # STATUS OF BUG: 2020.10.04 - Photometrics acknowledges this is a problem that we encounter and they verify.
        #                             No solution offered.

        # set camera to internal trigger mode
        core.set_property('Camera','TriggerMode','Internal Trigger')
        time.sleep(10)

        # set camera to trigger first mode
        core.set_property('Camera','TriggerMode','Trigger first')
        time.sleep(10)
        #-----------------------------------------------------------------------------------------------------------

    # set camera to internal trigger mode
    core.set_property('Camera','TriggerMode','Internal Trigger')

# run
if __name__ == "__main__":
    main()