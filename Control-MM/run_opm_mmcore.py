#!/usr/bin/env python
'''
OPM stage control
Reverting to calling core MM functions for acquisition and npy2bdv for storage.
NDTiffStorage is extremely slow to load datasets when storing a lot of data.

Open question if Python-Java bridge is fast enough to pull images to store in H5...we'll find out.
It isn't quite fast enough to keep up with data generation.
End up with 20% of total images in sequence buffer at end of scan.


TO DO:
1. Investigate adding rotation + deskew affine after creating H5/XML.
2. Swap out Python-Java bridge for native camera API to avoid pulling across Pycromanager bridge.
3. Integrate fluidics components from MERFISH branch. These will go into TIME dimension in BDV H5.
4. Optimize block size for writing. Can we do this over network?

Shepherd 11/20
'''

# imports
from pycromanager import Bridge
from pathlib import Path
import npy2bdv
import numpy as np
import time

def main():

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------Begin setup of scan parameters--------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # lasers to use
    # 0 -> inactive
    # 1 -> active
    state_405 = 1
    state_488 = 0
    state_561 = 0
    state_635 = 1
    state_730 = 0

    # laser powers (0 -> 100%)
    power_405 = 10
    power_488 = 0
    power_561 = 0
    power_635 = 10
    power_730 = 0

    # exposure time
    exposure_ms = 5.

    # scan axis limits. Use stage positions reported by MM
    scan_axis_start_um = -26000. #unit: um
    scan_axis_end_um = -25500. #unit: um

    # tile axis limits. Use stage positions reported by MM
    tile_axis_start_um = -7000 #unit: um
    tile_axis_end_um = -6500. #unit: um

    # height axis limits. Use stage positions reported by MM
    height_axis_start_um = 345.#unit: um
    height_axis_end_um = 375. #unit:  um

    # FOV parameters
    # ONLY MODIFY IF NECESSARY
    ROI = [0, 1024, 1599, 255] #unit: pixels

    # setup file name
    save_directory=Path('E:/20201130/')
    save_name = Path('shaffer_lung_v1.h5')

    #------------------------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------End setup of scan parameters----------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------

    # instantiate the Python-Java bridge to MM
    bridge = Bridge()
    core = bridge.get_core()

    # turn off lasers
    core.set_config('Coherent-State','off')
    core.wait_for_config('Coherent-State','off')

    # set camera into 16bit readout mode
    core.set_property('Camera','ReadoutRate','100MHz 16bit')
    time.sleep(1)

    # set camera into low noise readout mode
    core.set_property('Camera','Gain','2-CMS')
    time.sleep(1)

    # set camera to trigger first mode
    # TO DO: photometrics claims this setting doesn't exist in PVCAM. Why is it necessary then?
    core.set_property('Camera','Trigger Timeout (secs)',300)
    time.sleep(1)

    # set camera to internal trigger
    core.set_property('Camera','TriggerMode','Internal Trigger')
    time.sleep(1)

    # change core timeout for long stage moves
    core.set_property('Core','TimeoutMs',100000)

    # crop FOV
    #core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # grab one image to determine actual framerate


    # get actual framerate from micromanager properties
    actual_readout_ms = float(core.get_property('Camera','ActualInterval-ms')) #unit: ms

    # camera pixel size
    pixel_size_um = .115 # unit: um

    # scan axis setup
    scan_axis_step_um = 0.4  # unit: um
    scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
    scan_axis_start_mm = scan_axis_start_um / 1000. #unit: mm
    scan_axis_end_mm = scan_axis_end_um / 1000. #unit: mm
    scan_axis_range_um = np.abs(scan_axis_end_um-scan_axis_start_um)  # unit: um
    scan_axis_range_mm = scan_axis_range_um / 1000 #unit: mm
    actual_exposure_s = actual_readout_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / actual_exposure_s,2) #unit: mm/s
    scan_axis_positions = np.rint(scan_axis_range_mm / scan_axis_step_mm).astype(int)  #unit: number of positions

    # tile axis setup
    tile_axis_overlap=0.2 #unit: percentage
    tile_axis_range_um = np.abs(tile_axis_end_um - tile_axis_start_um) #unit: um
    tile_axis_range_mm = tile_axis_range_um / 1000 #unit: mm
    tile_axis_ROI = ROI[2]*pixel_size_um  #unit: um
    tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
    tile_axis_step_mm = tile_axis_step_um / 1000 #unit: mm
    tile_axis_positions = np.rint(tile_axis_range_mm / tile_axis_step_mm).astype(int)  #unit: number of positions
    # if tile_axis_positions rounded to zero, make sure we acquire at least one position
    if tile_axis_positions == 0:
        tile_axis_positions=1

    # height axis setup
    # this is more complicated, since we have an oblique light sheet
    # the height of the scan is the length of the ROI in the tilted direction * sin(tilt angle)
    height_axis_overlap=0.2 #unit: percentage
    height_axis_range_um = np.abs(height_axis_end_um-height_axis_start_um) #unit: um
    height_axis_range_mm = height_axis_range_um / 1000 #unit: mm
    height_axis_ROI = ROI[3]*pixel_size_um*np.sin(30*(np.pi/180.)) #unit: um
    height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
    height_axis_step_mm = height_axis_step_um / 1000  #unit: mm
    height_axis_positions = np.rint(height_axis_range_mm / height_axis_step_mm).astype(int) #unit: number of positions
    # if height_axis_positions rounded to zero, make sure we acquire at least one position
    if height_axis_positions==0:
        height_axis_positions=1

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # Setup PLC card to give start trigger
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    #addrOutputBNC3 = 35
    addrOutputBNC1 = 33
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    
    # connect stage sync signal to BNC output
    core.set_property(plcName, propPosition, addrOutputBNC1)
    core.set_property(plcName, propCellConfig, addrStageSync)

    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set tile axis speed for all moves
    command = 'SPEED Y=.1'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set scan axis speed for large move to initial position
    command = 'SPEED X=.1'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # move scan scan stage to initial position
    core.set_xy_position(scan_axis_start_um,tile_axis_start_um)
    core.wait_for_device(xy_stage)
    core.set_position(height_axis_start_um)
    core.wait_for_device(z_stage)

    # turn on 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # set scan axis speed to correct speed for continuous stage scan
    # expects mm/s
    command = 'SPEED X='+str(scan_axis_speed)
    core.set_property('TigerCommHub','SerialCommand',command)

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

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # set range and return speed (10% of max) for scan axis
    # expects mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=10'
    core.set_property('TigerCommHub','SerialCommand',command)

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)
  
    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

    # construct boolean array for lasers to use
    channel_states = [state_405,state_488,state_561,state_635,state_730]
    channel_powers = [power_405,power_488,power_561,power_635,power_730]

    # set lasers to user defined power
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    # calculate total tiles
    total_tiles = tile_axis_positions * height_axis_positions

    # output acquisition metadata
    print('Number of X positions: '+str(scan_axis_positions))
    print('Number of Y tiles: '+str(tile_axis_positions))
    print('Number of Z slabs: '+str(height_axis_positions))
    print('Number of channels:' +str(np.sum(channel_states)))
    print('Number of BDV H5 tiles: '+str(total_tiles))

    # define unit tranformation matrix
    unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                        (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                        (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

    # create BDV H5 using npy2bdv
    fname = save_directory/save_name
    bdv_writer = npy2bdv.BdvWriter(fname, nchannels=np.sum(channel_states), ntiles=total_tiles, subsamp=((1, 1, 1),), blockdim=((1,128,256),))

    # reset tile index for BDV H5 file
    tile_index = 0
    
    for y in range(tile_axis_positions):
        # calculate tile axis position
        tile_position_um = tile_axis_start_um+(tile_axis_step_um*y)
        
        # move XY stage to new tile axis position
        core.set_xy_position(scan_axis_start_um,tile_position_um)
        core.wait_for_device(xy_stage)
            
        for z in range(height_axis_positions):

            print('Tile index: '+str(tile_index))

            # calculate height axis position
            height_position_um = height_axis_start_um+(height_axis_step_um*z)

            # move Z stage to new height axis position
            core.set_position(height_position_um)
            core.wait_for_device(z_stage)

            # reset channel index for BDV H5 file
            channel_index = 0
            
            for c in range(len(channel_states)):

                # determine active channel
                if channel_states[c]==1:
                    if (c==0):
                        core.set_config('Coherent-State','405nm')
                        core.wait_for_config('Coherent-State','405nm')
                    elif (c==1):
                        core.set_config('Coherent-State','488nm')
                        core.wait_for_config('Coherent-State','488nm')
                    elif (c==2):
                        core.set_config('Coherent-State','561nm')
                        core.wait_for_config('Coherent-State','561nm')
                    elif (c==3):
                        core.set_config('Coherent-State','637nm')
                        core.wait_for_config('Coherent-State','637nm')
                    elif (c==4):
                        core.set_config('Coherent-State','730nm')
                        core.wait_for_config('Coherent-State','730nm')

                    print('Channel index: '+str(channel_index))
                    print('Active channel: '+str(c))

                    # set camera to trigger first mode for stage synchronization
                    core.set_property('Camera','TriggerMode','Trigger first')
                    time.sleep(1)

                    # get current X, Y, and Z stage positions for translation transformation
                    point = core.get_xy_stage_position()
                    x_now = point.get_x()
                    y_now = point.get_y()
                    z_now = core.get_position(z_stage)
                            
                    # calculate affine matrix components for translation transformation
                    affine_matrix = unit_matrix
                    affine_matrix[0,3] = y_now/pixel_size_um # x axis in BDV H5 (tile axis on scope).
                    affine_matrix[1,3] = z_now/(pixel_size_um*np.sin(30.*np.pi/180.)) # y axis in BDV H5 (height axis on scope).
                    affine_matrix[2,3] = x_now/(pixel_size_um*np.cos(30.*np.pi/180.)) # z axis in BDV H5 (scan axis on scope).

                    # turn off 'transmit repeated commands' for Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

                    # check to make sure Tiger is not busy
                    ready='B'
                    while(ready!='N'):
                        command = 'STATUS'
                        core.set_property('TigerCommHub','SerialCommand',command)
                        ready = core.get_property('TigerCommHub','SerialResponse')
                        time.sleep(.500)

                    # turn off 'transmit repeated commands' for Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

                    # start acquistion
                    core.start_sequence_acquisition(int(scan_axis_positions),float(0.0),True)

                    # tell stage to execute scan
                    command='1SCAN'
                    core.set_property('TigerCommHub','SerialCommand',command)

                    # reset image counter
                    image_counter = 0

                    # place stack into BDV H5
                    bdv_writer.append_view(stack=None, virtual_stack_dim=(scan_axis_positions,ROI[3],ROI[2]), time=0, channel=channel_index, tile=tile_index, 
                                                m_affine=affine_matrix, name_affine = 'stage translation', 
                                                voxel_size_xyz=(.115,.115,.200), voxel_units='um')

                    # grab images from buffer
                    while (image_counter < scan_axis_positions):

                        # if there are images in the buffer, grab and process
                        if (core.get_remaining_image_count() > 0):
                            # grab top image in buffer
                            tagged_image = core.pop_next_tagged_image()

                            # grab metadata to convert 1D array to 2D image
                            image_height = tagged_image.tags['Height']
                            image_width = tagged_image.tags['Width']

                            # convert to 2D image and place into virtual stack in BDV H5
                            bdv_writer.append_plane(plane=np.flipud(tagged_image.pix.reshape((image_height,image_width))), 
                                                    plane_index=image_counter, time=0, channel=channel_index)

                            # increment image counter
                            image_counter = image_counter + 1

                        # no images in buffer, wait for another image to arrive.
                        else:
                            time.sleep(np.minimum(.01*exposure_ms, 1)/1000)
                    
                    # clean up acquistion
                    core.stop_sequence_acquisition()

                    # turn off lasers
                    core.set_config('Coherent-State','off')
                    core.wait_for_config('Coherent-State','off')

                    # set camera to internal trigger
                    # this is necessary to avoid PVCAM driver issues that we keep having for long acquisitions.
                    core.set_property('Camera','TriggerMode','Internal Trigger')
                    time.sleep(1)

                    # increment channel index for BDV H5 file
                    channel_index = channel_index + 1

                    # turn off 'transmit repeated commands' for Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','No')

                    # check to make sure Tiger is not busy
                    ready='B'
                    while(ready!='N'):
                        command = 'STATUS'
                        core.set_property('TigerCommHub','SerialCommand',command)
                        ready = core.get_property('TigerCommHub','SerialResponse')
                        time.sleep(.500)

                    # turn off 'transmit repeated commands' for Tiger
                    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')
                    
            # increment tile index for BDV H5
            tile_index = tile_index + 1

    # write BDV XML and close BDV H5
    bdv_writer.write_xml_file(ntimes=1)
    bdv_writer.close()

# run
if __name__ == "__main__":
    main()