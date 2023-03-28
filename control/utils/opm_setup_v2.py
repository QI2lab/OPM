import time
import gc
from pathlib import Path
import numpy as np
import easygui
from pycromanager import Core

def setup_asi_tiger(core,scan_axis_speed,scan_axis_start_mm,scan_axis_end_mm):
    """
    Setup ASI Tiger controller for constant speed stage scan

    :param core: Core
        current pycromanager core
    :param scan_axis_speed: float
        speed of scan axis in mm/s
    :param scan_axis_start_mm: float
        starting point for stage scan in mm
    :param scan_axis_end_mm: float
        stopping point for stage scan in mm
    
    :return None:
    """


    # Setup Tiger controller to pass signal when the scan stage cross the start position to the PLC
    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    addrOutputBNC1 = 33 # BNC1 on the PLC front panel
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

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        core.set_property('TigerCommHub','SerialCommand',command)
        ready = core.get_property('TigerCommHub','SerialResponse')
        time.sleep(.500)

    # turn off 'transmit repeated commands' for Tiger
    core.set_property('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

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

    # set range and return speed (5% of max) for scan axis
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


def setup_obis_laser_boxx(core,channel_powers,state):
    """
    Setup Coherent obis laser boxx

    :param core: Core
        current pycromanager core
    :param channel_powers: ndarray
        array of powers as percentage of max power
    :param state: str
        how laser boxx should expect input ('software' or 'digital')
    
    :return None:
    """

    # turn off lasers
    core.set_config('Laser','Off')
    core.wait_for_config('Laser','Off')

    # set lasers to user defined power
    core.set_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)',channel_powers[0])
    core.set_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)',channel_powers[1])
    core.set_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)',channel_powers[2])
    core.set_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)',channel_powers[3])
    core.set_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)',channel_powers[4])

    if state == 'software':
        # set all lasers to software control
        core.set_config('Modulation-405','CW (constant power)')
        core.wait_for_config('Modulation-405','CW (constant power)')
        core.set_config('Modulation-488','CW (constant power)')
        core.wait_for_config('Modulation-488','CW (constant power)')
        core.set_config('Modulation-561','CW (constant power)')
        core.wait_for_config('Modulation-561','CW (constant power)')
        core.set_config('Modulation-637','CW (constant power)')
        core.wait_for_config('Modulation-637','CW (constant power)')
        core.set_config('Modulation-730','CW (constant power)')
        core.wait_for_config('Modulation-730','CW (constant power)')

        # turn off lasers
        core.set_config('Laser','Off')
        core.wait_for_config('Laser','Off')
    elif state == 'digital':
        # set all laser to external triggering
        core.set_config('Modulation-405','External-Digital')
        core.wait_for_config('Modulation-405','External-Digital')
        core.set_config('Modulation-488','External-Digital')
        core.wait_for_config('Modulation-488','External-Digital')
        core.set_config('Modulation-561','External-Digital')
        core.wait_for_config('Modulation-561','External-Digital')
        core.set_config('Modulation-637','External-Digital')
        core.wait_for_config('Modulation-637','External-Digital')
        core.set_config('Modulation-730','External-Digital')
        core.wait_for_config('Modulation-730','External-Digital')

        # turn all lasers on
        core.set_config('Laser','AllOn')
        core.wait_for_config('Laser','AllOn')

def camera_hook_fn(event):
    """
    Hook function to start stage controller once camera is activated in EXTERNAL/START mode

    :param core: Core
        current pycromanager core
    :param event: dict
        dictionary of pycromanager events
    :param bridge: Bridge
        active pycromanager bridge between python and java
    :param event_queue: dict
        dictionary of pycromanager event queue
    
    :return None:
    """

    core_trigger = Core()
    command='1SCAN'
    core_trigger.set_property('TigerCommHub','SerialCommand',command)
    
    core_trigger = None
    del core_trigger
    gc.collect()

    return event

def retrieve_setup_from_MM(core,studio,df_config,debug=False):
    """
    Parse MM GUI to retrieve exposure time, channels to use, powers, and stage positions

    :param core: Core
        active pycromanager MMcore object
    :param core: Studio
        active pycromanager MMstudio object
    :param df_config: dict
        dictonary containing instrument setup information
    :param debug: boolean
        flag to bring debug information
    
    :return df_MM_setup: dict
        dictonary containing scan configuration settings from MM GUI
    """

    debug = True

    # pull current MDA window settings
    acq_manager = studio.acquisitions()
    acq_settings = acq_manager.get_acquisition_settings()

    # grab settings from MM
    # grab and setup save directory and filename
    save_directory=Path(acq_settings.root())
    save_name=Path(acq_settings.prefix())

    # pull active lasers from MDA window
    channel_labels = ['405', '488', '561', '637', '730']
    channel_states = [False,False,False,False,False] #define array to keep active channels
    channels = acq_settings.channels() # get active channels in MDA window
    for idx in range(channels.size()):
        channel = channels.get(idx) # pull channel information
        if channel.config() == channel_labels[0]: 
            channel_states[0]=True
        if channel.config() == channel_labels[1]: 
            channel_states[1]=True
        elif channel.config() == channel_labels[2]: 
            channel_states[2]=True
        elif channel.config() == channel_labels[3]: 
            channel_states[3]=True
        elif channel.config() == channel_labels[4]: 
            channel_states[4]=True
    do_ch_pins = [df_config['laser0_do_pin'], 
                  df_config['laser1_do_pin'], 
                  df_config['laser2_do_pin'], 
                  df_config['laser3_do_pin'], 
                  df_config['laser4_do_pin']] # digital output line corresponding to each channel
    
    # pull laser powers from main window
    channel_powers = [0.,0.,0.,0.,0.]
    channel_powers[0] = core.get_property('Coherent-Scientific Remote','Laser 405-100C - PowerSetpoint (%)')
    channel_powers[1] = core.get_property('Coherent-Scientific Remote','Laser 488-150C - PowerSetpoint (%)')
    channel_powers[2] = core.get_property('Coherent-Scientific Remote','Laser OBIS LS 561-150 - PowerSetpoint (%)')
    channel_powers[3] = core.get_property('Coherent-Scientific Remote','Laser 637-140C - PowerSetpoint (%)')
    channel_powers[4] = core.get_property('Coherent-Scientific Remote','Laser 730-30C - PowerSetpoint (%)')

    # parse which channels are active
    active_channel_indices = [ind for ind, st in zip(do_ch_pins, channel_states) if st]
    n_active_channels = len(active_channel_indices)

    # set up XY positions
    position_list_manager = studio.positions()
    position_list = position_list_manager.get_position_list()
    number_positions = position_list.get_number_of_positions()
    x_positions = np.empty(number_positions)
    y_positions = np.empty(number_positions)
    z_positions = np.empty(number_positions)

    # iterate through position list to extract XY positions    
    for idx in range(number_positions):
        pos = position_list.get_position(idx)
        for ipos in range(pos.size()):
            stage_pos = pos.get(ipos)
            if (stage_pos.get_stage_device_label() == 'XYStage:XY:31'):
                x_positions[idx] = stage_pos.x
                y_positions[idx] = stage_pos.y
            if (stage_pos.get_stage_device_label() == 'ZStage:M:37'):
                z_positions[idx] = stage_pos.x

    # determine corners for XY stage and stop/bottom for Z stage
    # TO DO: setup interpolation and split up XY positions to avoid brute force Z scanning
    scan_axis_start_um = np.round(x_positions.min(),0)
    scan_axis_end_um = np.round(x_positions.max(),0)

    tile_axis_start_um = np.round(y_positions.min(),0)
    tile_axis_end_um = np.round(y_positions.max(),0)

    height_axis_start_um = np.round(z_positions.min(),0)
    height_axis_end_um = np.round(z_positions.max(),0)

    # set pixel size
    pixel_size_um = float(df_config['pixel_size']) # unit: um 

    # get exposure time from main window
    exposure_ms = core.get_exposure()

    # determine image size
    core.snap_image()
    y_pixels = core.get_image_height()
    x_pixels = core.get_image_width()

    core.set_config('Camera-Setup','ScanMode3')
    core.wait_for_config('Camera-Setup','ScanMode3')

    if debug: print(f'Exposure time: {exposure_ms}.')
    
    # enforce exposure time
    core.set_exposure(exposure_ms)

    # grab exposure
    true_exposure = core.get_exposure()

    # get actual framerate from micromanager properties
    #actual_readout_ms = true_exposure+float(core.get_property('OrcaFusionBT','ReadoutTime')) #unit: ms
    # DPS test how this change alters readout frames in strip scan
    actual_readout_ms = true_exposure
    if debug: print(f'Full readout time: {actual_readout_ms}.')

    # WIP: account for small coverslip tilt over large scans

    # calculate slope using height positions, assuming user set with coverslip at top of camera ROI
    coverslip_slope_um = np.round((np.abs(height_axis_end_um-height_axis_start_um) / np.abs(scan_axis_end_um-scan_axis_start_um)),6)
    if debug: 
        print(f'Coverslip low: {height_axis_start_um}')
        print(f'Coverslip high: {height_axis_end_um}')
        print(f'Scan start: {scan_axis_start_um}')
        print(f'Scan end: {scan_axis_end_um}')
        print(f'Coverslip slope: {coverslip_slope_um}')

    # maximum allowed height change
    # for now, hardcode to 10% of coverslip height
    max_height_change_um = 3.0

    # calculate allowed scan length and number of scan tiles for allowed coverslip height change
    scan_tile_length_um = np.round((max_height_change_um / coverslip_slope_um),2)
    num_scan_tiles = np.rint(np.abs(scan_axis_end_um-scan_axis_start_um) / scan_tile_length_um)
    

    # calculate scan axis tile locations
    scan_tile_overlap = .2 # unit: percentage
    scan_axis_step_um = float(df_config['scan_axis_step_um'])  # unit: um 

    scan_axis_step_mm = scan_axis_step_um / 1000. #unit: mm
    scan_axis_start_mm = scan_axis_start_um / 1000. #unit: mm
    scan_axis_end_mm = scan_axis_end_um / 1000. #unit: mm
    scan_tile_length_mm = scan_tile_length_um / 1000. # unit: mm

    scan_axis_start_pos_mm = np.round(np.arange(scan_axis_start_mm,scan_axis_end_mm+(1-scan_tile_overlap)*scan_tile_length_mm,(1-scan_tile_overlap)*scan_tile_length_mm),2) #unit: mm
    scan_axis_end_pos_mm = np.round(scan_axis_start_pos_mm + scan_tile_length_mm * (1+scan_tile_overlap),2)
    scan_axis_start_pos_mm = scan_axis_start_pos_mm[0:-1]
    scan_axis_end_pos_mm = scan_axis_end_pos_mm[0:-1]
    scan_tile_length_w_overlap_mm = scan_axis_end_pos_mm[0]-scan_axis_start_pos_mm[0]
    scan_axis_positions = np.rint(scan_tile_length_w_overlap_mm / scan_axis_step_mm).astype(int)
    num_scan_tiles = len(scan_axis_start_pos_mm)
    actual_exposure_s = actual_readout_ms / 1000. #unit: s
    scan_axis_speed = np.round(scan_axis_step_mm / actual_exposure_s / n_active_channels,5) #unit: mm/s
    #scan_axis_positions = np.rint((scan_tile_length_mm* (1+scan_tile_overlap)) / scan_axis_step_mm).astype(int)  #unit: number of positions
    if debug: 
        print(f'Number scan tiles: {num_scan_tiles}')
        print(f'Scan axis start positions: {scan_axis_start_pos_mm}.')
        print(f'Scan axis end positions: {scan_axis_end_pos_mm}.')
        print(f'Scan axis positions: {scan_axis_positions}')
        print(f'Scan tile size: {scan_tile_length_w_overlap_mm}')

    # calculate starting height axis locations
    height_axis_start_pos_um = np.round(np.linspace(height_axis_start_um,height_axis_end_um,len(scan_axis_start_pos_mm)),2)
    if len(height_axis_start_pos_um) > 1:
        height_axis_step_um = float(height_axis_start_pos_um[1]-height_axis_start_pos_um[0])
    else:
        height_axis_step_um = 0
    if debug: print(f'Height axis start positions: {height_axis_start_pos_um}.')

    # calculate tile axis locations
    tile_axis_overlap=0.15 #unit: percentage
    tile_axis_ROI = x_pixels*pixel_size_um  #unit: um
    tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
    tile_axis_pos_um = np.round(np.arange(tile_axis_start_um,tile_axis_end_um+tile_axis_step_um,tile_axis_step_um),2)
    if debug: 
        print(f'Tile axis step: {tile_axis_step_um}')
        print(f'Tile axis start positions: {tile_axis_pos_um}.')

    # generate dictionary to return with scan parameters
    df_MM_setup = {'tile_axis_positions': int(len(tile_axis_pos_um)),
                    'tile_axis_start_um': float(tile_axis_start_um),
                    'tile_axis_end_um': float(tile_axis_end_um),
                    'tile_axis_step_um': float(tile_axis_step_um),
                    'coverslip_axis_start_um': float(height_axis_start_pos_um[0]),
                    'coverslip_axis_end_um': float(height_axis_start_pos_um[-1]),
                    'coverslip_axis_step_um': float(height_axis_step_um),
                    'n_active_channels': int(n_active_channels),
                    'scan_axis_tile_positions' : int(num_scan_tiles),
                    'scan_axis_positions': int(scan_axis_positions),
                    'scan_axis_start_mm': float(scan_axis_start_mm),
                    'scan_axis_end_mm': float(scan_axis_end_mm),
                    'scan_axis_start_um': float(scan_axis_start_um),
                    'scan_axis_end_um': float(scan_axis_end_um),
                    'scan_axis_speed': float(scan_axis_speed),
                    'save_directory': str(save_directory),
                    'save_name': str(save_name),
                    'y_pixels': int(y_pixels),
                    'x_pixels': int(x_pixels),
                    '405_active': bool(channel_states[0]),
                    '488_active': bool(channel_states[1]),
                    '561_active': bool(channel_states[2]),
                    '635_active': bool(channel_states[3]),
                    '730_active': bool(channel_states[4]),
                    '405_power': float(channel_powers[0]),
                    '488_power': float(channel_powers[1]),
                    '561_power': float(channel_powers[2]),
                    '635_power': float(channel_powers[3]),
                    '730_power': float(channel_powers[4]),
                    'active_channel_indices': active_channel_indices,
                    'exposure_ms': exposure_ms}

    return df_MM_setup, active_channel_indices, scan_axis_start_pos_mm, scan_axis_end_pos_mm, height_axis_start_pos_um, tile_axis_pos_um