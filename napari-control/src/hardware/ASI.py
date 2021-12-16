
#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
OPM ASI Tiger functions
----------------------------------------------------------------------------------------
Douglas Shepherd
12/11/2021
douglas.shepherd@asu.edu
----------------------------------------------------------------------------------------
'''

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
from pymmcore_plus import RemoteMMCore
import time

def check_if_busy(mmcore_stage):
    '''
    Check if ASI Tiger controller is busy executing a command

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :return None:
    '''

    # turn on 'transmit repeated commands' for Tiger
    mmcore_stage.setProperty('TigerCommHub','OnlySendSerialCommandOnChange','No')

    # check to make sure Tiger is not busy
    ready='B'
    while(ready!='N'):
        command = 'STATUS'
        mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)
        ready = mmcore_stage.getProperty('TigerCommHub','SerialResponse')
        time.sleep(.010)

    # turn off 'transmit repeated commands' for Tiger
    mmcore_stage.setProperty('TigerCommHub','OnlySendSerialCommandOnChange','Yes')

def set_joystick_mode(mmcore_stage,x_stage_name,z_stage_name,joystick_mode):
    '''
    Turn ASI Tiger joystick input on or off 

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :param x_stage_name: str
        name of xy stage in MM config
    :param z_stage_name: str
        name of z stage in MM config
    :param joystick_mode: bool
        joystick input state
    :return None:
    '''
    if joystick_mode:
        mmcore_stage.setProperty(x_stage_name,'JoystickEnabled','Yes')
        mmcore_stage.setProperty(z_stage_name,'JoystickInput','22 - right wheel')
    else:
        mmcore_stage.setProperty(x_stage_name,'JoystickEnabled','No')
        mmcore_stage.setProperty(z_stage_name,'JoystickInput','0 - none')

def set_axis_speed(mmcore_stage,axis,axis_speed):
    '''
    Change ASI Tiger X/Y axis movement speed

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :param axis: str
        name of axis ('X' or 'Y')
    :param axis_speed: float
        speed in mm/s
    :return None:
    '''

    if axis == 'X':
        command = 'SPEED X='+str(axis_speed)
        mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)
    elif axis == 'Y':
        command = 'SPEED Y='+str(axis_speed)
        mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)

def set_xy_position(mmcore_stage,stage_x,stage_y):
    '''
    Set ASI Tiger XY stage position

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :param stage_x_um: float
        x axis position in um
    :param stage_y_um: float
        x axis position in um
    :return None:
    '''

    mmcore_stage.setXYPosition(stage_x,stage_y)
    mmcore_stage.waitForDevice(mmcore_stage.getXYStageDevice())

def set_z_position(mmcore_stage,stage_z):
    '''
    Set ASI Tiger Z stage position

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :param stage_z_um: float
        z axis position in um
    :return None:
    '''
    mmcore_stage.setZPosition(stage_z)
    mmcore_stage.waitForDevice(mmcore_stage.getFocusDevice())

def set_1d_stage_scan(mmcore_stage):
    '''
    Setup ASI Tiger for constant speed stage scan on X axis

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :return None:
    '''
    command = '1SCAN X? Y=0 Z=9 F=0'
    mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)

def set_1d_stage_scan_area(mmcore_stage,scan_axis_start_mm,scan_axis_end_mm):
    '''
    Setup ASI Tiger limits for X axis constant speed scan

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :param scan_axis_start_mm: float
        z axis position in mm
    :param scan_axis_end_mm: float
        z axis position in mm
    :return None:
    '''
    scan_axis_start_mm = scan_axis_start_mm
    scan_axis_end_mm = scan_axis_end_mm
    command = '1SCANR X='+str(scan_axis_start_mm)+' Y='+str(scan_axis_end_mm)+' R=10'
    mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)

def setup_start_trigger_output(mmcore_stage):
    '''
    Setup ASI Tiger trigger ouput on PLC add-on card

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :return None:
    '''

    plcName = 'PLogic:E:36'
    propPosition = 'PointerPosition'
    propCellConfig = 'EditCellConfig'
    addrOutputBNC1 = 33 # BNC1 on the PLC front panel
    addrStageSync = 46  # TTL5 on Tiger backplane = stage sync signal
    # connect stage sync signal to BNC output
    mmcore_stage.setProperty(plcName, propPosition, addrOutputBNC1)
    mmcore_stage.setProperty(plcName, propCellConfig, addrStageSync)

def start_1d_stage_scan(mmcore_stage):
    '''
    Send ASI Tiger "start" command for constant speed stage scan. 
    Should be called after acquisition sequence is started.

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :return None:
    '''
    command='1SCAN'
    mmcore_stage.setProperty('TigerCommHub','SerialCommand',command)

def get_xyz_position(mmcore_stage):
    '''
    Get ASI Tiger stage position

    :param mmcore_stage: RemoteMMCore
        handle to existing RemoteMMCore
    :return stage_x_um: float
        x stage position in micron
    :return stage_y_um: float
        y stage position in micron
    :return stage_z_um: float
        z stage position in micron
    '''
    xy_pos = mmcore_stage.getXYPosition()
    stage_x_um = xy_pos[0]
    stage_y_um = xy_pos[1]
    stage_z_um = mmcore_stage.getPosition()

    return stage_x_um,stage_y_um,stage_z_um