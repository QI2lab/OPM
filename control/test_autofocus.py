#!/usr/bin/env python

from hardware.ArduinoShutter import ArduinoShutter
import instruments as ik
import instruments.units as u
from pycromanager import Bridge
from utils.autofocus_remote_unit import manage_O3_focus
import gc
import easygui

# test autofocus control

# connect to alignment laser shutter
shutter_com_port = 'COM7'
shutter_parameters = {'arduino_com_port': shutter_com_port,
                        'verbose': False}
shutter_controller = ArduinoShutter(shutter_parameters)

# connect to piezo controller
# controller must be setup to have a virtual com port. Might need to follow 
piezo_controller = ik.thorlabs.APTPiezoInertiaActuator.open_serial('COM8', baud=115200)
piezo_channel = piezo_controller.channel[0]
piezo_channel.enabled_single = True
max_volts = u.Quantity(110, u.V)
step_rate = u.Quantity(1000, 1/u.s)
acceleration = u.Quantity(10000, 1/u.s**2)
piezo_channel.drive_op_parameters = [max_volts, step_rate, acceleration]


shutter_controller.openShutter()
setup_done = False
while not(setup_done):
    setup_done = easygui.ynbox('Done aligning?', 'Title', ('Yes', 'No'))
shutter_controller.closeShutter()

roi_alignment = [1130,1130,32,32]
roi_experiment = [252,896,1800,512]

bridge = Bridge()
core = bridge.get_core()

reference_image, best_focus_positions = manage_O3_focus(core,roi_alignment,shutter_controller,piezo_channel,initialize=True,reference_image=None)
reference_image, best_focus_positions = manage_O3_focus(core,roi_alignment,shutter_controller,piezo_channel,initialize=False,reference_image=reference_image)


shutter_controller.openShutter()
setup_done = False
while not(setup_done):
    setup_done = easygui.ynbox('Done checking alignment?', 'Title', ('Yes', 'No'))
shutter_controller.closeShutter()

shutter_controller.close()

del piezo_controller, core
bridge.close()
gc.collect()