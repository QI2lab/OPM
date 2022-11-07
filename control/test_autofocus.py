from pycromanager import Core
from hardware.PicardShutter import PicardShutter
from utils.autofocus_remote_unit import manage_O3_focus

# O3 piezo stage name
O3_stage_name = 'MCL NanoDrive Z Stage'

# Connect to MM via pycromanager
core = Core()

# create PicardShutter
shutter_controller = PicardShutter(shutter_id=712,verbose=False)

# run autofocus routine (make sure O3 alignment laser is on!)
updated_O3_stage_pos = manage_O3_focus(core,shutter_controller,O3_stage_name,verbose=False)

print(f'Updated O3 stage position: {updated_O3_stage_pos}')