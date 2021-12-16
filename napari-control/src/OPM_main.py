#!/usr/bin/python
from magicgui import magicgui
from magicclass import magicclass, set_design
from pathlib import Path

@magicclass(labels=False)
@set_design(text="Stage monitor")
class OPMmain():

    def __init__(self):

        self.path_to_fluidics_flush_program = Path('C:/Users/qi2lab/Documents/GitHub/common_fluidics_programs/wash_fluidics_wells1-9.csv')
        self.path_to_mm_config = Path('C:/Program Files/Micro-Manager-2.0gamma/temp_HamDCAM.cfg')
        self.pump_COM_port = 'COM5'
        self.valve_COM_port = 'COM6'
        self.pump_parameters = {'pump_com_port': self.pump_COM_port,
                                'pump_ID': 30,
                                'verbose': True,
                                'simulate_pump': False,
                                'serial_verbose': False,
                                'flip_flow_direction': False}

    @magicgui(
        auto_call=False,
        opm_mode = {
            "widget_type": "Select", 
            "choices": [
                "Flush fluidics",
                "Run iterative experiment",
                "Run timelapse experiment",
                "Reconstruct iterative experiment",
                "Reconstruct timelapse experiment"], 
            "allow_multiple": False, 
            "label": "OPM mode"},
        call_button ='Select mode'
    )
    def select_mode(self,opm_mode):
        if opm_mode[0]=="Flush fluidics":
            # import fluidics libraries
            from src.hardware.APump import APump
            from src.hardware.HamiltonMVP import HamiltonMVP
            import src.utils.data_io as data_io 
            from src.utils.fluidics_control import run_fluidic_program

            # connect to pump
            pump_controller = APump(self.pump_parameters)
            # set pump to remote control
            pump_controller.enableRemoteControl(True)

            # connect to valves
            valve_controller = HamiltonMVP(com_port=self.valve_COM_port)
            # initialize valves
            valve_controller.autoAddress()

            # load fluidics flush program
            df_fluidics = data_io.read_fluidics_program(self.path_to_fluidics_flush_program)

            # run fluidics flush
            for r_idx in range(int(df_fluidics['round'].max())):             
                success_fluidics = run_fluidic_program(r_idx,df_fluidics,valve_controller,pump_controller)
                if not(success_fluidics):
                    raise Exception('Error in fluidics.')
            
        elif opm_mode[0]=="Run iterative experiment":
            import opm_iterative_control 
            opm_iterative_control.main(self.path_to_mm_config)
        elif opm_mode[0]=="Run timelapse experiment":
            self.close()
            import opm_timelapse_control
            opm_timelapse_control.main(self.path_to_mm_config)
        elif opm_mode[0]=="Reconstruct iterative experiment":
            #import 
            #opm_iterative_reconstruct()
            pass
        elif opm_mode[0]=="Reconstruct timelapse experiment":
            import opm_timelapse_reconstruction
            opm_timelapse_reconstruction.main()

def main():

    opmmain = OPMmain()
    opmmain.show(run=True)

if __name__ == "__main__":
    main()