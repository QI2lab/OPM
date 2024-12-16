from hardware.ElveFlow_fluidics import FlowControl
from utils.data_io import read_fluidics_program
# params
dll_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64\DLL64"
sdk_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64"
config_path = "/home/steven/Documents/qi2lab/github/OPM/control/hardware/ob1flow/elveflow_config.json"
ob1_name = "ASRL8::INSTR"
mux1_name = "ASRL5::INSTR"
mux2_name = "ASRL6::INSTR"

program_path = "/home/steven/Documents/qi2lab/github/OPM/control/test_fluidics_run.xlsx"
fl = FlowControl(elveflowDLL=dll_path,
                 elveflowSDK=sdk_path,
                 config_path=program_path,
                 ob1_name=ob1_name,
                 mux1_name=mux1_name,
                 mux2_name=mux2_name)


# load test program
program_list = read_fluidics_program(program_path=program_path)

 UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9d in position 54: invalid start byte


fl.startup()



fl.run_system_prime(verbose=True)
fl.run_system_flush(verbose=True)
fl.shutdown()
