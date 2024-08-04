from fluidics_control import FlowControl

# params
dll_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64\DLL64"
sdk_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64"
config_path = r"C:\Users\qi2lab\Documents\github\ob1flow\elveflow_config.json"
ob1_name = "ASRL8::INSTR"
mux1_name = "ASRL5::INSTR"
mux2_name = "ASRL6::INSTR"

fl = FlowControl(dll_path,
                 sdk_path,
                 config_path,
                 ob1_name,
                 mux1_name,
                 mux2_name)

fl.startup()



fl.run_system_prime(verbose=True)
fl.run_system_flush(verbose=True)
fl.shutdown()
