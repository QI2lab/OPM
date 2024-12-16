"""
Control class for Iterative FISH fluidics using ElveFlow OB1 + (2)MUX
Use pycrofluidics to communicate with Elveflow SDK

07/2024 Steven Sheppard
"""

import time
import pycrofluidics as pf
from pathlib import Path
import json
import sys

# params
dll_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64\DLL64"
sdk_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64"
# config_path = r"C:\Users\qi2lab\Documents\github\ob1flow\elveflow_config.json"
config_path = "/home/steven/Documents/qi2lab/github/OPM/control/hardware/ob1flow/elveflow_config.json"
ob1_name = "ASRL8::INSTR"
mux1_name = "ASRL5::INSTR"
mux2_name = "ASRL6::INSTR"

class FlowControl():
    """
    Control class for ElveFlow OB1 + (2) MUX
    """
    def __init__(self,
                 elveflowDLL: str = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64\DLL64",
                 elveflowSDK: str = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64",
                 config_path: str = "/home/steven/Documents/qi2lab/github/OPM/control/hardware/ob1flow/elveflow_config.json",
                 ob1_name: str = "ASRL8::INSTR",
                 mux1_name: str = "ASRL5::INSTR",
                 mux2_name: str = "ASRL6::INSTR"):

        # TODO: load path and names from config file

        self.dll = elveflowDLL
        self.sdk = elveflowSDK
        self.ob_name = ob1_name
        self.mux1_name = mux1_name
        self.mux2_name = mux2_name
        self.ob1_callibrated = False
        self.config_path=Path(config_path)
        self.ob1 = None
        self.mux1 = None
        self.mux2 = None
        self.load_config()
        self.pid_p = self.config["PID"]["p"]
        self.pid_i = self.config["PID"]["i"]
        self.current_valves = ["off", "off"]


    def load_config(self):
        """
        Load configuration as dictionary
        """
        if not self.config_path.exists():
            print("configuration not found!")
            print(self.config_path)
            print(self.config_path.exists())
            # sys.exit()
            return None
        else:
            with open(self.config_path) as config:
                self.config = json.load(config)


    def save_config(self, verbose: bool = False):
        """Saves the configuration to a JSON file."""
        # Set up config for saving
        self.config[f"mux1"] = dict(sorted(self.config[f"mux1"].items(), key=lambda item: item[1]))
        self.config[f"mux2"] = dict(sorted(self.config[f"mux2"].items(), key=lambda item: item[1]))

        # verify there exist an "off" valve or force v12 to be off by default, if not code breaks
        mux1_keys = [_ for _ in self.config["mux1"].keys()]
        mux2_keys = [_ for _ in self.config["mux2"].keys()]
        if not "off" in mux1_keys:
            key_to_pop = None
            for key, val in self.fc.config["mux1"].items():
                if val==12:
                    key_to_pop=key
            if key_to_pop:
                self.config["mux1"].pop(key_to_pop)
            self.config["mux1"]["off"] = 12
        if not "off" in mux2_keys:
            key_to_pop = None
            for key, val in self.fc.config["mux2"].items():
                if val==12:
                    key_to_pop=key
            if key_to_pop:
                self.config["mux2"].pop(key_to_pop)
            self.config["mux2"]["off"] = 12
        # Enforce the MUX2 to MUX1 line at valve 11 on mux1
        if not "mux2" in mux1_keys:
            self.config["mux1"]["mux2"] = 11

        # Overwrite json with config
        try:
            with open(self.config_path, 'w') as _f:
                json.dump(self.config, _f, indent=4)
        except Exception as e:
            print(f"Error: Could not save to {self.config_path}. {e}")
        if verbose:
            print("Config file saved!")


    def startup(self):
        """
        Initialize connections and return devices.
        Operate in remote mode.
        """
        if not self.ob1:
            self.ob1 = pf.OB1elve(elveflowDLL=dll_path,
                                  elveflowSDK=sdk_path,
                                  deviceName=ob1_name)

            # open connections
            self.ob1.open()

            try:
                self.ob1.loadCallibration()
            except:
                self.ob1.performCallibration()
                self.ob1_callibrated = True

            # Initiallize digital flow sensor
            self.ob1.addSensor(channel=1,
                               sensorType=6,
                               resolution=3,
                               sensorDig=0)

            # start remote operation
            self.ob1.startRemote()

            # Add the PID parameters
            self.ob1.remoteAddPID(channelP=1,
                                  channelS=1,
                                  P=self.pid_p,
                                  I=self.pid_i,
                                  run=False)

        # Open MUX distribution control valve connections
        # Set current valve to "off".
        if not self.mux1:
            self.mux1 = pf.MUXelve(elveflowDLL=dll_path,
                                   elveflowSDK=sdk_path,
                                   deviceName=mux1_name,
                                   deviceID=1)
            self.mux1.open()
            self.mux1.set_valve(self.config["mux1"]["off"])
        if not self.mux2:
            self.mux2 = pf.MUXelve(elveflowDLL=dll_path,
                                   elveflowSDK=sdk_path,
                                   deviceName=mux2_name,
                                   deviceID=2)
            self.mux2.open()
            self.mux2.set_valve(self.config["mux2"]["off"])


    def reset(self):
        """
        Reset controller and MUX (1+2) to startup settings
        """
        # Reset OB1 to startup state
        if self.ob1:
            # Check if PID loop is running
            if self.ob1.runningPIDs[0]:
                self.ob1.remotePausePID(1)
            # Set the pressure to 0
            self.ob1.remoteSetTarget(1, 0)

        # Move MUX valves to "off"
        if self.mux1:
            self.mux1.set_valve(self.config["mux1"]["off"])
        if self.mux2:
            self.mux2.set_valve(self.config["mux2"]["off"])


    def shutdown(self):
        """
        Close device connections
        """
        if self.ob1:
            self.stop_flow()
            self.ob1.close()
            self.ob1 = None
        if self.mux1:
            self.mux1.set_valve(self.config["mux1"]["off"])
            self.mux1.close()
            self.mux1 = None
        if self.mux2:
            self.mux2.set_valve(self.config["mux2"]["off"])
            self.mux2.close()
            self.mux2 = None


    def set_valves(self,
                   key: str = None,
                   verbose: bool = False):
        """
        Configure MUX1 and MUX2 valves using source keys.

        :param str key: dictionary key matching mux mapping dictionary
        """

        if verbose:
            _m1_0 = self.mux1.get_valve()
            _m2_0 = self.mux2.get_valve()

        if key=="off":
            # Change both MUX devices to NO flow states
            key1 = "off"
            key2 = "off"
        elif key in [k for k in self.config["mux1"].keys()]:
            # Flow through MUX1 only
            key2="off"
            key1 = key
        elif key in [k for k in self.config["mux2"].keys()]:
            # Flow through MUX2 and through MUX1
            key1 = "mux2"
            key2 = key
        else:
            key1=None
            key2=None
            print("No valid keys given, valves not changed.")

        if key1 and key2:
            self.mux1.set_valve(self.config["mux1"][key1])
            self.mux2.set_valve(self.config["mux2"][key2])
            self.current_valves = [key1, key2]

        if verbose:
            _m1_1 = self.mux1.get_valve()
            _m2_1 = self.mux2.get_valve()
            print(f"Changed MUX valves from: {_m1_0}/{_m2_0} to {_m1_1}/{_m2_1}")

        return key1, key2


    def start_flow(self,
                   rate: float = 500.0,
                   verbose: bool = False):
        """
        Start remote operated PID controlled flow at specified rate.
        If no runtime is given, flow will run until stop_flow command is recieved.

        :param float rate: Volumetric rate uL/min
        :param float p:
        :param float i:
        :param bool verbose:
        """
        # if PID loop is not running, start it.
        if not self.ob1.runningPIDs[0]:
            self.ob1.remoteStartPID(1)

        # Start fluid push using rate
        self.ob1.remoteSetTarget(1, target=rate)

        if verbose:
            print(f"PID control flow started, rate={rate:.2f}uL/min")


    def stop_flow(self,
                  verbose: bool = False):
        """
        Stop PID controlled flow and set the valves to off position.
        """
        # Pause the PID control
        if self.ob1.runningPIDs[0]:
            self.ob1.remotePausePID(1)

        # Set the target pressure to 0
        self.ob1.remoteSetTarget(1, 0)

        if verbose:
            print("PID control off and pressure set to 0!")


    def start_pressure(self,
                       pressure: float = 1000,
                       verbose: bool = None):
        """
        Start flow using pressure operation
        """
        # if running in PID loop, pause to control pressure
        if self.ob1.runningPIDs[0]:
            self.ob1.remotePausePID(1)

        # Set target pressue
        self.ob1.remoteSetTarget(1, pressure)

        if verbose:
            print(f"Pressure control flow started, pressure={pressure:.2f}")


    def stop_pressure(self,
                      verbose: bool = False):
        """
        Set the pressure to 0
        """
        # Pause the PID control
        if self.ob1.runningPIDs[0]:
            self.ob1.remotePausePID(1)

        # Set the target pressure to 0
        self.ob1.remoteSetTarget(1, 0)

        if verbose:
            print(f"Pressure control flow stopped!")


    def run_pid_loop(self,
                     source: str = None,
                     rate: float = None,
                     volume: float = None,
                     runtime: float = None,
                     wait: float = None,
                     reset: bool = False,
                     verbose: bool = False):
        """
        Run PID over to infuse a fixed volume at specific flow rate.

        Must specify either volume or runtime.

        :param str source: MUX valve key to push from.
        :param float rate: mL/min flow rate to maintain.
        :param float volume: mL of mux2_key to push (optional).
        :param float runtime: seconds to run flow (optional).
        :param float p: TODO
        :param float i: TODO
        """
        if runtime and volume:
            print(f"volume and runtime args given, defaulting to volume: {volume}uL")
            runtime=None
        if not source:
            if self.current_valves[0]=="off":
                volume = None
                runtime = 0
        else:
            self.set_valves(source, verbose=verbose)
            print("setting valve in PID loop")

        if not rate:
            rate = self.config["rates"]["default"]

        if not wait:
            wait = 0
        print(source, rate, volume)
        #------------------------------------------------------#
        # Start remote operated flow for given volume
        self.start_flow(rate, verbose=verbose)
        total_volume = 0
        elapsed_time = 0
        t_start = time.time()
        dt = 0
        if volume is not None:
            while total_volume < volume:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.2f} uL/min ; Pressure = {_p:.2f} ; Volume = {total_volume:.2f} uL ; dt = {dt:.3f} seconds ; elapsed time = {elapsed_time:.2f} seconds", end="")

        elif runtime is not None:
            while elapsed_time < runtime:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.2f} uL/min ; Pressure = {_p:.2f} ; Volume = {total_volume:.2f} uL ; dt = {dt:.3f} seconds ; elapsed time = {elapsed_time:.2f} seconds", end="")
        else:
            # no run occured
            total_volume = 0
            elapsed_time = 0

        self.stop_flow(verbose=verbose)

        if reset:
            self.set_valves('off', verbose=True)

        if verbose:
            print(f"\nTotal time:{elapsed_time:.5f} seconds",
                  f"\nTotal volume:{total_volume:.5f} uL")

        if verbose:
            print(f"Waiting {wait:.1f}sec!")
        time.sleep(wait)

        return total_volume, elapsed_time


    def run_pressure_loop(self,
                          source: str = None,
                          pressure: float = None,
                          volume: float = None,
                          runtime: float = None,
                          wait: float = None,
                          reset: bool = False,
                          verbose: bool = False):
        """
        Run PID over to infuse a fixed volume at specific flow rate.

        Must specify either volume or runtime.

        :param float rate: mL/min flow rate to maintain.
        :param float volume: mL of mux2_key to push (optional).
        :param float runtime: seconds to run flow (optional).
        :param float p: TODO
        :param float i: TODO
        """
        if not source:
            source = "off"
            volume = None
            runtime = 0
        if volume and runtime:
            print("Both volume and runtime given, resorting to volume.")
            runtime = None
        if not wait:
            wait = 0

        #------------------------------------------------------#
        # Start remote operated flow for given volume
        self.start_pressure(pressure, verbose=verbose)
        total_volume = 0
        elapsed_time = 0
        t_start = time.time()
        dt = 0
        if volume is not None:
            while total_volume < volume:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.2f} uL/min ; Pressure = {_p:.2f} ; Volume = {total_volume:.2f} uL ; dt = {dt:.3f} seconds ; elapsed time = {elapsed_time:.2f} seconds", end="")
        elif runtime is not None:
            while elapsed_time < runtime:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.2f} uL/min ; Pressure = {_p:.2f} ; Volume = {total_volume:.2f} uL ; dt = {dt:.3f} seconds ; elapsed time = {elapsed_time:.2f} seconds", end="")
        else:
            # no run occured
            total_volume = 0
            elapsed_time = 0

        self.stop_pressure(verbose=verbose)

        if reset:
            self.set_valves("off", verbose=True)

        if verbose:
            print(f"\nTotal time:{elapsed_time:.5f} seconds",
                  f"\nTotal volume:{total_volume:.5f} uL")

        if verbose:
            print(f"Waiting {wait:.1f}sec!")
        time.sleep(wait)

        return total_volume, elapsed_time


    def run_system_prime(self,
                         verbose: bool = False):
        """
        Prime tubing and follow with flush of wash buffer
        """
        # First prime MUX1 sources
        # self.mux2.set_valve(self.config["mux2"]["off"])
        for _k, _v in self.config["mux1"].items():
            if not (_k=="off" or _k=="mux2" or _k=="none" or _k==""):
                print(f"priming MUX1: {_k}={_v}")
                self.run_pid_loop(source=_k,
                                  rate=self.config["rates"]["prime"],
                                  volume=self.config["volumes"]["prime_mux1"],
                                  verbose=verbose,
                                  reset=True)
                time.sleep(2)

        # Prime MUX2 sources
        # self.mux1.set_valve(self.config["mux1"]["mux2"])
        for _k, _v in self.config["mux2"].items():
            if not (_k=="off" or _k=="none" or _k==""):
                print(f"priming MUX2: {_k}={_v}")
                self.run_pid_loop(source=_k,
                                  rate=self.config["rates"]["prime"],
                                  volume=self.config["volumes"]["prime_mux2"],
                                  verbose=verbose,
                                  reset=True)
                time.sleep(2)

        print("Source prime complete, flushing line")
        self.run_flush(verbose=verbose)

        if verbose:
            print(f"System primed!")

        return True


    def run_flush(self,
                  source: str = None,
                  rate: float = None,
                  volume: float = None,
                  wait: float = None,
                  verbose=False):
        if not source:
            source = self.config["sources"]["flush_media"]
        if not rate:
            rate = self.config["rates"]["flush"]
        if not volume:
            volume = self.config["volumes"]["clear_sample_chamber"]
        if not wait:
            wait = 0
        #----------------------------------------------------------#
        # Run flush PID controlled flow
        print(source, rate, volume, wait, verbose)
        total_volume, elapsed_time = self.run_pid_loop(source=source,
                                                       rate=rate,
                                                       volume=volume,
                                                       wait=wait,
                                                       reset=True,
                                                       verbose=verbose)
        time.sleep(2)
        if total_volume>volume:
            flush_success=True
        else:
            flush_success=False
        if verbose:
            print(f"Flush success!")

        return flush_success


    def run_prime_source(self,
                         source: str = None,
                         prime_buffer: str = None,
                         rate: float = None,
                         volume: float = None,
                         wait: float = None,
                         verbose: bool = False):
        """
        Deliver specified volume of source to the sample at the "prime" rate
        - Assumes the system is already primed
        - Assumes the prime_buffer originates from MUX2

        """
        if not source:
            source = "off"
        if not prime_buffer:
            prime_buffer = self.config["sources"]["prime_media"]
        if not rate:
            rate = self.config["rate"]["prime"]
        if not volume:
            volume = self.config["volumes"]["mux2_to_sample"]
        if not wait:
            wait = 0

        # Calculate the volume from source to the sample chamber
        volume_to_sample = 0

        # if source is on MUX2, then we have to prime MUX2->MUX1
        if self.current_valves[0]=="mux2":
            volume_to_sample += self.config["volumes"]["mux2_to_mux1"]

        # add volume from MUX1 to the sample chamber
        volume_to_sample += self.config["volumes"]["mux1_to_sample"]

        # if the total volume requested is larger than the volume to sample then push
        # the extra volume into the sample chamber at prime rate
        if volume>volume_to_sample:
            prime_buffer_volume = 0
        else:
            prime_buffer_volume = volume_to_sample - volume

        # Push volume of source at prime rate
        self.run_pid_loop(source=source,
                          rate=rate,
                          volume=volume,
                          reset=True,
                          verbose=verbose)

        if prime_buffer_volume>0:
            # Push remaining volume to sample chamber using prime_buffer at prime rate
            self.run_pid_loop(source=prime_buffer,
                              rate=rate,
                              volume=prime_buffer_volume,
                              reset=True,
                              verbose=verbose)
        if verbose:
            print(f"Primed {volume:.3f}uL of {source} to sample!\n",
                  f"Waiting {wait:.1f}sec!")

        time.sleep(wait)

        return True