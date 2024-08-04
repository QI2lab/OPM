"""
Control class for Iterative FISH fluidics using ElveFlow OB1 + (2)MUX
Use pycrofluidics to communicate with Elveflow SDK

07/2024 Steven Sheppard
"""

import time
import pycrofluidics as pf
from pathlib import Path
import json

# params
dll_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64\DLL64"
sdk_path = r"C:\Users\qi2lab\Documents\github\ob1flow\hardware\SDK_V3_08_04\DLL\Python\Python_64"
config_path = r"C:\Users\qi2lab\Documents\github\ob1flow\elveflow_config.json"
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
                 config_path: str = r"C:\Users\qi2lab\Documents\github\ob1flow\elveflow_config.json",
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
            return None
        else:
            with open(self.config_path) as config:
                self.config = json.load(config)


    def save_config(self):
        """Saves the configuration to a JSON file."""
        # Set up config for saving
        self.config[f"mux1"] = dict(sorted(self.config[f"mux1"].items(), key=lambda item: item[1]))
        self.config[f"mux2"] = dict(sorted(self.config[f"mux2"].items(), key=lambda item: item[1]))
        try:
            with open(self.config_path, 'w') as _f:
                json.dump(self.config, _f, indent=4)
        except Exception as e:
            print(f"Error: Could not save to {self.config_path}. {e}")


    def startup(self,
                         start_remote: bool = True):
        """
        Initialize connections and return devices
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
                               sensorType=4,
                               resolution=7,
                               sensorDig=0,
                               sensorIPACalib=0)
            if start_remote:
                self.ob1.startRemote()

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
            if self.ob1.insideRemote:
                # Check if PID loop is running
                if self.ob1.confPIDs[0]:
                    self.ob1.remotePausePID(1)
                # Set the pressure to 0
                self.ob1.remoteSetTarget(1, 0)
        else:
            self.ob1.setPressure(1, 0)

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
        Configure MUX1 and MUX2 valves

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
            print(f"Canged MUX valves from: {_m1_0}/{_m2_0} to {_m1_1}/{_m2_1}")

        return key1, key2

    def start_flow(self,
                   rate: float = 500.0,
                   p: float = None,
                   i: float = None,
                   verbose: bool = False):
        """
        Start remote operated PID controlled flow at specified rate.
        If no runtime is given, flow will run until stop_flow command is recieved.

        :param float rate: Volumetric rate uL/min
        :param float p:
        :param float i:
        :param bool verbose:
        """
        # Check for new PID parameters
        if p:
            self.pid_p = p
        if i:
            self.pid_i = i

        #----------------------------------------------------------#
        # Setup and start PID loop
        if not self.ob1.insideRemote:
            self.ob1.startRemote()

        # Add the PID parameters
        self.ob1.remoteAddPID(channelP=1,
                              channelS=1,
                              P=self.pid_p,
                              I=self.pid_i,
                              run=False)

        # Start fluid push using rate
        self.ob1.remoteStartPID(1)
        self.ob1.remoteSetTarget(1, target=rate)

        if verbose:
            print(f"Flow started, rate={rate}")


    def stop_flow(self,
                  verbose: bool = False):
        # Set the valves to closed positions
        self.set_valves("off")

        # if using remote operation or flow control
        if self.ob1.insideRemote:
            if self.ob1.confPIDs[0]:
                self.ob1.remotePausePID(1)

            # Set the pressure target to 0
            self.ob1.remoteSetTarget(1, 0)

        # if using pressure control
        else:
            self.ob1.setPressure(1, 0)

        if verbose:
            print("Flow stopped, pressure set to 0!")


    def run_pid_loop(self,
                     rate: float = None,
                     volume: float = None,
                     runtime: float = None,
                     wait: float = None,
                     p: float = None,
                     i: float = None,
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
        # Check for new PID parameters
        if p:
            self.pid_p = p
        if i:
            self.pid_i = i

        #----------------------------------------------------------#
        # Setup PID loop parameters
        # Set volume based on given arg or runtime, if none, don't run
        if runtime and volume:
            print(f"volume and runtime args given, using given volume: {volume}uL")
        elif not volume and runtime:
            volume = rate * (runtime/60)
        elif not volume and not runtime:
            print("Missing volume OR runtime")
            rate = 0
            volume = -1

        if not wait:
            wait = 0

        if volume > 0:
            #------------------------------------------------------#
            # Start remote operated flow for given volume
            self.start_flow(rate, verbose=verbose)
            total_volume = 0
            elapsed_time = 0
            t_start = time.time()
            dt = 0
            while total_volume < volume:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.3f} / Pressure = {_p:.3f} Volume = {total_volume:.3f} / dt = {dt:.4f}", end="")

            if verbose:
                print(f"\nTotal time:{elapsed_time:.5f} seconds",
                      f"\nTotal volume:{total_volume:.5f} uL")

            #------------------------------------------------------#
            # Stop flow and restore device settings
            if not reset:
                self.ob1.remotePausePID(1)
                self.ob1.remoteSetTarget(1, 0)

                print("Pressure set to 0!")
            else:
                self.stop_flow(verbose)

        if verbose:
            print(f"Waiting {wait:.1f}sec!")
        time.sleep(wait)

        return total_volume, elapsed_time


    def run_at_pressure(self,
                        pressure: float = 1000,
                        volume: float = None,
                        runtime: float = None,
                        reset: bool = False,
                        verbose: bool = None):
        """
        Start flow using pressure operation.

        Optionally set the volume or runtime, if none are supplied,
        run indefinetly(not recognmended).

        :param float volume: mL of mux2_key to push (optional).
        :param float runtime: seconds to run flow (optional).
        """

        #----------------------------------------------------------#
        # Setup PID loop parameters
        # Set volume based on given arg or runtime, if none, don't run
        if not volume and not runtime:
            print("Missing volume OR runtime")
            pressure = 0
            runtime = 0

        if not self.ob1.insideRemote:
            self.ob1.startRemote()

        if self.ob1.confPIDs[0]:
            self.ob1.remotePausePID(1)
        self.ob1.remoteSetTarget(1, pressure)

        # Run loop calculating volume and tracking time
        total_volume = 0
        elapsed_time = 0
        t_start = time.time()
        if not volume:
            while elapsed_time < runtime:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.3f} / Pressure = {_p:.3f} Volume = {total_volume:.3f} / dt = {dt:.4f}", end="")

        else:
            while total_volume < volume:
                time.sleep(0.100)
                _f = self.ob1.getFlowUniversal(1)
                _p = self.ob1.getPressureUniversal(1)
                dt = (time.time() - (t_start + elapsed_time))
                elapsed_time += dt
                total_volume += dt * _f / 60
                if verbose:
                    print(f"\rFlow rate = {_f:.3f} / Pressure = {_p:.3f} Volume = {total_volume:.3f} / dt = {dt:.4f}", end="")

        if verbose:
            print(f"\nTotal time:{elapsed_time:.5f} seconds",
                  f"\nTotal volume:{total_volume:.5f} uL")

        # restore 0 pressure state
        if self.ob1.insideRemote:
            self.ob1.remoteSetTarget(1, 0)
        else:
            self.ob1.setPressure(1, 0)
        if reset:
            # Move MUX valves to "off"
            self.set_valves("off")
        if verbose:
            print("Pressure set to 0!")


    def run_pid(self,
                source: str = None,
                rate: float = None,
                volume: float = None,
                wait: float = None,
                verbose=False):
        if not source:
            source = self.config["sources"]["default"]
        if not rate:
            rate = self.config["rates"]["default"]
        if not volume:
            volume = self.config["volumes"]["mux2_to_sample"]
        if not wait:
            wait = 0

        #----------------------------------------------------------#
        # change MUX channels
        self.set_valves(source,
                        verbose=verbose)
        time.sleep(1)

        #----------------------------------------------------------#
        # Run flush PID controlled flow
        self.run_pid_loop(rate=rate,
                          volume=volume,
                          reset=True,
                          wait=wait,
                          verbose=verbose)

        if verbose:
            print(f"ElveFlow pushed: {volume:.3f}uL of {source} at {rate:.3}uL/min!")


    def run_system_prime(self,
                         verbose: bool = False):
        """
        Prime tubing and follow with flush of wash buffer
        """
        # First prime MUX1 sources
        self.mux2.set_valve(self.config["mux2"]["off"])
        for _k, _v in self.config["mux1"].items():
            if _k != "off":
                print(f"priming MUX1: {_v}")
                self.mux1.set_valve(_v)
                time.sleep(1)
                self.run_pid_loop(rate=self.config["rates"]["prime"],
                                  volume=self.config["volumes"]["prime_mux1"],
                                  verbose=verbose,
                                  reset=True)
                time.sleep(1)

        # Prime MUX2 sources
        self.mux1.set_valve(self.config["mux1"]["off"])
        for _k, _v in self.config["mux2"].items():
            if _k != "off":
                print(f"priming MUX2: {_v}")
                self.mux2.set_valve(_v)
                self.run_pid_loop(self.config["rates"]["prime"],
                                volume=self.config["volumes"]["prime_mux2"],
                                verbose=verbose,
                                reset=True)
                time.sleep(1)

        # Flush
        self.run_flush()

        if verbose:
            print(f"System primed!")


    def run_flush(self,
                  source: str = None,
                  rate: float = None,
                  volume: float = None,
                  wait: float = None,
                  verbose=False):
        if not source:
            source = self.config["sources"]["rinse_buffer"]
        if not rate:
            rate = self.config["rates"]["ssc_wash"]
        if not volume:
            volume = self.config["volumes"]["ssc_wash"]
        if not wait:
            wait = 0

        #----------------------------------------------------------#
        # change MUX channels
        self.set_valves(source,
                        verbose=verbose)
        time.sleep(1)

        #----------------------------------------------------------#
        # Run flush PID controlled flow
        total_volume, elapsed_time = self.run_pid(source=source,
                                                  rate=rate,
                                                  volume=volume,
                                                  reset=True,
                                                  wait=wait,
                                                  verbose=verbose)
        if total_volume>volume:
            flush_success=True
        else:
            flush_success=False
        if verbose:
            print(f"Flushed {volume:.3f}uL of {source} at {rate:.3}uL/min!")

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
            prime_buffer = "SSC"
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
        volume_to_sample += self.config["volumes"]["mux2_to_sample"]

        # if the total volume requested is larger than the volume to sample then push
        # the extra volume into the sample chamber at prime rate
        if volume>volume_to_sample:
            prime_buffer_volume = 0
        else:
            prime_buffer_volume = volume_to_sample - volume

        # Push volume of source at prime rate
        self.run_pid(source=source,
                     rate=rate,
                     volume=volume,
                     reset=True,
                     verbose=verbose)

        if prime_buffer_volume>0:
            # Push remaining volume to sample chamber using prime_buffer at prime rate
            self.run_pid(source=prime_buffer,
                        rate=rate,
                        volume=prime_buffer_volume,
                        reset=True,
                        verbose=verbose)
        if verbose:
            print(f"Primed {volume:.3f}uL of {source} to sample!\n",
                  f"Waiting {wait:.1f}sec!")

        time.sleep(wait)

        return True
