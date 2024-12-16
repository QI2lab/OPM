"""
Fluidics control widget

TODO: Live mode does not work when running.

07/2024 Steven Sheppard
"""
import sys
import threading
import atexit
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QDoubleSpinBox, QPushButton,
                             QLabel, QCheckBox, QFormLayout,QTabWidget,
                             QLineEdit, QGroupBox, QApplication, QComboBox)
from PyQt5.QtCore import QTimer
from ElveFlow_fluidics import FlowControl  # Adjust the import as necessary

class FlowControlWidget(QWidget):
    def __init__(self):
        super().__init__()

        # initialize controls
        self.fc = FlowControl()
        self.fc.startup()

        self.pressure = 1000
        self.flowrate = self.fc.config["rates"]["default"]
        self.pid_i = self.fc.config["PID"]["i"]
        self.pid_p = self.fc.config["PID"]["p"]
        self.volume = 0
        self.runtime = 0
        self.run_pressure = False
        self.run_flowrate = True
        self.running = False
        self.run_live = False

        # Timer for live monitoring
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.update_monitor)

        # initialize UI
        self.create_ui()
        self.refresh_config_tabs()


    def create_ui(self):
        # define a main layout to place all groups/widgets
        self.setMinimumSize(2000, 650)
        self.main_layout = QGridLayout()

        # on the left have a tabbed section for setting up configurations
        self.config_group = QGroupBox("Configuration Settings")
        self.config_layout = QVBoxLayout()
        self.config_tabs = QTabWidget()
        # Create a tabs for setting the MUX valves
        self.mux1_group = QGroupBox("MUX1 Valve keys")
        self.mux2_group = QGroupBox("MUX2 Valve keys")
        # Create a form form layout for each set of valves
        self.mux1_form = QFormLayout()
        self.mux1_lines = []
        self.mux2_form = QFormLayout()
        self.mux2_lines = []
        # Create a tab for setting the PI D parameters
        self.pid_group = QGroupBox("PID control parameters")
        # Create a tabe for setting the default rates and volumes
        self.specs_group = QGroupBox("rates/volumes")
        self.specs_layout = QVBoxLayout()
        self.rates_group = QGroupBox("Default Rates")
        self.volumes_group = QGroupBox("Default Volumes")
        # Finish config group with buttons for refresh/updating file

        # in the middle have a panel for running specific programs
        self.testing_group = QGroupBox("Device Functions")
        self.testing_layout = QVBoxLayout()
        # Create a group for function arguements
        self.args_group = QGroupBox("Program settings")
        self.args_layout= QVBoxLayout()
        # Create group for the drop down menu and run button
        self.func_group = QGroupBox("")

        # on the right have a panel for device control and running
        self.control_group = QGroupBox("Device Control")
        self.control_layout = QVBoxLayout()
        # Create a group for setting parameters and an update button
        self.current_group = QGroupBox("")
        self.current_layout = QVBoxLayout()
        # Create a group for stop and start
        self.run_group = QGroupBox("")
        self.run_layout = QHBoxLayout()
        # Create a monitor group
        self.live_group = QGroupBox("")
        self.live_layout = QVBoxLayout()

        #---------------------------------------------------------#
        # Create MUX tabs and initialize with empty values
        n_valves = 12
        for ii in range(n_valves):
            valve = ii+1
            key1 = ""
            key2 = ""
            line1 = QLineEdit(key1)
            line2 = QLineEdit(key2)
            line1.textChanged.connect(lambda text, valve=valve: self.update_valves_config(dev="1", key=text, valve=valve))
            line2.textChanged.connect(lambda text, valve=valve: self.update_valves_config(dev="2", key=text, valve=valve))
            self.mux1_form.addRow(str(ii + 1), line1)
            self.mux2_form.addRow(str(ii + 1), line2)
            self.mux1_lines.append(line1)
            self.mux2_lines.append(line2)

        self.mux1_group.setLayout(self.mux1_form)
        self.mux2_group.setLayout(self.mux2_form)

        #---------------------------------------------------------#
        # Create tab for setting PID parameters
        self.pid_layout= QVBoxLayout()
        self.pid_p_layout = QHBoxLayout()
        self.pid_i_layout = QHBoxLayout()
        self.pid_p_label = QLabel("(P)roportional:")
        self.pid_p_spbx = QDoubleSpinBox()
        self.pid_p_spbx.setRange(0, 1)
        self.pid_p_spbx.setSingleStep(0.01)
        self.pid_p_spbx.setValue(self.fc.config["PID"]["p"])
        self.pid_p_spbx.valueChanged.connect(self.update_pid_config)
        self.pid_p_layout.addWidget(self.pid_p_label)
        self.pid_p_layout.addWidget(self.pid_p_spbx)

        self.pid_i_label = QLabel("(I)integral:")
        self.pid_i_spbx = QDoubleSpinBox()
        self.pid_i_spbx.setRange(0, 1)
        self.pid_i_spbx.setSingleStep(0.01)
        self.pid_i_spbx.setValue(self.fc.config["PID"]["i"])
        self.pid_i_spbx.valueChanged.connect(self.update_pid_config)
        self.pid_i_layout.addWidget(self.pid_i_label)
        self.pid_i_layout.addWidget(self.pid_i_spbx)

        self.pid_layout.addLayout(self.pid_p_layout)
        self.pid_layout.addLayout(self.pid_i_layout)
        self.pid_group.setLayout(self.pid_layout)

        #---------------------------------------------------------#
        # Create tab for setting rate/volumes parameters
        self.rates_layout = QVBoxLayout()
        self.volumes_layout = QVBoxLayout()

        self.r1_layout = QHBoxLayout()
        self.r1_label = QLabel("default")
        self.r1_spbx = QDoubleSpinBox()
        self.r1_spbx.setRange(0, 2000)
        self.r1_spbx.setSingleStep(1.0)
        self.r1_spbx.setValue(self.fc.config["rates"]["default"])
        self.r1_spbx.valueChanged.connect(self.update_specs_config)
        self.r1_layout.addWidget(self.r1_label)
        self.r1_layout.addWidget(self.r1_spbx)

        self.r2_layout = QHBoxLayout()
        self.r2_label = QLabel("prime")
        self.r2_spbx = QDoubleSpinBox()
        self.r2_spbx.setRange(0, 2000)
        self.r2_spbx.setSingleStep(1.0)
        self.r2_spbx.setValue(self.fc.config["rates"]["prime"])
        self.r2_spbx.valueChanged.connect(self.update_specs_config)
        self.r2_layout.addWidget(self.r2_label)
        self.r2_layout.addWidget(self.r2_spbx)

        self.r3_layout = QHBoxLayout()
        self.r3_label = QLabel("flush")
        self.r3_spbx = QDoubleSpinBox()
        self.r3_spbx.setRange(0, 2000)
        self.r3_spbx.setSingleStep(1.0)
        self.r3_spbx.setValue(self.fc.config["rates"]["flush"])
        self.r3_spbx.valueChanged.connect(self.update_specs_config)
        self.r3_layout.addWidget(self.r3_label)
        self.r3_layout.addWidget(self.r3_spbx)

        self.rates_layout.addLayout(self.r1_layout)
        self.rates_layout.addLayout(self.r2_layout)
        self.rates_layout.addLayout(self.r3_layout)
        self.rates_group.setLayout(self.rates_layout)

        self.v1_layout = QHBoxLayout()
        self.v1_label = QLabel("Prime Mux1")
        self.v1_spbx = QDoubleSpinBox()
        self.v1_spbx.setRange(0, 2000)
        self.v1_spbx.setSingleStep(10.0)
        self.v1_spbx.setValue(self.fc.config["volumes"]["prime_mux1"])
        self.v1_spbx.valueChanged.connect(self.update_specs_config)
        self.v1_layout.addWidget(self.v1_label)
        self.v1_layout.addWidget(self.v1_spbx)

        self.v2_layout = QHBoxLayout()
        self.v2_label = QLabel("Prime MUX2")
        self.v2_spbx = QDoubleSpinBox()
        self.v2_spbx.setRange(0, 2000)
        self.v2_spbx.setSingleStep(1.0)
        self.v2_spbx.setValue(self.fc.config["volumes"]["prime_mux2"])
        self.v2_spbx.valueChanged.connect(self.update_specs_config)
        self.v2_layout.addWidget(self.v2_label)
        self.v2_layout.addWidget(self.v2_spbx)

        self.v3_layout = QHBoxLayout()
        self.v3_label = QLabel("MUX2 to MUX1")
        self.v3_spbx = QDoubleSpinBox()
        self.v3_spbx.setRange(0, 2000)
        self.v3_spbx.setSingleStep(10.0)
        self.v3_spbx.setValue(self.fc.config["volumes"]["mux2_to_mux1"])
        self.v3_spbx.valueChanged.connect(self.update_specs_config)
        self.v3_layout.addWidget(self.v3_label)
        self.v3_layout.addWidget(self.v3_spbx)

        self.v4_layout = QHBoxLayout()
        self.v4_label = QLabel("MUX1 to sample")
        self.v4_spbx = QDoubleSpinBox()
        self.v4_spbx.setRange(0, 2000)
        self.v4_spbx.setSingleStep(10.0)
        self.v4_spbx.setValue(self.fc.config["volumes"]["mux1_to_sample"])
        self.v4_spbx.valueChanged.connect(self.update_specs_config)
        self.v4_layout.addWidget(self.v4_label)
        self.v4_layout.addWidget(self.v4_spbx)

        self.v5_layout = QHBoxLayout()
        self.v5_label = QLabel("Flush samle chamber")
        self.v5_spbx = QDoubleSpinBox()
        self.v5_spbx.setRange(0, 10000)
        self.v5_spbx.setSingleStep(100.0)
        self.v5_spbx.setValue(self.fc.config["volumes"]["clear_sample_chamber"])
        self.v5_spbx.valueChanged.connect(self.update_specs_config)
        self.v5_layout.addWidget(self.v5_label)
        self.v5_layout.addWidget(self.v5_spbx)

        self.volumes_layout.addLayout(self.v5_layout)
        self.volumes_layout.addLayout(self.v2_layout)
        self.volumes_layout.addLayout(self.v3_layout)
        self.volumes_layout.addLayout(self.v4_layout)
        self.volumes_layout.addLayout(self.v5_layout)
        self.volumes_group.setLayout(self.volumes_layout)

        self.specs_layout.addWidget(self.rates_group)
        self.specs_layout.addWidget(self.volumes_group)
        self.specs_group.setLayout(self.specs_layout)

        #---------------------------------------------------------#
        # update tab group
        self.config_tabs.addTab(self.mux1_group, "MUX1 Valves")
        self.config_tabs.addTab(self.mux2_group, "MUX2 Valves")
        self.config_tabs.addTab(self.pid_group, "PID")
        self.config_tabs.addTab(self.specs_group, "Rates/Volumes")

        #---------------------------------------------------------#
        # setup refresh/update file button and update group
        self.config_update_group = QGroupBox()
        self.config_bt_layout = QVBoxLayout()
        self.refresh_config_bt = QPushButton("Refresh from file")
        self.refresh_config_bt.clicked.connect(self.refresh_config_tabs)
        self.update_config_bt = QPushButton("Save configuration file")
        self.update_config_bt.clicked.connect(self.save_config)

        self.config_bt_layout.addWidget(self.refresh_config_bt)
        self.config_bt_layout.addWidget(self.update_config_bt)
        self.config_update_group.setLayout(self.config_bt_layout)

        #---------------------------------------------------------#
        # update config group
        self.config_layout.addWidget(self.config_tabs)
        self.config_layout.addWidget(self.config_update_group)
        self.config_group.setLayout(self.config_layout)

        #---------------------------------------------------------#
        # Define area to run a program
        self.fn_source_layout = QHBoxLayout()
        self.fn_source_label = QLabel("source:")
        self.fn_source_line = QLineEdit()
        self.fn_source_line.setText("off")
        self.fn_source_layout.addWidget(self.fn_source_label)
        self.fn_source_layout.addWidget(self.fn_source_line)

        self.fn_prime_layout = QHBoxLayout()
        self.fn_prime_label = QLabel("prime source:")
        self.fn_prime_line = QLineEdit()
        self.fn_prime_line.setText("off")
        self.fn_prime_layout.addWidget(self.fn_prime_label)
        self.fn_prime_layout.addWidget(self.fn_prime_line)

        self.fn_rate_layout = QHBoxLayout()
        self.fn_rate_label = QLabel("flow rate (uL/min):")
        self.fn_rate_spbx = QDoubleSpinBox()
        self.fn_rate_spbx.setRange(0, 2000)
        self.fn_rate_spbx.setSingleStep(5)
        self.fn_rate_layout.addWidget(self.fn_rate_label)
        self.fn_rate_layout.addWidget(self.fn_rate_spbx)

        self.fn_volume_layout = QHBoxLayout()
        self.fn_volume_label = QLabel("volume (uL):")
        self.fn_volume_spbx = QDoubleSpinBox()
        self.fn_volume_spbx.setRange(0, 10000)
        self.fn_volume_spbx.setSingleStep(10)
        self.fn_volume_layout.addWidget(self.fn_volume_label)
        self.fn_volume_layout.addWidget(self.fn_volume_spbx)

        self.run_wait_layout = QHBoxLayout()
        self.run_wait_label = QLabel("wait time (sec.)")
        self.run_wait_spbx = QDoubleSpinBox()
        self.run_wait_spbx.setRange(0, 600)
        self.run_wait_spbx.setSingleStep(1)
        self.run_wait_layout.addWidget(self.run_wait_label)
        self.run_wait_layout.addWidget(self.run_wait_spbx)

        self.run_program_menu = QComboBox()
        self.run_program_menu.addItems(["Run flush", "Run system prime", "Run PID loop", "Run source prime"])

        self.run_program_bt = QPushButton("Run program")
        self.run_program_bt.clicked.connect(self.run_program)

        self.args_group.setLayout(self.args_layout)
        self.args_layout.addLayout(self.fn_source_layout)
        self.args_layout.addLayout(self.fn_prime_layout)
        self.args_layout.addLayout(self.fn_rate_layout)
        self.args_layout.addLayout(self.fn_volume_layout)
        self.args_layout.addLayout(self.run_wait_layout)
        self.args_layout.addWidget(self.run_program_menu)

        self.testing_layout.addWidget(self.args_group)
        self.testing_layout.addWidget(self.run_program_bt)
        self.testing_group.setLayout(self.testing_layout)

        #---------------------------------------------------------#
        # Create panel for controlling device
        # Create layout for setting
        self.flowrate_layout = QHBoxLayout()
        self.pressure_layout = QHBoxLayout()
        self.valves_layout = QHBoxLayout()
        self.pidp_layout = QHBoxLayout()
        self.pidi_layout = QHBoxLayout()

        # create flow rate area
        self.flowrate_label = QLabel("Flow rate (uL/min) =")
        self.flowrate_spbx = QDoubleSpinBox()
        self.flowrate_spbx.setRange(0, 2000)
        self.flowrate_spbx.setValue(0)
        self.flowrate_spbx.setSingleStep(10)
        self.flowrate_spbx.editingFinished.connect(self.update_flow_params)
        self.flowrate_ch = QCheckBox()
        self.flowrate_ch.clicked.connect(self.update_flow_params)
        self.flowrate_ch.setChecked(True)
        self.flowrate_layout.addWidget(self.flowrate_label)
        self.flowrate_layout.addWidget(self.flowrate_spbx)
        self.flowrate_layout.addWidget(self.flowrate_ch)

        # create pressure area
        self.pressure_label = QLabel("Pressure () =")
        self.pressure_spbx = QDoubleSpinBox()
        self.pressure_spbx.setRange(0, 2000)
        self.pressure_spbx.setValue(0)
        self.pressure_spbx.setSingleStep(10)
        self.pressure_spbx.editingFinished.connect(self.update_flow_params)
        self.pressure_ch = QCheckBox()
        self.pressure_ch.clicked.connect(self.update_flow_params)
        self.pressure_ch.setChecked(False)
        self.pressure_layout.addWidget(self.pressure_label)
        self.pressure_layout.addWidget(self.pressure_spbx)
        self.pressure_layout.addWidget(self.pressure_ch)

        # Create valve area
        self.valve_label = QLabel("Valve:")
        self.valve_line = QLineEdit()
        self.valve_line.setText("off")
        self.valve_line.editingFinished.connect(self.update_current_valve)
        self.valves_layout.addWidget(self.valve_label)
        self.valves_layout.addWidget(self.valve_line)

        # Create PID ara
        self.pidp_label = QLabel("PID P:")
        self.pidp_spbx = QDoubleSpinBox()
        self.pidp_spbx.setValue(self.pid_p)
        self.pidp_spbx.editingFinished.connect(self.update_flow_params)
        self.pidp_layout.addWidget(self.pidp_label)
        self.pidp_layout.addWidget(self.pidp_spbx)

        self.pidi_label = QLabel("PID I:")
        self.pidi_spbx = QDoubleSpinBox()
        self.pidi_spbx.setValue(self.pid_i)
        self.pidi_spbx.editingFinished.connect(self.update_flow_params)
        self.pidi_layout.addWidget(self.pidi_label)
        self.pidi_layout.addWidget(self.pidi_spbx)

        # Setup current group
        self.current_layout.addLayout(self.flowrate_layout)
        self.current_layout.addLayout(self.pressure_layout)
        self.current_layout.addLayout(self.valves_layout)
        self.current_layout.addLayout(self.pidp_layout)
        self.current_layout.addLayout(self.pidi_layout)
        self.current_group.setLayout(self.current_layout)

        #---------------------------------------------------------#
        # Create area for pressing start / stop
        self.control_start = QPushButton("Start")
        self.control_start.clicked.connect(self.start)
        self.control_stop = QPushButton("Stop")
        self.control_stop.clicked.connect(self.stop)
        self.run_layout.addWidget(self.control_start)
        self.run_layout.addWidget(self.control_stop)
        self.run_group.setLayout(self.run_layout)

        #---------------------------------------------------------#
        # Create area for live monitor
        self.live_pressure_layout = QHBoxLayout()
        self.live_flowrate_layout = QHBoxLayout()

        # self.live_label = QLabel("Live")
        self.live_ch = QCheckBox("Live update")
        self.live_ch.setChecked(False)
        self.live_ch.clicked.connect(self.start_live)

        self.pressure_label = QLabel("pressure = 0.0 mBar")
        self.flowrate_label = QLabel("flow rate = 0.0 uL/min")

        self.live_layout.addWidget(self.live_ch)
        self.live_layout.addWidget(self.pressure_label)
        self.live_layout.addWidget(self.flowrate_label)
        self.live_group.setLayout(self.live_layout)

        #---------------------------------------------------------#
        # configure control group
        self.control_layout.addWidget(self.current_group, stretch=2)
        self.control_layout.addWidget(self.run_group)
        self.control_layout.addWidget(self.live_group)
        self.control_group.setLayout(self.control_layout)

        #---------------------------------------------------------#
        # Configure main window
        self.main_layout.columnMinimumWidth(500)
        self.main_layout.addWidget(self.config_group, 0, 0, 3, 1)
        self.main_layout.addWidget(self.testing_group, 0, 1, 3, 1)
        self.main_layout.addWidget(self.control_group, 0, 2, 3, 1)
        self.setLayout(self.main_layout)
        self.layout()


    def start_live(self):
        if self.live_ch.isChecked():
            self.run_live = True
            self.live_timer.start(1000)  # Check every second
        else:
            self.run_live = False
            self.live_timer.stop()


    def update_monitor(self):
        if self.run_live:
            p = self.fc.ob1.getPressureUniversal(1)  # Assume this method exists
            f = self.fc.ob1.getFlowUniversal(1)  # Assume this method exists
            self.pressure_label.setText(f"Pressure = {p} ()")
            self.flowrate_label.setText(f"Flow Rate = {f} uL/min")


    def start(self):
        """Start flow"""
        if self.run_pressure:
            print("Starting pressure controlled flow!")
            self.fc.start_pressure(self.pressure,
                                    verbose=True)
        elif self.run_flowrate:
            print("Starting rate controlled flow!")
            self.fc.start_flow(rate=self.flowrate,
                                 verbose=True)

        self.running = True


    def stop(self):
        """Stop flow"""
        if self.run_pressure:
            print("Starting pressure controlled flow!")
            self.fc.stop_pressure(verbose=True)
        elif self.run_flowrate:
            print("Starting rate controlled flow!")
            self.fc.stop_flow(verbose=True)

        self.running = False


    def update_current_valve(self):
        """Change the valve to specified key"""
        try:
            valve_key = self.valve_line.text()
            self.fc.set_valves(valve_key, verbose=True)
            self.valve = valve_key
        except Exception as e:
            print(f"Key not found, setting to OFF, {e}")
            self.fc.set_valves("off", verbose=True)
            self.valve = "off"
            self.valve_line.setText("off")
        pass


    def update_flow_params(self):
        """Update flow parameters """
        # Make the checkboxes mutually exclusive
        sender = self.sender()
        if sender == self.flowrate_ch:
            self.run_flowrate=True
            self.run_pressure=False
            if self.pressure_ch.isChecked():
                self.pressure_ch.setChecked(False)
        elif sender == self.pressure_ch:
            self.run_flowrate = False
            self.run_pressure = True
            if self.flowrate_ch.isChecked():
                self.flowrate_ch.setChecked(False)
        elif sender == self.flowrate_spbx:
            self.flowrate = self.flowrate_spbx.value()
            if self.running and self.run_flowrate:
                self.fc.ob1.remoteSetTarget(1, self.flowrate)
        elif sender == self.pressure_spbx:
            self.pressure = self.pressure_spbx.value()
            if self.running and self.run_pressure:
                self.fc.ob1.remoteSetTarget(1, self.pressure)
        elif sender == self.pidi_spbx or sender == self.pidp_spbx:
            self.pid_p = self.pidp_spbx.value()
            self.pid_i = self.pidi_spbx.value()
            if self.running:
                reset = True
            else:
                reset = False
            self.fc.ob1.remoteChangePID(1, self.pid_p, self.pid_i, reset=reset)

        print("Flow parameters updated!")


    def save_config(self):
        """
        """
        self.fc.save_config(verbose=True)


    def refresh_config_tabs(self):
        """
        Update current config values with config dictionary
        """
        # refresh the MUX valves
        n_valves = 12
        for ii in range(n_valves):
            line1 = self.mux1_lines[ii]
            line2 = self.mux2_lines[ii]
            key1, key2 = None, None

            for k, v in self.fc.config["mux1"].items():
                if v==ii+1:
                    key1 = k
            for k, v in self.fc.config["mux2"].items():
                if v==ii+1:
                    key2 = k

            line1.setText(key1)
            line2.setText(key2)

        # refresh the PID params
        self.pid_i_spbx.setValue(self.fc.config["PID"]["i"])
        self.pid_p_spbx.setValue(self.fc.config["PID"]["p"])

        # refresh default rates/volumes
        # self.fc.config["rates"]["default"]
        # self.fc.config["rates"]["default"]
        # self.fc.config["rates"]["default"]


    def update_valves_config(self,
                             dev: str = "1",
                             valve: int = None,
                             key: str = None):
        """
        Update MUX configuration dict
        """
        old_key = None
        empty_key = None

        # update the given dev valve key
        for k, val in self.fc.config[f"mux{dev}"].items():
            if val==valve:
                old_key=k
            if k == "":
                empty_key=True
        if empty_key:
            self.fc.config[f"mux{dev}"].pop("")
        if old_key:
            self.fc.config[f"mux{dev}"].pop(old_key)

        # Redefine the valve configuration
        self.fc.config[f"mux{dev}"][key] = valve
        print(f"Valve configuration updated! {key}: {self.fc.config[f'mux{dev}'][key]}")


    def update_pid_config(self):
        """Update config with new P and I params"""
        self.fc.config["PID"]["p"] = round(self.pid_p_spbx.value(),2)
        self.fc.config["PID"]["i"] = round(self.pid_i_spbx.value(),2)
        print("PID configuration updated!")


    def update_specs_config(self):
        pass


    def update_valves_config(self,
                             dev: str = "1",
                             valve: int = None,
                             key: str = None):
        """
        Update MUX configuration dict
        """
        old_key = None
        empty_key = None

        # update the given dev valve key
        for k, val in self.fc.config[f"mux{dev}"].items():
            if val==valve:
                old_key=k
            if k == "":
                empty_key=True
        if empty_key:
            self.fc.config[f"mux{dev}"].pop("")
        if old_key:
            self.fc.config[f"mux{dev}"].pop(old_key)

        # Redefine the valve configuration
        self.fc.config[f"mux{dev}"][key] = valve
        print(f"Valve config updated! {key}: {self.fc.config[f'mux{dev}'][key]}")


    def run_program(self):
        """Run selected program"""
        # Turn off live monitor
        if self.live_ch.isChecked():
            self.live_ch.setChecked(False)
            self.start_live()

        current_program = self.run_program_menu.currentText()
        source = self.fn_source_line.text()
        prime_buffer = self.fn_prime_line.text()
        volume = self.fn_volume_spbx.value()
        rate = self.fn_rate_spbx.value()
        wait = self.run_wait_spbx.value()
        try:
            print(f"Runnning proram: {current_program}")
            if current_program == "Run flush":
                self.fc.run_flush(wait=wait, verbose=True)
            elif current_program == "Run system prime":
                self.fc.run_system_prime(True)
            elif current_program == "Run PID loop":
                self.fc.run_pid_loop(source, rate, volume=volume, wait=wait, reset=True, verbose=True)
            elif current_program == "Run source prime":
                self.fc.run_prime_source(source, prime_buffer, rate, volume, wait, True)
        except Exception as e:
            print(f"Exception occured: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FlowControlWidget()
    ex.show()
    sys.exit(app.exec_())
