"""
Fluidics control widget

TODO: Live mode does not work when running.

07/2024 Steven Sheppard
"""
import sys
import threading
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QDoubleSpinBox, QPushButton,
                             QLabel, QCheckBox, QFormLayout,
                             QLineEdit, QGroupBox, QApplication, QComboBox)
from PyQt5.QtCore import QTimer
from hardware.ElveFlow_fluidics import FlowControl  # Adjust the import as necessary

class FlowControlWidget(QWidget):
    def __init__(self):
        super().__init__()

        # initialize controls
        self.fc = FlowControl()
        self.fc.startup()
        self.create_ui()

        self.pressure = 0
        self.flowrate = 0
        self.volume = 0
        self.runtime = 0
        self.run_pressure = False
        self.run_flowrate = True
        self.run_runtime = False
        self.run_volume = True
        self.run_live = False

        # Timer for live monitoring
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_monitor)

    def create_ui(self):
        self.main_layout = QGridLayout()

        #---------------------------------------------------------#
        # Define area to set the Valve configuration
        self.mux_group = QGroupBox("MUX valve configuration")
        self.mux_layout = QVBoxLayout()

        self.mux_config_layout = QHBoxLayout()
        self.mux1_group = QGroupBox("device 1")
        self.mux1_form = QFormLayout()
        self.mux2_group = QGroupBox("device 2")
        self.mux2_form = QFormLayout()

        # sort dictionary before displayiny
        self.fc.config["mux1"] = dict(sorted(self.fc.config["mux1"].items(), key=lambda item: item[1]))
        self.fc.config["mux2"] = dict(sorted(self.fc.config["mux2"].items(), key=lambda item: item[1]))

        for key, val in self.fc.config["mux1"].items():
            # Create a QlineEdit and map to change MUX config function
            line_edit = QLineEdit(str(key))
            line_edit.textChanged.connect(lambda text, val=val: self.update_mux_config("1", val, text))
            self.mux1_form.addRow(str(val), line_edit)


        self.fc.config["mux2"] = dict(sorted(self.fc.config["mux2"].items(), key=lambda item: item[1]))
        for key, val in self.fc.config["mux2"].items():
            # Create a QlineEdit and map to change MUX config function
            line_edit = QLineEdit(str(key))
            line_edit.textChanged.connect(lambda text, val=val: self.update_mux_config("2", val, text))
            self.mux2_form.addRow(str(val), line_edit)

        self.mux1_group.setLayout(self.mux1_form)
        self.mux2_group.setLayout(self.mux2_form)
        self.mux_config_layout.addWidget(self.mux1_group)
        self.mux_config_layout.addWidget(self.mux2_group)

        self.mux_control_layout = QHBoxLayout()
        self.mux_change_bt = QPushButton("Set current state")
        self.mux_current = QLineEdit()
        self.mux_current.setText("off")
        self.mux_change_bt.clicked.connect(self.update_mux)

        self.mux_control_layout.addWidget(self.mux_current)
        self.mux_control_layout.addWidget(self.mux_change_bt)
        self.mux_layout.addLayout(self.mux_config_layout)
        self.mux_layout.addLayout(self.mux_control_layout)
        self.mux_group.setLayout(self.mux_layout)

        #---------------------------------------------------------#
        # Define area to configure controlled flow
        self.flow_group = QGroupBox("Flow control settings")
        self.flow_group_layout = QVBoxLayout()

        # Configure PID parameters
        self.pid_group = QGroupBox("PID parameters")
        self.pid_layout = QVBoxLayout()

        self.pid_p_layout = QHBoxLayout()
        self.pid_p_label = QLabel("P =")
        self.pid_p_spinbox = QDoubleSpinBox()
        self.pid_p_spinbox.setRange(0, 10)
        self.pid_p_spinbox.setSingleStep(0.01)
        self.pid_p_spinbox.setValue(self.fc.config["PID"]["p"])
        self.pid_p_layout.addWidget(self.pid_p_label)
        self.pid_p_layout.addWidget(self.pid_p_spinbox)

        self.pid_i_layout = QHBoxLayout()
        self.pid_i_label = QLabel("I =")
        self.pid_i_spinbox = QDoubleSpinBox()
        self.pid_i_spinbox.setRange(0, 10)
        self.pid_i_spinbox.setSingleStep(0.01)
        self.pid_i_spinbox.setValue(self.fc.config["PID"]["i"])
        self.pid_i_layout.addWidget(self.pid_i_label)
        self.pid_i_layout.addWidget(self.pid_i_spinbox)

        self.pid_update_bt = QPushButton("Update")
        self.pid_update_bt.clicked.connect(self.update_pid_config)

        self.pid_layout.addLayout(self.pid_p_layout)
        self.pid_layout.addLayout(self.pid_i_layout)
        self.pid_layout.addWidget(self.pid_update_bt)
        self.pid_group.setLayout(self.pid_layout)

        # Configure flow control settings
        self.flow_params_group = QGroupBox("Flow control units")
        self.flow_params_layout = QVBoxLayout()

        self.flow_rate_layout = QHBoxLayout()
        self.flow_rate_label = QLabel("Flow rate =")
        self.flow_rate_spinbox = QDoubleSpinBox()
        self.flow_rate_spinbox.setRange(0, 1500)
        self.flow_rate_spinbox.setValue(0)
        self.flow_rate_spinbox.setSingleStep(10)
        self.flow_rate_ch = QCheckBox()
        self.flow_rate_ch.clicked.connect(self.update_flow_params)
        self.flow_rate_ch.setChecked(True)
        self.flow_rate_layout.addWidget(self.flow_rate_label)
        self.flow_rate_layout.addWidget(self.flow_rate_spinbox)
        self.flow_rate_layout.addWidget(self.flow_rate_ch)

        self.flow_pressure_layout = QHBoxLayout()
        self.flow_pressure_label = QLabel("Pressure =")
        self.flow_pressure_spinbox = QDoubleSpinBox()
        self.flow_pressure_spinbox.setRange(0, 2000)
        self.flow_pressure_spinbox.setValue(0)
        self.flow_pressure_spinbox.setSingleStep(10)
        self.flow_pressure_ch = QCheckBox()
        self.flow_pressure_ch.clicked.connect(self.update_flow_params)
        self.flow_pressure_ch.setChecked(False)
        self.flow_pressure_layout.addWidget(self.flow_pressure_label)
        self.flow_pressure_layout.addWidget(self.flow_pressure_spinbox)
        self.flow_pressure_layout.addWidget(self.flow_pressure_ch)

        # Configure flow quantity settings
        self.flow_qty_group = QGroupBox("Flow quantity units")
        self.flow_qty_layout = QVBoxLayout()

        self.flow_volume_layout = QHBoxLayout()
        self.flow_volume_label = QLabel("Volume (uL) =")
        self.flow_volume_spinbox = QDoubleSpinBox()
        self.flow_volume_spinbox.setRange(0, 20000)
        self.flow_volume_spinbox.setSingleStep(50)
        self.flow_volume_ch = QCheckBox()
        self.flow_volume_ch.setChecked(True)
        self.flow_volume_ch.clicked.connect(self.update_flow_params)
        self.flow_volume_layout.addWidget(self.flow_volume_label)
        self.flow_volume_layout.addWidget(self.flow_volume_spinbox)
        self.flow_volume_layout.addWidget(self.flow_volume_ch)

        self.flow_runtime_layout = QHBoxLayout()
        self.flow_runtime_label = QLabel("Runtime (seconds) =")
        self.flow_runtime_spinbox = QDoubleSpinBox()
        self.flow_runtime_spinbox.setRange(0, 3600)
        self.flow_runtime_spinbox.setSingleStep(1)
        self.flow_runtime_ch = QCheckBox()
        self.flow_runtime_ch.setChecked(False)
        self.flow_runtime_ch.clicked.connect(self.update_flow_params)
        self.flow_runtime_layout.addWidget(self.flow_runtime_label)
        self.flow_runtime_layout.addWidget(self.flow_runtime_spinbox)
        self.flow_runtime_layout.addWidget(self.flow_runtime_ch)

        self.flow_params_bt = QPushButton("Update flow settings")
        self.flow_params_bt.clicked.connect(self.update_flow_params)

        self.flow_params_layout.addLayout(self.flow_rate_layout)
        self.flow_params_layout.addLayout(self.flow_pressure_layout)
        self.flow_params_layout.addLayout(self.flow_volume_layout)
        self.flow_params_layout.addLayout(self.flow_runtime_layout)
        self.flow_params_layout.addWidget(self.flow_params_bt)
        self.flow_params_group.setLayout(self.flow_params_layout)

        # Configure group
        self.flow_group_layout.addWidget(self.pid_group)
        self.flow_group_layout.addWidget(self.flow_params_group, stretch=2)
        self.flow_group.setLayout(self.flow_group_layout)

        #---------------------------------------------------------#
        # Define buttons to control device state
        self.states_group = QGroupBox("Device State Controls")
        self.state_layout = QVBoxLayout()

        self.startup_btn = QPushButton('Open connection', self)
        self.startup_btn.clicked.connect(self.fc.startup)

        self.reset_btn = QPushButton('Reset settings', self)
        self.reset_btn.clicked.connect(self.fc.reset)

        self.shutdown_btn = QPushButton('Close connection', self)
        self.shutdown_btn.clicked.connect(self.fc.shutdown)

        self.state_layout.addWidget(self.startup_btn)
        self.state_layout.addWidget(self.reset_btn)
        self.state_layout.addWidget(self.shutdown_btn)
        self.states_group.setLayout(self.state_layout)

        #---------------------------------------------------------#
        # Define area to configure Pressure controlled flow
        self.monitor_group = QGroupBox("Device Monitor")
        self.monitor_layout = QVBoxLayout()
        self.monitor_pressure_layout = QHBoxLayout()
        self.monitor_velocity_layout = QHBoxLayout()
        self.monitor_live_layout = QHBoxLayout()

        self.monitor_live_label = QLabel("Live")
        self.monitor_live_ch = QCheckBox()
        self.monitor_live_ch.setChecked(False)
        self.monitor_live_ch.clicked.connect(self.start_live)

        self.pressure_label = QLabel("pressure = 0.0 mBar")
        self.flowrate_label = QLabel("flow rate = 0.0 uL/min")

        self.monitor_live_layout.addWidget(self.monitor_live_label)
        self.monitor_live_layout.addWidget(self.monitor_live_ch)
        self.monitor_pressure_layout.addWidget(self.pressure_label)
        self.monitor_velocity_layout.addWidget(self.flowrate_label)

        self.monitor_layout.addLayout(self.monitor_live_layout)
        self.monitor_layout.addLayout(self.monitor_pressure_layout)
        self.monitor_layout.addLayout(self.monitor_velocity_layout)
        self.monitor_group.setLayout(self.monitor_layout)

        #---------------------------------------------------------#
        # Define area to run a program
        self.run_program_group = QGroupBox("Run programs")
        self.run_program_layout = QVBoxLayout()

        self.source_layout = QHBoxLayout()
        self.source_label = QLabel("source")
        self.source_line = QLineEdit()
        self.source_line.setText("off")
        self.source_layout.addWidget(self.source_label)
        self.source_layout.addWidget(self.source_line)

        self.primesource_layout = QHBoxLayout()
        self.primesource_label = QLabel("prime source")
        self.primesource_line = QLineEdit()
        self.primesource_line.setText("off")
        self.primesource_layout.addWidget(self.primesource_label)
        self.primesource_layout.addWidget(self.primesource_line)

        self.run_rate_layout = QHBoxLayout()
        self.run_rate_label = QLabel("rate")
        self.run_rate_spinbox = QDoubleSpinBox()
        self.run_rate_spinbox.setRange(0, 2000)
        self.run_rate_spinbox.setSingleStep(5)
        self.run_rate_layout.addWidget(self.run_rate_label)
        self.run_rate_layout.addWidget(self.run_rate_spinbox)

        self.run_volume_layout = QHBoxLayout()
        self.run_volume_label = QLabel("volume")
        self.run_volume_spinbox = QDoubleSpinBox()
        self.run_volume_spinbox.setRange(0, 10000)
        self.run_volume_spinbox.setSingleStep(10)
        self.run_volume_layout.addWidget(self.run_volume_label)
        self.run_volume_layout.addWidget(self.run_volume_spinbox)

        self.run_wait_layout = QHBoxLayout()
        self.run_wait_label = QLabel("wait time")
        self.run_wait_spinbox = QDoubleSpinBox()
        self.run_wait_spinbox.setRange(0, 600)
        self.run_wait_spinbox.setSingleStep(1)
        self.run_wait_layout.addWidget(self.run_wait_label)
        self.run_wait_layout.addWidget(self.run_wait_spinbox)

        self.run_program_menu = QComboBox()
        self.run_program_menu.addItems(["Run flush", "Run system prime", "Run PID", "Run prime"])

        self.run_program_bt = QPushButton("Run program")
        self.run_program_bt.clicked.connect(self.run_program)

        self.run_program_layout.addLayout(self.source_layout)
        self.run_program_layout.addLayout(self.primesource_layout)
        self.run_program_layout.addLayout(self.run_rate_layout)
        self.run_program_layout.addLayout(self.run_volume_layout)
        self.run_program_layout.addLayout(self.run_wait_layout)
        self.run_program_layout.addWidget(self.run_program_menu)
        self.run_program_layout.addWidget(self.run_program_bt)
        self.run_program_group.setLayout(self.run_program_layout)

        #---------------------------------------------------------#
        # Define area to stop and start flow, show checkbox for for
        # selecting which type of flow to start
        self.control_group = QGroupBox("Flow Controls")
        self.control_group_layout = QVBoxLayout()
        self.flow_Status_layout = QHBoxLayout()
        self.control_start = QPushButton("Start")
        self.control_start.clicked.connect(self.start)
        self.control_stop = QPushButton("Stop")
        self.control_stop.clicked.connect(self.stop)
        self.flow_Status_layout.addWidget(self.control_start)
        self.flow_Status_layout.addWidget(self.control_stop)
        self.control_group.setLayout(self.flow_Status_layout)

        #---------------------------------------------------------#
        # Configure main window
        self.main_layout.addWidget(self.mux_group, 1, 1, 3, 1)
        self.main_layout.addWidget(self.flow_group, 1, 2, 3, 1)
        self.main_layout.addWidget(self.run_program_group, 2, 3, 2, 1)
        self.main_layout.addWidget(self.states_group, 1, 4, 1, 1)
        self.main_layout.addWidget(self.control_group, 2, 4, 1, 1)
        self.main_layout.addWidget(self.monitor_group, 3, 4, 1, 1)
        self.setLayout(self.main_layout)
        self.layout()

    def start_live(self):
        if self.monitor_live_ch.isChecked():
            self.run_live = True
            self.monitor_timer.start(1000)  # Check every second
        else:
            self.run_live = False
            self.monitor_timer.stop()

    def monitor_thread(self):
        self.monitor_thread = threading.Thread(target=self.update_monitor)
        self.monitor_thread.start()

    def update_monitor(self):
        if self.run_live:
            pressure = self.fc.ob1.getPressureUniversal(1)  # Assume this method exists
            flowrate = self.fc.ob1.getFlowUniversal(1)  # Assume this method exists
            self.pressure_label.setText(f"Pressure = {pressure}")
            self.flowrate_label.setText(f"Flow Rate = {flowrate} uL/min")

    def start_thread(self):
        self.start_thread = threading.Thread(target=self.start)
        self.start_thread.start()

    def stop_thread(self):
        self.stop_thread = threading.Thread(target=self.stop)
        self.stop_thread.start()

    def start(self):
        """Start flow"""
        if self.run_pressure:
            print("Running pressure controlled flow!")
            if self.run_runtime:
                runtime = self.runtime
                volume=None
            elif self.run_volume:
                volume = self.volume
                runtime=None

            self.fc.run_at_pressure(self.pressure,
                                volume=volume,
                                runtime=runtime,
                                reset=False,
                                verbose=True)
        elif self.run_flowrate:
            print("Running rate controlled flow!")
            if self.run_runtime:
                runtime = self.runtime
                volume=None
            elif self.run_volume:
                volume = self.volume
                runtime=None

            self.fc.run_pid_loop(rate=self.flowrate,
                                    volume=volume,
                                    runtime=runtime,
                                    verbose=True)

    def stop(self):
        """Stop flow"""
        self.fc.stop_flow()

    def update_mux(self):
        """Change the valve to specified key"""
        valve_key = self.mux_current.text()
        self.fc.set_valves(valve_key, verbose=True)
        pass

    def update_flow_params(self):
        """Update flow parameters """
        # Make the checkboxes mutually exclusive
        sender = self.sender()
        if sender == self.flow_volume_ch:
            self.run_volume = True
            self.run_runtime = False
            if self.flow_runtime_ch.isChecked():
                self.flow_runtime_ch.setChecked(False)
        elif sender == self.flow_runtime_ch:
            self.run_runtime=True
            self.run_volume=False
            if self.flow_volume_ch.isChecked():
                self.flow_volume_ch.setChecked(False)
        elif sender == self.flow_rate_ch:
            self.run_flowrate =True
            self.run_pressure = False
            if self.flow_pressure_ch.isChecked():
                self.flow_pressure_ch.setChecked(False)
        elif sender == self.flow_pressure_ch:
            self.run_flowrate = False
            self.run_pressure = True
            if self.flow_rate_ch.isChecked():
                self.flow_rate_ch.setChecked(False)
        elif sender == self.flow_params_bt:
            self.volume = self.flow_volume_spinbox.value()
            self.runtime = self.flow_runtime_spinbox.value()
            self.flowrate = self.flow_rate_spinbox.value()
            self.pressure = self.flow_pressure_spinbox.value()

            print("Flow parameters updated!")

    def update_pid_config(self):
        """Update config with new P and I params"""
        self.fc.config["PID"]["p"] = self.pid_p_spinbox.value()
        self.fc.config["PID"]["i"] = self.pid_i_spinbox.value()
        self.fc.save_config()
        print("PID configuration updated!")

    def update_mux_config(self,
                          mux_id: str = "1",
                          mux_valve: int = None,
                          mux_key: str = None):
        """
        Update MUX configuration dict
        """
        old_key = None
        empty_key = None
        for key, val in self.fc.config[f"mux{mux_id}"].items():
            if val==mux_valve:
                old_key=key
            if key == "":
                empty_key=True
        if empty_key:
            self.fc.config[f"mux{mux_id}"].pop("")
        if old_key:
            self.fc.config[f"mux{mux_id}"].pop(old_key)

        # Redefine the valve configuration
        self.fc.config[f"mux{mux_id}"][mux_key] = mux_valve
        # self.fc.config[f"mux{mux_id}"] = dict(sorted(self.fc.config[f"mux{mux_id}"].items(), key=lambda item: item[1]))
        self.fc.save_config()
        print("MUX configuration saved")

    def run_program(self):
        """Run selected program"""

        current_program = self.run_program_menu.currentText()
        source = self.source_line.text()
        prime_buffer = self.primesource_line.text()
        volume = self.run_volume_spinbox.value()
        rate = self.run_rate_spinbox.value()
        wait = self.run_wait_spinbox.value()
        try:
            print(f"Trying to run proram: {current_program}")
            if current_program == "Run flush":
                self.fc.run_flush(source, rate, volume, wait, True)
            elif current_program == "Run system prime":
                self.fc.run_system_prime(True)
            elif current_program == "Run PID":
                self.fc.run_pid(source, rate, volume, wait, True)
            elif current_program == "Run prime":
                self.fc.run_prime(source, prime_buffer, rate, volume, wait, False)
        except Exception as e:
            print(f"Exception occured: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FlowControlWidget()
    ex.show()
    sys.exit(app.exec_())
