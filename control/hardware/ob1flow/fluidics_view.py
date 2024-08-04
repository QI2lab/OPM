"""
Run fluidics control and save configuration file

07/2024 Steven Sheppard
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from fluidics_widget import FlowControlWidget  # Adjust the import as necessary

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Main Window with Flow Control Widget')

        # Create central widget and set the layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create an instance of FlowControlWidget and add it to the layout
        self.flow_control_widget = FlowControlWidget()
        layout.addWidget(self.flow_control_widget)

        # Set central widget
        self.setCentralWidget(central_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
