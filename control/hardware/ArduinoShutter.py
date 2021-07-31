#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# The basic I/O class for Arduino controller of Edmund Optics TTL controller
# ----------------------------------------------------------------------------------------
# Doug Shepherd
# 07/2021
# douglas.shepherd@asu.edu
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
import serial
import time

# ----------------------------------------------------------------------------------------
# ArduinoShutter Class Definition
# ----------------------------------------------------------------------------------------
class ArduinoShutter():
    def __init__(self,
                 parameters = False):

        # Define attributes
        self.com_port = parameters.get('arduino_com_port', 'COM7')
        self.verbose = parameters.get('verbose',False)   
        
        # Create serial port
        self.serial = serial.Serial(port = self.com_port, 
                                    baudrate = 115200, 
                                    parity= serial.PARITY_EVEN, 
                                    bytesize=serial.EIGHTBITS, 
                                    stopbits=serial.STOPBITS_TWO, 
                                    timeout=0.1)

        # Define initial shutter status
        self.shutter_state = 'Closed'
        self.closeShutter()
    
    def openShutter(self):
        if self.getShutterState=='Closed':
            self.shutters_state == 'Open'
        
        if self.verbose: 
            print("Shutter opened")
    
    def closeShutter(self):
        if self.getShutterState=='Open':
            self.shutter_state == 'Closed'

        if self.verbose: print("Shutter closed")

    def getShutterState(self):
        if self.verbose: 
            print(self.shutter_state)
        return self.shutter_state

    def close(self):
        self.serial.close()
        if self.verbose: 
            print("Closed Arduino shutter controller")