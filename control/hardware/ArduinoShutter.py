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
from warnings import resetwarnings
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
        self.serial = serial.Serial(port=self.com_port, baudrate=115200, timeout=.1)
        time.sleep(2)
  
    def openShutter(self):
        response = self.writeSerialPort('o')
        if self.verbose: print(response)    
        if self.verbose: print("Shutter opened")
        time.sleep(0.5)
    
    def closeShutter(self):
        response = self.writeSerialPort('c')
        if self.verbose: print(response)
        if self.verbose: print("Shutter closed")
        time.sleep(0.5)

    def writeSerialPort(self,command):
        self.serial.flushInput
        self.serial.flushOutput
        self.serial.write((command+'\n').encode())
        time.sleep(0.02)
        data = self.serial.readline().decode('ascii').strip('\r\n')
        return data

    def close(self):
        self.closeShutter()
        self.serial.close()
        if self.verbose: 
            print("Closed Arduino shutter controller")