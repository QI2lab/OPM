#!/usr/bin/env python
"""
Stage scanning Snouty control software
Original code for stage scanning dual objective open top light sheet - Adam Glaser
Modified code for stage scanning single objective Snouty light sheet - Doug Shepherd

Glaser 2020
Shepherd 2020
"""

import numpy as np

############# SCAN PARAMETERS #############

# saving parameters
storage_drive = 'Y'                             # hard disk to save on.
filename = 'human_kidney_1_638.561_10.16.2019'  # directory name

# scan parameters 
x_min = -16.0   # (mm)
x_max = 4.0     # (mm)
y_min = -1.2    # (mm)
y_max = 1.2     # (mm)
z_min = 0.400   # (mm)
z_max = 0.500   # (mm)

# camera parameters
cam_size_X = 256        # (pixels) # DPS notes: figure out for our rig
cam_size_Y = 2048       # (pixels)
exposure_time = 50.0    # (ms) DPS notes: what is the max exposure time we can use to get correct spacing?
binning = '1x1'
pixel_size = 6.5/40     # (um) pixel size in um with Snouty and 200 mm TL 
                        # DPS notes: do we need to take stretching due to Snouty angle into account here?

# overlaps
x_width = (cam_size_X*pixel_size/np.sqrt(2))*0.001    # (mm) step in X to give 45-45-90 triangle for processing
y_width = (cam_size_Y*pixel_size*0.9)*0.001           # (mm) 10% overlap between adjacent lateral tiles
z_width = (x_width*np.sin(30.*np.pi/180.)*0.9)*0.001  # (mm) 10% overlap in Z given a 30 degree Snouty tilt

# laser parameters
laser_wavelengths = np.array([405, 561, 605])    # (nm) lambda
laser_powers = np.array([12.0,24.0])             # (%power) 

######### INITIALIZE PARAMETERS ###########
x_length = x_max - x_min        # (mm)
x_offset = x_max - x_length/2   # (mm)
y_length = y_max - y_min        # (mm)
y_offset = y_max - y_length/2   # (mm)
z_length = z_max - z_min        # (mm)
z_offset = z_max - z_length/2   # (mm)

############ BEGIN SCANNING ##############
lsmfx.scan3D(storage_drive, filename, # file information
             x_offset, y_offset, z_offset,
             x_length, y_length, z_length,
             x_width, y_width, z_width, 
             cam_size_X, cam_size_Y, exposure_time, binning, 
             laser_wavelengths, laser_powers)


# The MIT License
#
# Copyright (c) 2020 Adam Glaser, University of Washington
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
