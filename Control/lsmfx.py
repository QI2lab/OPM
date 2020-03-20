#!/usr/bin/env python
"""
Stage scanning LSFM code

# Adam Glaser 07/19
# Douglas Shepherd 03/20

"""

import ctypes
import ctypes.util
import numpy
import time
import math
import camera.hamamatsu_camera as hc
import rs232.RS232 as RS232
import laser.obis as obis
import xyz_stage.tiger as tiger
import utils.utils as utils
import h5py
import warnings
import os.path
import errno
import sys
import scipy.ndimage
import h5py
import warnings
import gc
import os
import os.path
from os import path
from pathlib import Path
import shutil
import logging
import tifffile

def scan3D(storage_drive, filename, 
        	x_offset, y_offset, z_offset,
            x_length, y_length, z_length,
            x_width, y_width, z_width, 
            cam_size_X, cam_size_Y, exposure_time, binning, 
            laser_wavelengths, laser_powers, vis = 0):

	# decode binning string
	if binning == '1x1':
		binFactor = 1
	elif binning == '2x2':
		binFactor = 2
	elif binning == '4x4':
		binFactor = 4
	else:
		binFactor = 1

	# determine number of laser wavelengths requested
	if laser_wavelengths.size > 1:
		num_wavelengths = len(laser_wavelengths)
	else:
		num_wavelengths = 1

	############# SETUP CAMERA #############

	# Initialize camera
	hcam = hc.HamamatsuCameraMR(camera_id=0)
	print(hcam)

	# Set readout properties
	hcam.setPropertyValue("defect_correct_mode", "OFF") 	# keep defect mode off
	hcam.setPropertyValue("readout_speed", 2) 				# 1 or 2. 2 is fastest mode
	
	# DPS note: fix for our setup
	# set subarray properties
	hcam.setPropertyValue("subarray_vsize", cam_size_Y)
	hcam.setPropertyValue("subarray_hsize", cam_size_X)
	hcam.setPropertyValue("subarray_hpos", 1024-cam_size_X/2) 	# DPS notes: double check this works for our rig

	# Set binning properties
	hcam.setPropertyValue("binning", binFactor)

	# Set trigger properties
	hcam.setPropertyValue("trigger_source", 'INTERNAL') 	# 1 (internal), 2 (external), 3 (software)
	hcam.setPropertyValue("trigger_mode", 'START') 			# 1 (normal), 6 (start)
	hcam.setPropertyValue("trigger_active", 'EDGE') 		# 1 (edge), 2 (level), 3 (syncreadout)
	hcam.setPropertyValue("trigger_polarity", 'POSITIVE') 	# 1 (negative), 2 (positive)

	############# SETUP SCANNING PROPERTIES #############

	# DPS notes: Jon Daniels provided that the stage quanta is 0.172 um/s, 
	#            but they think the slowest stable speed is around 2 um/s.
	# TO DO: Need to verify what the actual slowest scan is. Once we have this, 
	#        we can create a lookup table of exposure time vs scan speed
	# stage scanning speed
	scan_speed = .002 										# (mm/s)

	# Adjust for binning factor
	x_width = x_width*binFactor								# (mm)
	y_width = y_width*binFactor								# (mm)
	cam_size_X = int(cam_size_X/binFactor)					# (pixels)
	cam_size_Y = int(cam_size_Y/binFactor)					# (pixels)
	num_frames = int(round(x_length/x_width))				# (frames)
	y_tiles = int(round(y_length/y_width))					# (tiles)
	exposure_time = ((x_width/cam_size_X) / scan_speed) 	# (s)

	# DPS notes: double check scan speed calculation so that this automatically produces the 45-45-90 needed for
	# 			 Adam's HDF5 strategy.

	# Set exposure time
	hcam.setPropertyValue("exposure_time", exposure_time) 	# (s)

	# Set aquisition mode
	hcam.setACQMode("fixed_length", nFrames)
		
	############# SETUP XYZ STAGE #############

	xyzStage = tiger.Tiger(baudrate = 115200, port = 'COM7') # DPS notes: fix scan setup call for Tiger controller
	xyzStage.setScan(1) # DPS notes: fix scan setup call for Tiger controller
	xyzStage.setBacklash(0)
	initialPos = xyzStage.getPosition()
	print(xyzStage)

	############### SETUP LASERS ##############

	# DPS notes: have to figure this out for Coherent OBIS laser box
	# 			 one alternative would be to just use digital triggering 
	#            with DAQmx until I finish the rs232 driver
	laser = obis.Obis(baudrate = 115200, port = 'COM11')
	laser.turnOff(405)
	laser.turnOff(488)
	laser.turnOff(561)
	laser.turnOff(637)
	laser.turnOff(730)

	########## PREPARE FILE STRUCTURE ###########

	storage_path = Path(storage_drive)
	directory_path = storage_path / sample_directory

	if directory_path.exists():
		shutil.rmtree(directory_path, ignore_errors=True)
	time.sleep(1)
	os.makedirs(directory_path)
	logging.basicConfig(filename=(directory_path / 'log.txt'))
	dest = directory_path / 'data.h5'
	imgShape = (nFrames, cam_size_X, cam_size_Y)
	chunkSize1 = 256/binFactor
	chunkSize2 = 32/binFactor
	chunkSize3 = 256/binFactor
	write_threads = []
	im = numpy.zeros((nFrames, cam_size_X, cam_size_Y), dtype = 'uint16')

	############## START SCANNING #############
	
	f = h5py.File(dest,'a')
	
	for i in range(num_wavelengths):
		for j in range(z_tiles):
			for k in range(y_tiles):

				# GET TILE NUMBER
				idx = k+j*y_tiles+i*z_tiles
				idx_tile = k+j*y_tiles
				idx_channel = i

				# GET NAME FOR NEW VISUALIZATION FILE
				if idx == 0:
					dest_vis = directory_path / 'vis' + str(idx) + '.h5'
				else:
					dest_vis = directory_path / 'vis' + str(idx) + '.h5'
					dest_vis_prev = directory_path / 'vis' + str(idx-1) + '.h5'

				# INITIALIZE H5 FILE VARIABLES
				if idx == 0:
					tgroup = f.create_group('/t00000')
				resgroup = f.create_group('/t00000/s' + str(idx).zfill(2) + '/' + str(0))

				# DPS notes: this is for gzip (non-lossy) compression
				data = f.require_dataset('/t00000/s' + str(idx).zfill(2) + '/' + str(0) + 
										 '/cells', chunks = (chunkSize1, chunkSize2, chunkSize3), 
										 dtype = 'int16', shape = imgShape, compression = "gzip")

				# GO TO INITIAL POSITIONS
				xyzStage.setVelocity('X',0.5)
				xyzStage.setVelocity('Y',0.5)
				xyzStage.setVelocity('Z',0.5)

				x_pos = x_length/2.0 - x_offset							# (mm)
				y_pos = -y_length/2.0+k*y_width+y_width/2.0 + y_offset	# (mm)
				z_pos = # TO DO: Fix
				print('Starting tile ' + str(idx) + '/' + str(yTiles*nWavelengths))
				print('x position: ' + str(x_pos)+ ' mm')
				print('y position: ' + str(y_pos)+ ' mm')
				print('z position: ' + str(z_pos)+ ' mm')

				xyzStage.goAbsolute('X', -x_pos-0.035, False)	# (mm)
				xyzStage.goAbsolute('Y', y_pos, False)			# (mm)
				xyzStage.goAbsolute('Z', z_pos, False)			# (mm)

				xyzStage.setVelocity('X',scan_speed)			# (mm/s)
				xyzStage.setScanR(-x_pos, -x_pos + x_length)   	# (mm) DPS notes: check SCANR setup
				xyzStage.setScanV(y_pos) 						# (mm) DPS notes: check SCANV setup.

				# TURN LASER ON
				# TO DO: fix somehow for OBIS laser box
				if laser_wavelengths.size > 1:
					laser.setPower(laser_wavelengths[i], laser_powers[i])
					laser.turnOn(laser_wavelengths[i])
				else:
					laser.setPower(laser_wavelengths, laser_powers[i])
					laser.turnOn(laser_wavelengths)

				# START SCAN
				hcam.startAcquisition()
				xyzStage.scan(False)
				print('Writing resolution level 0')

				# CAPTURE IMAGES
				count_old = 0
				count_new = 0
				count = 0

				while count < num_frames-1:
					time.sleep(0.01)
					# Get frames.
					[frames, dims] = hcam.getFrames()
					count_old = count
					# Save frames.
					for aframe in frames:
						np_data = aframe.getData()
						im[count] = numpy.reshape(np_data, (cam_size_X, cam_size_Y))
						count += 1
					count_new = count
					if count_new == count_old:
						count = num_frames
					print(str(count_new) + '/' + str(num_frames) + ' frames collected...')
					data[count_old:count_new] = im[count_old:count_new]

				# TURN LASER OFF
				if laser_wavelengths.size > 1:
					laser.turnOff(laser_wavelengths[i])
				else:
					laser.turnOff(laser_wavelengths)

				# WRITE DOWNSAMPLED RESOLUTIONS
				if idx > 0:
					previous_thread = write_threads[idx-1]
					while previous_thread.alive() == True:
						time.sleep(0.1)

				current_thread = utils.writeBDV(f, im, idx, binFactor)
				write_threads.append(current_thread)
				current_thread.start()

				if idx == (nWavelengths*y_tiles-1):
					current_thread.join()

				hcam.stopAcquisition()
				gc.collect()

	utils.write_xml(drive = drive, save_dir = save_dir, idx = idx, idx_tile = idx_tile, 
					idx_channel = idx_channel, channels = num_wavelengths, tiles_y = y_tiles, 
					tiles_z = z_tiles, sampling = x_width, binning = binFactor, offset_y = y_width, 
					offset_z = z_width, x = imgShape[0], y = imgShape[1], z = imgShape[2])

	laser.shutDown()
	hcam.shutdown()
	f.close()

	xyzStage.setVelocity(0.5,0.5)

	xyzStage.goAbsolute('X', initialPos[0], False)
	xyzStage.goAbsolute('Y', initialPos[1], False)

	xyzStage.shutDown()
	
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
