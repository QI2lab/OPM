#!/usr/bin/python
"""
BDV and BigStitcher utils

# Adam Glaser 07/19
# Douglas Shepherd 03/20

"""

import threading
import time
import math
import numpy
import warnings
import scipy
import h5py
import gc

class writeBDV(object):

	def __init__(self, f, img_3d, idx, binFactor): 

		self.f = f
		self.img_3d = img_3d
		self.idx = idx
		self.binFactor = binFactor

		self.thread = threading.Thread(target=self.run, args=())

	def run(self):

		res_list = [1, 2, 4, 8]

		res_np = numpy.zeros((len(res_list), 3), dtype = 'float64')
		res_np[:,0] = res_list
		res_np[:,1] = res_list
		res_np[:,2] = res_list
		
		sgroup = self.f.create_group('/s' + str(self.idx).zfill(2))
		resolutions = self.f.require_dataset('/s' + str(self.idx).zfill(2) + '/resolutions', 
											 chunks = (res_np.shape), dtype = 'float64', 
											 shape = (res_np.shape), data = res_np)

		subdiv_np = numpy.zeros((len(res_list), 3), dtype = 'uint32')

		for z in range(len(res_list)-1, -1, -1):

			chunkSize1 = 256/self.binFactor
			chunkSize2 = 32/self.binFactor
			chunkSize3 = 256/self.binFactor

			res = res_list[z]

			subdiv_np[z, 0] = chunkSize1
			subdiv_np[z, 1] = chunkSize2
			subdiv_np[z, 2] = chunkSize3

			if z != 0:

				print('Writing resolution level ' + str(z))

				resgroup = self.f.create_group('/t00000/s' + str(self.idx).zfill(2) + '/' + str(z))

				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					img_3d_temp = self.img_3d[0::int(res), 0::int(res), 0::int(res)]

				data = self.f.require_dataset('/t00000/s' + str(self.idx).zfill(2) + '/' + str(z) + '/cells', 
											  chunks = (chunkSize1, chunkSize2, chunkSize3), dtype = 'int16', 
											  shape = img_3d_temp.shape, compression = "gzip")			
				data[:] = img_3d_temp

		subdivisions = self.f.require_dataset('/s' + str(self.idx).zfill(2) + '/subdivisions', 
											  chunks = (res_np.shape), dtype = 'uint32', 
											  shape = (subdiv_np.shape), data = subdiv_np)
		
		del self.img_3d
		del img_3d_temp
		
		gc.collect()

	def start(self):
		self.thread.start()

	def join(self):
		self.thread.join()

	def alive(self):
		flag = self.thread.isAlive()
		return flag

def write_xml(drive, save_dir, idx, idx_tile, idx_channel, channels = 1, tiles_y = 1, tiles_z = 1, 
			  sampling = 0.448, binning = 1, offset_y = 800, offset_z = 70, x = 1, y = 1, z = 1):

	# What is the sampling in the function definition?
	# TO DO: pass offset_y, offset_z as parameters
	
	print("Writing BigDataViewer XML file...")

	c = channels
	tx = tiles_y
	tz = tiles_z
	t = tx*tz
	sx = sampling
	binFactor = binning
	ox = offset_y*1000
	oz = offset_z*1000

	sx = sx
	sy = sx
	sz = sx

	f = open(drive + ':\\' + save_dir + '\\data.xml', 'w')
	f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
	f.write('<SpimData version="0.2">\n')
	f.write('\t<BasePath type="relative">.</BasePath>\n')
	f.write('\t<SequenceDescription>\n')
	f.write('\t\t<ImageLoader format="bdv.hdf5">\n')
	f.write('\t\t\t<hdf5 type="relative">data.h5</hdf5>\n')
	f.write('\t\t</ImageLoader>\n')
	f.write('\t\t<ViewSetups>\n')

	for i in range (0, c):
		for j in range(0, t):
			ind = j+i*c
			if ind <= idx:
				f.write('\t\t\t<ViewSetup>\n')
				f.write('\t\t\t\t<id>' + str(t*i+j) + '</id>\n')
				f.write('\t\t\t\t<name>' + str(t*i+j) + '</name>\n')
				f.write('\t\t\t\t<size>' + str(z) + ' ' + str(y) + ' ' + str(x) + '</size>\n')
				f.write('\t\t\t\t<voxelSize>\n')
				f.write('\t\t\t\t\t<unit>um</unit>\n')
				f.write('\t\t\t\t\t<size>' + str(sx) + ' ' + str(sy) + ' ' + str(sz) + '</size>\n')
				f.write('\t\t\t\t</voxelSize>\n')
				f.write('\t\t\t\t<attributes>\n')
				f.write('\t\t\t\t\t<illumination>0</illumination>\n')
				f.write('\t\t\t\t\t<channel>' + str(i) + '</channel>\n')
				f.write('\t\t\t\t\t<tile>' + str(j) + '</tile>\n')
				f.write('\t\t\t\t\t<angle>0</angle>\n')
				f.write('\t\t\t\t</attributes>\n')
				f.write('\t\t\t</ViewSetup>\n')

	f.write('\t\t\t<Attributes name="illumination">\n')
	f.write('\t\t\t\t<Illumination>\n')
	f.write('\t\t\t\t\t<id>0</id>\n')
	f.write('\t\t\t\t\t<name>0</name>\n')
	f.write('\t\t\t\t</Illumination>\n')
	f.write('\t\t\t</Attributes>\n')
	f.write('\t\t\t<Attributes name="channel">\n')

	for i in range(0, c):
		ind = i
		if ind <= idx_channel:
			f.write('\t\t\t\t<Channel>\n')
			f.write('\t\t\t\t\t<id>' + str(i) + '</id>\n')
			f.write('\t\t\t\t\t<name>' + str(i) + '</name>\n')
			f.write('\t\t\t\t</Channel>\n')

	f.write('\t\t\t</Attributes>\n')
	f.write('\t\t\t<Attributes name="tile">\n')

	for i in range(0, t):
		ind = i
		if ind <= idx_tile:
			f.write('\t\t\t\t<Tile>\n')
			f.write('\t\t\t\t\t<id>' + str(i) + '</id>\n')
			f.write('\t\t\t\t\t<name>' + str(i) + '</name>\n')
			f.write('\t\t\t\t</Tile>\n')

	f.write('\t\t\t</Attributes>\n')
	f.write('\t\t\t<Attributes name="angle">\n')
	f.write('\t\t\t\t<Illumination>\n')
	f.write('\t\t\t\t\t<id>0</id>\n')
	f.write('\t\t\t\t\t<name>0</name>\n')
	f.write('\t\t\t\t</Illumination>\n')
	f.write('\t\t\t</Attributes>\n')
	f.write('\t\t</ViewSetups>\n')
	f.write('\t\t<Timepoints type="pattern">\n')
	f.write('\t\t\t<integerpattern>0</integerpattern>')
	f.write('\t\t</Timepoints>\n')
	f.write('\t\t<MissingViews />\n')
	f.write('\t</SequenceDescription>\n')

	f.write('\t<ViewRegistrations>\n')
	for i in range(0, c):
		for j in range(0, tz):
			for k in range(0, tx):

				ind = i*tz*tx + j*tx + k

				if ind <= idx:
					
					#transy = -y*j # DPS : swapped y -> x
					transx = -x*j
					
					#transz = -y/math.sqrt(2.0)*j # DPS : swapped y -> x
					transz = -x/math.sqrt(2.0)*j

					#shiftx = (ox/sx)*k # DPS: swapped x -> y
					shiftx = (x/math.sqrt(2.0)-((oz/sz))*j
					
					#shifty = (y/math.sqrt(2.0)-(oz/sz))*j # DPS: swapped y -> x
					shifty = (oy/sy)*k

					f.write('\t\t<ViewRegistration timepoint="0" setup="' + str(ind) + '">\n')
					f.write('\t\t\t<ViewTransform type="affine">\n')
					f.write('\t\t\t\t<Name>Overlap</Name>\n')
					# DPS: overlap matrix doesn't need to be altered
					f.write('\t\t\t\t<affine>1.0 0.0 0.0 ' + str(shiftx) + ' 0.0 1.0 0.0 ' + str(shifty) + ' 0.0 0.0 1.0 0.0</affine>\n')
					f.write('\t\t\t</ViewTransform>\n')
					f.write('\t\t\t<ViewTransform type="affine">\n')
					f.write('\t\t\t\t<Name>Deskew</Name>\n')
					# DPS: Deskew matrix needs to be altered
					#f.write('\t\t\t\t<affine>1.0 0.0 0.0 0.0 0.0 0.7071 0.0 0.0 0.0 -0.7071 1.0 0.0</affine>\n')
					f.write('\t\t\t\t<affine>1.0 0.7071 0.0 0.0 0.0 0.0 0.0 0.0 -0.7071 0.0 1.0 0.0</affine>\n')
					f.write('\t\t\t</ViewTransform>\n')
					f.write('\t\t\t<ViewTransform type="affine">\n')
					f.write('\t\t\t\t<Name>Translation to Regular Grid</Name>\n')
					# DPS: Translation to Regular Grid matrix needs to be altered
					#f.write('\t\t\t\t<affine>1.0 0.0 0.0 0.0 0.0 1.0 0.0 ' + str(transy) + ' 0.0 0.0 1.0 ' + str(transz) + '</affine>\n')
					f.write('\t\t\t\t<affine>1.0 0.0 0.0 ' + str(transy) + ' 0.0 1.0 0.0 0.0 0.0 0.0 1.0 ' + str(transz) + '</affine>\n')
					f.write('\t\t\t</ViewTransform>\n')
					f.write('\t\t</ViewRegistration>\n')

	f.write('\t</ViewRegistrations>\n')
	f.write('\t<ViewInterestPoints />\n')
	f.write('\t<BoundingBoxes />\n')
	f.write('\t<PointSpreadFunctions />\n')
	f.write('\t<StitchingResults />\n')
	f.write('\t<IntensityAdjustments />\n')
	f.write('</SpimData>')
	f.close()

#
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
#