#!/usr/bin/python
"""
BDV and BigStitcher H5 writing

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
		#self.thread.daemon = True

	def run(self):

		res_list = (1, 2, 4, 8)

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

				data = f.require_dataset('/t00000/s' + str(idx).zfill(2) + '/' + str(z) + '/cells', 
										 chunks = (chunkSize1, chunkSize2, chunkSize3), dtype = 'int16', 
										 shape = img_3d_temp.shape)
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