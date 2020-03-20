#!/usr/bin/env python
"""
rclone control

# Adam Glaser 07/19

"""

import subprocess
import sys
import time
import os
import threading

class rcloneUpload(object):

	def __init__(self, drive, fname, container): 

		self.drive = drive
		self.fname = fname
		self.container = container
		self.thread = threading.Thread(target=self.run, args=())

	def run(self):

		subprocess.Popen(['rclone', 'copy', self.drive + ':\\' + self.fname, self.container + ':' + self.fname, '--ignore-existing', '--stats', '10s','-vv','--transfers',  '4'])

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
#
