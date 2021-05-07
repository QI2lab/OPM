#!/usr/bin/env python

import numpy as np 
from pycromanager import Acquisition

events = []
for index, z_um in enumerate(np.arange(start=0, stop=10, step=0.5)):
    evt = {
            #'axes' is required. It is used by the image viewer and data storage to
            #identify the acquired image
            'axes': {'z': index},

            #the 'z' field provides the z position in µm
            'z': z_um}
    events.append(evt)

for i in range(5):
  with Acquisition('c:/test/', 'testacq') as acq:
      acq.acquire(events)
  acq = None