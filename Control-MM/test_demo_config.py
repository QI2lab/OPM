#!/usr/bin/env python

'''
OPM stage control
Stripped down example using demo config.

Shepherd 10/20
'''

# imports
from pycromanager import Bridge, Acquisition
from pathlib import Path


def hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    print('camera')
    print(event)

    return event

def setup_scan_fn(event,bridge,event_queue):

    core = bridge.get_core()

    print('stage')
    print(event)

    return event

def main():

    bridge = Bridge()
    core = bridge.get_core()

    # FOV parameters
    ROI = [1024, 0, 256, 1024] #unit: pixels

    # camera exposure
    exposure_ms = 10 #unit: ms

    # set to high-res camera
    core.set_config('Camera','HighRes')

    # crop FOV
    core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # setup file name
    save_directory=Path('F:/data/test/')
    save_name = 'scan_'+'y_'+str(0).zfill(4)+'x_'+str(0).zfill(4)+'c_'+str(0).zfill(2)

    # create events to hold all of the scan axis images during constant speed stage scan
    # we call this 'z' here, even though it is actually oblique images acquired by moving scan axis (x) in our system
    events = []
    for y in range(2):
        for z in range(10):
            evt = { 'axes': {'y': y, 'z': z}, 'y': y}
            events.append(evt)

    # run acquisition
    # TO DO: properly handle an error here if camera driver fails to return expected number of images.
    with Acquisition(directory=save_directory, name=save_name, post_hardware_hook_fn=setup_scan_fn, post_camera_hook_fn=hook_fn, show_display=True, max_multi_res_index=0) as acq:
        acq.acquire(events)

# run
if __name__ == "__main__":
    main()