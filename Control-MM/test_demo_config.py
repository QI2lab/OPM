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

    print('post camera hook fn.')
    print('fired stage scan')

    return event
    
def setup_scan_fn(event,bridge,event_queue):

    core = bridge.get_core()

    print('post hardware hook fn.')
    print('verified that stage controller is ready')

    return event

def main():

    bridge = Bridge()
    core = bridge.get_core()

    # FOV parameters
    ROI = [1024, 0, 256, 1024] #unit: pixels

    # camera exposure
    exposure_ms = 5 #unit: ms

    # set to high-res camera
    core.set_config('Camera','HighRes')

    # crop FOV
    core.set_roi(*ROI)

    # set exposure
    core.set_exposure(exposure_ms)

    # setup file name
    save_directory=Path('C:/data/test/')
    save_name = 'test_stages'

    # get handle to xy and z stages
    xy_stage = core.get_xy_stage_device()
    z_stage = core.get_focus_device()

    # move the stages to verify core can talk to them
    # positions chosen at random
    core.set_xy_position(100.,100.)
    core.wait_for_device(xy_stage)
    core.set_position(50.)
    core.wait_for_device(z_stage)

    # create events to hold all of the scan axis images during constant speed stage scan
    channel_configs = ['DAPI','FITC','Rhodamine','Cy5']
    events = []
    for y in range(2):
        for z in range(2):
            for c in range(len(channel_configs)):
                for x in range(2):
                        evt = { 'axes': {'x': x, 'y': y, 'z': z}, 'x': 100, 'y': y*1000, 'z': z*100, 'channel': {'group': 'Channel', 'config': channel_configs[c]}}
                        events.append(evt)

    # run acquisition
    # TO DO: properly handle an error here if camera driver fails to return expected number of images.
    with Acquisition(directory=save_directory, name=save_name, post_hardware_hook_fn=setup_scan_fn, 
                    post_camera_hook_fn=hook_fn, show_display=False, max_multi_res_index=0, debug=False) as acq:
        acq.acquire(events)
        acq.acquire(None)
        acq.await_completion()

# run
if __name__ == "__main__":
    main()