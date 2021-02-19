#!/usr/bin/env python

'''
Testing

Shepherd 01/21
'''
# imports
from pycromanager import Bridge, Acquisition
from pathlib import Path
import time

def camera_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()

    return event
    
def post_hook_fn(event,bridge,event_queue):

    core = bridge.get_core()
    
    return event


def main():

    bridge = Bridge()
    core  = bridge.get_core()

    # create ROI similar to sCMOS for fast readout
    core.set_property('Camera','OnCameraCCDXSize',2048)
    core.set_property('Camera','OnCameraCCDYSize',2048)
    core.set_property('Camera','FastImage',1)
    time.sleep(10)

    # define channels to use
    channel_states = [1,0,1,1]

    # define dimensions of scan
    number_y = 10
    number_z = 3
    number_scan = 1000

    # exposure time
    exposure_ms = 15.0

    # setup file name
    save_directory=Path('e:/test_20210209/')
    save_name = 'test'

    # set exposure
    core.set_exposure(exposure_ms)

    # create events to execute scan
    events = []
    
    total = number_y*number_z*3*number_scan
    
    for i in range(total):
        evt = { 'axes': {'z': i, }, 'channel' : {'group': 'Channel', 'config': 'DAPI'}}
        events.append(evt)

    
    '''
    for y in range(number_y):
        for z in range(number_z):
            for c in range(len(channel_states)):
                for x in range(number_scan):
                    if channel_states[c]==1:
                        if (c==0):
                            evt = { 'axes': {'z': x, 'y': y, 'z': z}, 'x': 0, 'y': y*100, 'z': z * 10,
                                    'channel' : {'group': 'Channel', 'config': 'DAPI'}}
                            events.append(evt)
                        elif (c==1):
                            evt = { 'axes': {'z': x, 'y': y, 'z': z}, 'x': 0, 'y': y*100, 'z': z * 10,
                                    'channel' : {'group': 'Channel', 'config': 'FITC'}}
                            events.append(evt)
                        elif (c==2):
                            evt = { 'axes': {'z': x, 'y': y, 'z': z}, 'x': 0, 'y': y*100, 'z': z * 10,
                                    'channel' : {'group': 'Channel', 'config': 'Rhodamine'}}
                            events.append(evt)
                        elif (c==3):
                            evt = { 'axes': {'z': x, 'y': y, 'z': z}, 'x': 0, 'y': y*100, 'z': z * 10,
                                    'channel' : {'group': 'Channel', 'config': 'Cy5'}}
                            events.append(evt)
    '''
    # run acquisition
    with Acquisition(directory=save_directory, name=save_name, image_process_fn=None, event_generation_hook_fn=None, 
                             pre_hardware_hook_fn=None, post_hardware_hook_fn=post_hook_fn, post_camera_hook_fn=camera_hook_fn, 
                             show_display=False, tile_overlap=None, max_multi_res_index=None, magellan_acq_index=None, 
                             magellan_explore=False, process=False, saving_queue_size = 10000, debug=False,
                             ifd_off=True, metadata_off=True, omit_index=True) as acq:
        acq.acquire(events)
        
# run
if __name__ == "__main__":
    main()