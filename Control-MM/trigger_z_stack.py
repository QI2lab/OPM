import numpy as np

from pycromanager import Bridge
from pycromanager import Acquisition
from pathlib import Path

def upload_stage_sequence(bridge, start_end_pos, mid_pos, step_size, relative=True):
    """
    Upload a triangle waveform of z_stage positions and set the z_stage in UseFastSequence mode

    :param bridge: pycro-manager java bridge
    :type bridge: pycromanager.core.Bridge
    :param start_end_pos: start and end position of triangle waveform
    :type start_end_pos: float
    :param mid_pos: mid position of the triangle waveform
    :type mid_pos: float
    :param step_size: z_stage step size
    :type step_size: float
    :param relative: set to False if given start_end_pos and mid_pos are absolute

    :return: current z position and absolute positions of triangle waveform
    """

    mmc = bridge.get_core()

    z_stage = mmc.get_focus_device()

    pos_sequence = np.hstack((np.arange(start_end_pos, mid_pos + step_size, step_size),
                              np.arange(mid_pos, start_end_pos - step_size, -step_size))).astype('float64')

    z_pos = 0
    if relative:
        z_pos = mmc.get_position(z_stage)
        print(z_pos)
        pos_sequence += z_pos

    # construct java object
    positionJ = bridge.construct_java_object('mmcorej.DoubleVector')
    for i in pos_sequence:
        positionJ.add(float(i))

    # send sequence to stage
    mmc.set_property(z_stage, "UseSequence", "Yes")
    mmc.set_property(z_stage, "UseFastSequence", "No")
    mmc.load_stage_sequence(z_stage, positionJ)
    mmc.set_property(z_stage, "UseFastSequence", "Armed")

    return z_pos, pos_sequence


def main():

    bridge = Bridge()
    mmc = bridge.get_core()
    
    # Data set parameters
    path = Path('E://20201023//')
    name = 'test'

    # z stack parameters
    start_end_pos = -5
    mid_pos = 5
    step_size = .25
    relative = True

    # time series parameters
    exposure_time = 200  # in milliseconds
    
    num_z_positions = int(abs(mid_pos - start_end_pos)/step_size + 1)
    z_idx = list(range(num_z_positions))
    num_time_points = 10

    # setup cameras
    mmc.set_exposure(exposure_time)
    
    # setup z stage
    z_stage = mmc.get_focus_device()
    z_pos, pos_sequence = upload_stage_sequence(bridge, start_end_pos, mid_pos, step_size, relative)
    num_z_positions = len(pos_sequence)

    print(pos_sequence)

    # move to first position
    mmc.set_position(z_stage, pos_sequence[0])

    events = []
    z_idx_ = z_idx.copy()
    for i in range(num_time_points):
        for j in z_idx_:
            events.append({'axes': {'time':i, 'z': j}})
        z_idx_.reverse()

    with Acquisition(directory=path, name=name) as acq:
        acq.acquire(events)

    # turn off sequencing
    mmc.set_property(z_stage, "UseFastSequence", "No")
    mmc.set_property(z_stage, "UseSequence", "No")

    # move back to initial position
    #mmc.set_position(z_stage, z_pos)

# run
if __name__ == "__main__":
    main()