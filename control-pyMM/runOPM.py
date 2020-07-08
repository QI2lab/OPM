# 

from pycromanager import Bridge
import npy2bdv

# need to import fluidic drivers
# create separate python file with function to talk to ASI tiger controller (for timeout protection)

def save_bdv(image,metadata):

    # grab metadata from this image

    # redirect to BDV

    # write affine transformation for this position

    pass

def run_fluidics(fluidics_protocol):

    # parse structure to determine protocol

    pass


def run_acquisition(file_path,fluidics_settings=None):

    # check if this run is using fluidics

    # step through fluidics protocol until imaging

    # fluidics finished, run imaging

    # setup tiling parameters

    # setup stage scan parameters

    # 

    with Acquisition(image_process_fn=writeBDV) as acq:
        # setup acquisition
        events = multi_d_acquistion_events(num_images)

        # setup camera parameters

        # setup ASI stage parameters

        # setup ASI PLC parameters

        # setup ASI autofocus parameters

        # start camera

        # start acquisiton

    pass

def setup_acquisition():

    # have user set corners using joystick and keyboard

    return scan_corners


def main():

    bridge = Bridge()
    bridge.get_core()

if __name__ == "__main__":
    main()