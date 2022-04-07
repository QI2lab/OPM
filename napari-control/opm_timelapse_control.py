#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
ASU OPM timelapse via pymmcore-plus, napari, magic-class, and magic-gui
----------------------------------------------------------------------------------------
Peter Brown
Franky Djutanta
Douglas Shepherd
12/11/2021
douglas.shepherd@asu.edu
----------------------------------------------------------------------------------------
'''

from src.OPMMirrorScan import OPMMirrorScan
import napari
from pathlib import Path
from pymmcore_plus import CMMCorePlus

# I just replaced RemoteMMCore by CMMCorePlus here and in OPMMirrorScan.py
# maybe we need / can delete the several with clauses
# we could also use the same mmc variable to control the lasers and camera 
# instead of creating another MMCore-like object
# that may require to pass mmc as an atribute to instrument_control_widget
# but let's first test the initial creation of multiple instances of  CMMCorePlus
# to see if it bugs

def main(path_to_mm_config=Path('C:/Program Files/Micro-Manager-2.0gamma/temp_HamDCAM.cfg')):

     # launch pymmcore server
    with CMMCorePlus() as mmc:
        mmc.loadSystemConfiguration(str(path_to_mm_config))# setup OPM GUI and Napari viewer

        instrument_control_widget = OPMMirrorScan()
        # these methods have to be private to not show using magic-class. Maybe a better solution is available?
        instrument_control_widget._startup()

        viewer = napari.Viewer(title='ASU Snouty-OPM timelapse acquisition control')

        # these methods have to be private to not show using magic-class. Maybe a better solution is available?
        instrument_control_widget._set_viewer(viewer)

        # setup 2D imaging thread worker
        # these methods have to be private to not show using magic-class. Maybe a better solution is available?
        worker_2d = instrument_control_widget._acquire_2d_data()
        worker_2d.yielded.connect(instrument_control_widget._update_layers)
        instrument_control_widget._set_worker_2d(worker_2d)
        
        # setup 3D imaging thread worker 
        # these methods have to be private to not show using magic-class. Maybe a better solution is available?
        worker_3d = instrument_control_widget._acquire_3d_data()
        worker_3d.yielded.connect(instrument_control_widget._update_layers)
        instrument_control_widget._set_worker_3d(worker_3d)

        instrument_control_widget._create_3d_t_worker()

        viewer.window.add_dock_widget(instrument_control_widget,name='Instrument control')

        # start Napari
        napari.run(max_loop_level=2)

        # shutdown acquistion threads
        worker_2d.quit()
        worker_3d.quit()

        # shutdown instrument
        # these methods have to be private to not show using magic-class. Maybe a better solution is available?
        instrument_control_widget._shutdown()

if __name__ == "__main__":
    main()