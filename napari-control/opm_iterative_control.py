#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
ASU OPM with iterative fluidics control via Napari, magic-class, and magic-gui
----------------------------------------------------------------------------------------
Peter Brown
Franky Djutanta
Douglas Shepherd
12/11/2021
douglas.shepherd@asu.edu
----------------------------------------------------------------------------------------
'''

from src.OPMIterative import OPMIterative
from src.OPMStageMonitor import OPMStageMonitor
from src.OPMStageScan import OPMStageScan
import napari
from pathlib import Path
from pymmcore_plus import RemoteMMCore

def main(path_to_mm_config=Path('C:/Program Files/Micro-Manager-2.0gamma/temp_HamDCAM.cfg')):

    # launch pymmcore server
    with RemoteMMCore() as mmc:
        mmc.loadSystemConfiguration(str(path_to_mm_config))

        # create Napari viewer
        viewer = napari.Viewer(title='ASU OPM control -- iterative multiplexing')
        
        # setup OPM widgets     
        iterative_control_widget = OPMIterative()
        stage_display_widget = OPMStageMonitor()
        instrument_control_widget = OPMStageScan(iterative_control_widget)
        instrument_control_widget._set_viewer(viewer)
        
        # startup instrument
        instrument_control_widget._startup()

        # create thread workers
        # 2D
        worker_2d = instrument_control_widget._acquire_2d_data()
        worker_2d.yielded.connect(instrument_control_widget._update_layers)
        instrument_control_widget._set_worker_2d(worker_2d)
        
        # 3D
        worker_3d = instrument_control_widget._acquire_3d_data()
        worker_3d.yielded.connect(instrument_control_widget._update_layers)
        instrument_control_widget._set_worker_3d(worker_3d)

        # iterative 3D
        instrument_control_widget._create_worker_iterative()

        # instrument setup
        worker_iterative_setup = iterative_control_widget._return_experiment_setup()
        worker_iterative_setup.returned.connect(instrument_control_widget._set_iterative_configuration)
        iterative_control_widget._set_worker_iterative_setup(worker_iterative_setup)
        
        # add widgets to Napari viewer
        viewer.window.add_dock_widget(iterative_control_widget,area='bottom',name='Iterative setup')
        viewer.window.add_dock_widget(stage_display_widget,area='bottom',name='Stage monitor')
        viewer.window.add_dock_widget(instrument_control_widget,area='right',name='Instrument setup')

        # start Napari
        napari.run(max_loop_level=2)

if __name__ == "__main__":
    main()