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
from pymmcore_widgets import StageWidget
from pathlib import Path

def main(path_to_mm_config=Path(r'C:\Users\qi2lab\Documents\micro-manager_configs\OPM_20230320.cfg')):

    instrument_control_widget = OPMMirrorScan()
    # setup OPM GUI and Napari viewer
    instrument_control_widget.mmc.loadSystemConfiguration(str(path_to_mm_config)) 
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

    ao_worker_3d = instrument_control_widget._optimize_AO_3d()
    ao_worker_3d.yielded.connect(instrument_control_widget._update_layers)
    instrument_control_widget._set_ao_worker_3d(ao_worker_3d)

    instrument_control_widget._create_3d_t_worker()

    viewer.window.add_dock_widget(instrument_control_widget,name='Instrument control')
    
    stage_03 = StageWidget('MCL NanoDrive Z Stage',step=.1)
    viewer.window.add_dock_widget(stage_03,name='O3 Zstage')
    stage_01 = StageWidget('ZStage:M:37',step=10)
    viewer.window.add_dock_widget(stage_01,name='O1 Zstage')
    stage_xy = StageWidget('XYStage:XY:31',step=100)
    viewer.window.add_dock_widget(stage_xy,name='XYstage')

    # start Napari
    napari.run()

    # shutdown acquistion threads
    worker_2d.quit()
    worker_3d.quit()

    # shutdown instrument
    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    instrument_control_widget._shutdown()

if __name__ == "__main__":
    main()