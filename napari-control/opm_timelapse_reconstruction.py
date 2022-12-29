import napari
from src.OPMMirrorReconstruction import OPMMirrorReconstruction

def main():

    # setup OPM GUI and Napari viewer
    reconstruction_widget = OPMMirrorReconstruction()
    viewer = napari.Viewer()

    # these methods have to be private to not show using magic-class. Maybe a better solution is available?
    reconstruction_widget._set_viewer(viewer)

    # create processing worker
    reconstruction_widget._create_processing_worker()

    viewer.window.add_dock_widget(reconstruction_widget,name='ASU Snouty-OPM timelapse reconstruction')

    # start Napari
    napari.run()

if __name__ == "__main__":
    main()