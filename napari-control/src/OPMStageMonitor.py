# napari imports
from magicclass import magicclass, set_design
from magicgui import magicgui, widgets

#general python imports
from pymmcore_plus import RemoteMMCore

# Stage monitor class
@magicclass(labels=False)
@set_design(text="Stage monitor")
class OPMStageMonitor:

    def __init__(self):
        pass

    x_stage_pos = widgets.LineEdit(label='x:', value=f"{0:.1f}")
    y_stage_pos = widgets.LineEdit(label='y:', value=f"{0:.1f}")
    z_stage_pos = widgets.LineEdit(label='z:', value=f"{0:.1f}")

    @magicgui(
        auto_call=True,
        get_pos_xyz={"widget_type": "PushButton", "label": 'Get stage position'},
        layout='horizontal'
    )
    def get_stage_pos(self, get_pos_xyz):

        with RemoteMMCore() as mmc_stage_monitor:
            x = mmc_stage_monitor.getXPosition()
            y = mmc_stage_monitor.getYPosition()
            z = mmc_stage_monitor.getZPosition()

            self.x_stage_pos.value = (f"{x:.1f}")
            self.y_stage_pos.value = (f"{y:.1f}")
            self.y_stage_pos.value = (f"{z:.1f}")