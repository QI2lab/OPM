# napari imports
from magicclass import magicclass, set_design, MagicTemplate, set_options
from magicgui import magicgui, widgets
from napari.qt.threading import thread_worker

#general python imports
from pathlib import Path
from pymmcore_plus import RemoteMMCore
import numpy as np
from distutils.util import strtobool

# ASU OPM imports
import src.utils.data_io as data_io
from src.utils.fluidics_control import run_fluidic_program
from src.hardware.APump import APump
from src.hardware.HamiltonMVP import HamiltonMVP


# Fluidics loader, exposure, channels, and stage scan definitions
@magicclass(labels=False,widget_type="tabbed")
@set_design(text="Iterative scanning")
class OPMIterative(MagicTemplate):

    # initialize
    def __init__(self):

        self.fluidics_file_path = None
        self.fluidics_program = None
        self.n_iterative_rounds = 0
        self.fluidics_loaded = False
        
        self.scan_axis_step_um = 0.400      # unitL um
        self.scan_axis_start_um  = 0    # unit: um
        self.scan_axis_end_um  = 0       # unit: um
        self.scan_axis_positions = 0     # positions
        self.scan_axis_speed_readout = 0 # mm/s
        self.scan_axis_speed_nuclei = 0  # mm/s

        self.tile_axis_start_um = 0      # unit: um
        self.tile_axis_end_um = 0        # unit: um
        self.tile_axis_step_um = 0       # unit: um
        self.tile_axis_positions = 0

        self.height_axis_start_um = 0    # unit: um
        self.height_axis_end_um = 0      # unit: um
        self.height_axis_step_um = 0     # unit: um
        self.height_axis_positions = 0

        self.n_xy_tiles = 0              # number of xy tiles
        self.n_z_tiles = 0               # number of z tiles
        self.stage_volume_set = False

        self.channel_states_readout = [False,False,False,False,False]
        self.channel_states_nuclei = [False,False,False,False,False]
        self.channel_powers = [0.0,0.0,0.0,0.0,0.0]
        self.n_active_channels_readout = 0
        self.n_active_channels_nuclei = 0
        self.exposure_ms = 10.0
        self.pixel_size_um = 0.115
        self.channels_set = False
        
        self.setup_complete = False

        self.debug = False

    # thread worker for cross-class communication of setup
    def _set_worker_iterative_setup(self,worker_iterative_setup):
        self.worker_iterative_setup = worker_iterative_setup

    # calculate scan volume, pulling ROI from camera settings
    def _calculate_scan_volume(self):

        try:
            with RemoteMMCore() as mmc_stage_setup:

                # set experiment exposure
                mmc_stage_setup.setExposure(self.exposure_ms)

                # snap image
                mmc_stage_setup.snapImage()

                # grab exposure
                true_exposure = mmc_stage_setup.getExposure()

                # grab ROI
                _, _, self.y_pixels, self.x_pixels = mmc_stage_setup.getROI()

                if not((self.y_pixels == 256) or (self.y_pixels==512)):
                    raise Exception('Set camera ROI first.')

                # get actual framerate from micromanager properties
                actual_readout_ms = true_exposure+float(mmc_stage_setup.getProperty('OrcaFusionBT','ReadoutTime')) #unit: ms
                if self.debug: print('Full readout time = ' + str(actual_readout_ms))

                # scan axis setup
                scan_axis_step_mm = self.scan_axis_step_um / 1000. #unit: mm
                self.scan_axis_start_mm = self.scan_axis_start_um / 1000. #unit: mm
                self.scan_axis_end_mm = self.scan_axis_end_um / 1000. #unit: mm
                scan_axis_range_um = np.abs(self.scan_axis_end_um-self.scan_axis_start_um)  # unit: um
                self.scan_axis_range_mm = scan_axis_range_um / 1000 #unit: mm
                actual_exposure_s = actual_readout_ms / 1000. #unit: s
                self.scan_axis_speed_readout = np.round(scan_axis_step_mm / actual_exposure_s / self.n_active_channels_readout,5) #unit: mm/s
                self.scan_axis_speed_nuclei = np.round(scan_axis_step_mm / actual_exposure_s / self.n_active_channels_nuclei,5) #unit: mm/s
                self.scan_axis_positions = np.rint(self.scan_axis_range_mm / scan_axis_step_mm).astype(int)  #unit: number of positions

                # tile axis setup
                tile_axis_overlap=0.2 #unit: percentage
                tile_axis_range_um = np.abs(self.tile_axis_end_um - self.tile_axis_start_um) #unit: um
                tile_axis_ROI = self.x_pixels*self.pixel_size_um  #unit: um
                self.tile_axis_step_um = np.round((tile_axis_ROI) * (1-tile_axis_overlap),2) #unit: um
                self.n_xy_tiles = np.rint(tile_axis_range_um / self.tile_axis_step_um).astype(int)+1  #unit: number of positions
                # if tile_axis_positions rounded to zero, make sure we acquire at least one position
                if self.n_xy_tiles == 0:
                    self.n_xy_tiles=1

                # height axis setup
                # check if there are multiple heights
                height_axis_range_um = np.abs(self.height_axis_end_um-self.height_axis_start_um) #unit: um
                # if multiple heights, check if heights are due to uneven tissue position or for z tiling
                height_axis_overlap=0.2 #unit: percentage
                height_axis_ROI = self.y_pixels*self.pixel_size_um*np.sin(30.*np.pi/180.) #unit: um 
                self.height_axis_step_um = np.round((height_axis_ROI)*(1-height_axis_overlap),2) #unit: um
                self.n_z_tiles = np.rint(height_axis_range_um / self.height_axis_step_um).astype(int)+1 #unit: number of positions
                # if height_axis_positions rounded to zero, make sure we acquire at least one position
                if self.n_z_tiles==0:
                    self.n_z_tiles=1

                # create dictionary with scan settings
                self.scan_settings = [{'exposure_ms': float(self.exposure_ms),
                                    'scan_axis_start_um': float(self.scan_axis_start_um),
                                    'scan_axis_end_um': float(self.scan_axis_end_um),
                                    'scan_axis_step_um': float(self.scan_axis_step_um),
                                    'tile_axis_start_um': float(self.tile_axis_start_um),
                                    'tile_axis_end_um': float(self.tile_axis_end_um),
                                    'tile_axis_step_um': float(self.tile_axis_step_um),
                                    'height_axis_start_um': float(self.height_axis_start_um),
                                    'height_axis_end_um': float(self.height_axis_end_um),
                                    'height_axis_step_um': float(self.height_axis_step_um),
                                    'n_iterative_rounds': int(self.n_iterative_rounds),
                                    'nuclei_round': int(self.codebook['nuclei_round']),
                                    'num_xy_tiles': int(self.n_xy_tiles),
                                    'num_z_tiles': int(self.n_z_tiles),
                                    'num_ch_readout': int(self.n_active_channels_readout),
                                    'num_ch_nuclei': int(self.n_active_channels_nuclei),
                                    'scan_axis_positions': int(self.scan_axis_positions),
                                    'scan_axis_speed_readout': float(self.scan_axis_speed_readout),
                                    'scan_axis_speed_nuclei': float(self.scan_axis_speed_nuclei),
                                    'y_pixels': int(self.y_pixels),
                                    'x_pixels': int(self.x_pixels),
                                    '405_active_readout': bool(self.channel_states_readout[0]),
                                    '488_active_readout': bool(self.channel_states_readout[1]),
                                    '561_active_readout': bool(self.channel_states_readout[2]),
                                    '635_active_readout': bool(self.channel_states_readout[3]),
                                    '730_active_readout': bool(self.channel_states_readout[4]),
                                    '405_power_readout': float(self.channel_powers_readout[0]),
                                    '488_power_readout': float(self.channel_powers_readout[1]),
                                    '561_power_readout': float(self.channel_powers_readout[2]),
                                    '635_power_readout': float(self.channel_powers_readout[3]),
                                    '730_power_readout': float(self.channel_powers_readout[4]),
                                    '405_active_nuclei': bool(self.channel_states_nuclei[0]),
                                    '488_active_nuclei': bool(self.channel_states_nuclei[1]),
                                    '561_active_nuclei': bool(self.channel_states_nuclei[2]),
                                    '635_active_nuclei': bool(self.channel_states_nuclei[3]),
                                    '730_active_nuclei': bool(self.channel_states_nuclei[4]),
                                    '405_power_nuclei': float(self.channel_powers_nuclei[0]),
                                    '488_power_nuclei': float(self.channel_powers_nuclei[1]),
                                    '561_power_nuclei': float(self.channel_powers_nuclei[2]),
                                    '635_power_nuclei': float(self.channel_powers_nuclei[3]),
                                    '730_power_nuclei': float(self.channel_powers_nuclei[4])}]

                self.stage_volume_set = True
        except:
            raise Exception("Error in stage volume setup.")

    # load fluidics and codebook files
    def _load_fluidics(self):
        try:
            self.df_fluidics = data_io.read_fluidics_program(self.fluidics_file_path)
            self.codebook = data_io.read_config_file(self.codebook_file_path)
            self.fluidics_loaded = True
        except:
            raise Exception('Error in loading fluidics and/or codebook files.')
        
    # generate summary of fluidics and codebook files
    def _generate_fluidics_summary(self):

        self.n_iterative_rounds = int(self.codebook['n_rounds'])

        self.n_active_channels_readout = int(self.codebook['dyes_per_round'])
        self.channel_states_readout = [
            False,
            bool(strtobool(self.codebook['alexa488'])),
            bool(strtobool(self.codebook['atto565'])),
            bool(strtobool(self.codebook['alexa647'])),
            bool(strtobool(self.codebook['cy7']))]

        if not(self.codebook['nuclei_round']==-1):
            self.n_active_channels_nuclei = 2
            self.channel_states_nuclei = [
                True,
                True,
                False,
                False,
                False]

        fluidics_data = (f"Experiment type: {str(self.codebook['type'])} \n"
                         f"Number of iterative rounds: {str(self.codebook['n_rounds'])} \n\n"
                         f"Number of targets: {str(self.codebook['targets'])} \n"
                         f"Channels per round: {str(self.codebook['dyes_per_round'])} \n"
                         f"Alexa488 fidicual: {str(self.codebook['alexa488'])} \n"
                         f"Atto565 readout: {str(self.codebook['atto565'])} \n"
                         f"Alexa647 readout: {str(self.codebook['alexa647'])} \n"
                         f"Cy7 readout: {str(self.codebook['cy7'])} \n"
                         f"Nuclear marker round: {str(self.codebook['nuclei_round'])} \n\n")
        self.fluidics_summary.value = fluidics_data

    # generate summary of experimental setup
    def _generate_experiment_summary(self):

        exp_data = (f"Number of iterative rounds: {str(self.n_iterative_rounds)} \n\n"
                    f"Scan start: {str(self.scan_axis_start_um)}  \n"
                    f"Scan end:  {str(self.scan_axis_end_um)} \n"
                    f"Number of scan positions:  {str(self.scan_axis_positions)} \n"
                    f"Readout rounds scan speed:  {str(self.scan_axis_speed_readout)} \n"
                    f"Nuclei round scan speed:  {str(self.scan_axis_speed_nuclei)} \n\n"
                    f"Number of Y tiles:  {str(self.n_xy_tiles)} \n"
                    f"Tile start:  {str(self.tile_axis_start_um)} \n"
                    f"Tile end:  {str(self.tile_axis_end_um)} \n"
                    f"Tile step:  {str(self.tile_axis_step_um)} \n\n"
                    f"Number of Z slabs:  {str(self.n_z_tiles)} \n"
                    f"Height start:  {str(self.height_axis_start_um)} \n"
                    f"Height end:  {str(self.height_axis_end_um)} \n"
                    f"Height step:  {str(self.height_axis_step_um)} \n\n"
                    f"--------Readout rounds------- \n"
                    f"Number of channels:  {str(self.n_active_channels_readout)} \n"
                    f"Active lasers: {str(self.channel_states_readout)} \n"
                    f"Lasers powers: {str(self.channel_powers_readout)} \n\n"
                    f"--------Nuclei rounds------- \n"
                    f"Number of channels: {str(self.n_active_channels_nuclei)} \n"
                    f"Active lasers: {str(self.channel_states_nuclei)} \n"
                    f"Lasers powers: {str(self.channel_powers_nuclei)} \n\n")
        self.experiment_summary.value = exp_data

    @magicgui(
        auto_call=False,
        fluidics_file_path={"widget_type": "FileEdit", 'label': 'Fluidics program'},
        codebook_file_path={"widget_type": "FileEdit", 'label': 'Codebook'},
        layout='vertical',
        call_button='Load fluidics'
    )
    def load_fluidics_program(self, fluidics_file_path: Path, codebook_file_path: Path):
        self.fluidics_file_path = fluidics_file_path
        self.codebook_file_path = codebook_file_path
        self._load_fluidics()
        self._generate_fluidics_summary()

    @magicgui(
        auto_call=True,
        fluidics_file_path={"widget_type": "PushButton", 'label': 'Run first round'},
        layout='vertical'
    )
    def run_first_fluidics_round(self, fluidics_file_path: Path, codebook_file_path: Path):
        if self.fluidics_loaded:
            # connect to pump
            self.pump_controller = APump(self.pump_parameters)
            # set pump to remote control
            self.pump_controller.enableRemoteControl(True)

            # connect to valves
            self.valve_controller = HamiltonMVP(com_port=self.valve_COM_port)
            # initialize valves
            self.valve_controller.autoAddress()

            # run fluidics flush
            success_fluidics = False          
            success_fluidics = run_fluidic_program(0,self.df_fluidics,self.valve_controller,self.pump_controller)
            if not(success_fluidics):
                raise Exception('Error in fluidics unit.')
            else:
                self.first_round_run = True
        else:
            raise Exception('Configure fluidics first.')

    @magicgui(
        auto_call=False,
        exposure_ms={"widget_type": "FloatSpinBox",'min': 3, 'max': 60,'label': 'Exposure (same for all channels)'},
        power_405={"widget_type": "FloatSpinBox", 'min': 0, 'max': 100, 'label': '405 nm power'},
        power_488={"widget_type": "FloatSpinBox", 'min': 0, 'max': 100, 'label': '488 nm power'},
        power_561={"widget_type": "FloatSpinBox", 'min': 0, 'max': 100, 'label': '561 nm power'},
        power_635={"widget_type": "FloatSpinBox", 'min': 0, 'max': 100, 'label': '635 nm power'},
        power_730={"widget_type": "FloatSpinBox", 'min': 0, 'max': 100, 'label': '730 nm power'},
        call_button='Set lasers')
    def define_channels(
        self,
        exposure_ms=10.0,
        power_405=0.0,
        power_488=0.0,
        power_561=0.0,
        power_635=0.0,
        power_730=0.0):

        if (self.fluidics_loaded and self.first_round_run):
            self.channel_powers_readout = [0.0,power_488,power_561,power_635,power_730]

            if not(self.codebook['nuclei_round']==-1):
                self.channel_powers_nuclei =  [power_405,power_488,0.0,0.0,0.0]

            self.exposure_ms = exposure_ms
            self.channels_set = True
            self._generate_experiment_summary()
        else:
            raise Exception('Configure fluidics and run initial round first.') 

    @magicgui(
        auto_call=False,
        scan_axis_start_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000, 'label': 'Scan start:'},
        scan_axis_end_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000,'label': 'Scan end:'},
        tile_axis_start_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000,'label': 'Tile start:'},
        tile_axis_end_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000,'label': 'Tile end:'},
        height_axis_start_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000,'label': 'Height start:'},
        height_axis_end_um={"widget_type": "FloatSpinBox", 'min' : -20000, 'max': 20000,'label': 'Height end:'},
        call_button='Set scan volume')
    def define_scan_volume(
        self,
        scan_axis_start_um=0.0,
        scan_axis_end_um=0.0,
        tile_axis_start_um=0.0,
        tile_axis_end_um=0.0,
        height_axis_start_um=0.0,
        height_axis_end_um=0.0):

        if (self.channels_set and self.first_round_run and self.fluidics_loaded):
            self.scan_axis_start_um  = scan_axis_start_um
            self.scan_axis_end_um  = scan_axis_end_um
            self.tile_axis_start_um = tile_axis_start_um
            self.tile_axis_end_um = tile_axis_end_um
            self.height_axis_start_um = height_axis_start_um
            self.height_axis_end_um = height_axis_end_um
            self._calculate_scan_volume()
            self._generate_experiment_summary()
        else:
            raise Exception("Configure fluidics, run initial round, and configure channels first.")

    fluidics_summary = widgets.TextEdit(label='Fluidics Summary', value="None", name="Fluidics summary")
    experiment_summary = widgets.TextEdit(label='Experiment Summary', value="None", name="Experiment summary")

    @magicgui(
        auto_call=True,
        accept_setup_btn={"widget_type": "PushButton", 'label': 'Accept iterative experiment setup'},
        layout='vertical'
    )
    def accept_setup(self, accept_setup_btn):
        if (self.stage_volume_set and self.channels_set and self.first_round_run and self.fluidics_loaded):
            self.worker_iterative_setup.start()
            self.setup_complete = True
        else:
            raise Exception('Configure fluidics, ruin intial round, configure channels, and configure stage scan volume first.')
    
    
    # return fluidics, codebook, and experimental setup for running stage scan
    @thread_worker
    def _return_experiment_setup(self):
        return self.codebook, self.df_fluidics, self.scan_settings, self.valve_controller,self.pump_controller

def main():

    ui=OPMIterative()
    ui.show(run=True)

if __name__ == "__main__":
    main()