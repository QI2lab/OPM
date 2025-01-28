import numpy as np
from pathlib import Path
from numpy.typing import ArrayLike
import wavekit_py as wkpy
# import matplotlib.pyplot as plt
import time


mode_names = [
    "Vert. Tilt",
    "Horz. Tilt",
    "Defocus",
    "Vert. Asm.",
    "Oblq. Asm.",
    "Vert. Coma",
    "Horz. Coma",
    "3rd Spherical",
    "Vert. Tre.",
    "Horz. Tre.",
    "Vert. 5th Asm.",
    "Oblq. 5th Asm.",
    "Vert. 5th Coma",
    "Horz. 5th Coma",
    "5th Spherical",
    "Vert. Tetra.",
    "Oblq. Tetra.",
    "Vert. 7th Tre.",
    "Horz. 7th Tre.",
    "Vert. 7th Asm.",
    "Oblq. 7th Asm.",
    "Vert. 7th Coma",
    "Horz. 7th Coma",
    "7th Spherical",
    "Vert. Penta.",
    "Horz. Penta.",
    "Vert. 9th Tetra.",
    "Oblq. 9th Tetra.",
    "Vert. 9th Tre.",
    "Horz. 9th Tre.",
    "Vert. 9th Asm.",
    "Oblq. 9th Asm.",
]

class AOMirror:
    def __init__(self,
                 wfc_config_file_path: Path,
                 haso_config_file_path: Path,
                 interaction_matrix_file_path: Path,
                 flat_positions_file_path: Path = None,
                 coeff_file_path: Path = None,
                 n_modes: int = 32,
                 modes_to_ignore: list[int] = []):
        # confirm file paths
        self.haso_config_file_path = haso_config_file_path
        self.wfc_config_file_path = wfc_config_file_path
        self.interaction_matrix_file_path = interaction_matrix_file_path
        self.coeff_file_path = coeff_file_path
        self.flat_positions_file_path = flat_positions_file_path
        self.n_modes = n_modes
        self.modes_to_ignore = modes_to_ignore
        
        # create wkpy sub class objects
        # Wavefront corrector and set obejects
        self.wfc = wkpy.WavefrontCorrector(
            config_file_path = str(self.wfc_config_file_path)
        )
        self.wfc_set = wkpy.WavefrontCorrectorSet(wavefrontcorrector = self.wfc)
        self.wfc.connect(True)
        
        # create corrdata manager object and compute command matrix
        self.corr_data_manager = wkpy.CorrDataManager(haso_config_file_path = str(haso_config_file_path),
                                                      interaction_matrix_file_path = str(interaction_matrix_file_path))
        self.corr_data_manager.set_command_matrix_prefs(self.n_modes,True)
        self.corr_data_manager.compute_command_matrix()
           
        self.wfc.set_temporization(20)

        # create the configuration object
        self.haso_config, self.haso_specs, _ = wkpy.HasoConfig.get_config(config_file_path=str(self.haso_config_file_path))
        
        # construct pupil dimensions from haso specs
        pupil_dimensions = wkpy.dimensions(self.haso_specs.nb_subapertures,self.haso_specs.ulens_step)

        # create HasoSlopes object
        self.haso_slopes = wkpy.HasoSlopes(dimensions=pupil_dimensions,
                                           serial_number = self.haso_config.serial_number)
        
        # initiate an empty pupil
        self.pupil = wkpy.Pupil(dimensions=pupil_dimensions, value=False)
        pupil_buffer = self.corr_data_manager.get_greatest_common_pupil()
        self.pupil.set_data(pupil_buffer)

        center, radius = wkpy.ComputePupil.fit_zernike_pupil(
            self.pupil,
            wkpy.E_PUPIL_DETECTION.AUTOMATIC,
            wkpy.E_PUPIL_COVERING.CIRCUMSCRIBED,
            False
        )

        # create modal coeff object
        self.modal_coeff = wkpy.ModalCoef(modal_type=wkpy.E_MODAL.ZERNIKE)
        self.modal_coeff.set_zernike_prefs(
            zernike_normalisation=wkpy.E_ZERNIKE_NORM.RMS, 
            nb_zernike_coefs_total=self.n_modes, 
            coefs_to_filter=self.modes_to_ignore,
            projection_pupil=wkpy.ZernikePupil_t(
                center,
                radius
            )
        )

        if self.flat_positions_file_path is not None:
            # Configure mirror in the flat position
            self.flat_positions = np.asarray(self.wfc.get_positions_from_file(str(flat_positions_file_path)))
        else:
            self.flat_positions = np.zeros(self.wfc.nb_actuators)
                    
        self.current_coeffs = np.zeros(n_modes,dtype=np.float32)
        self.current_positions = np.asarray(self.flat_positions)
            
        self.set_mirror_flat()
        self.wfc_positions = {"flat":self.flat_positions}
    
    
    def __del__(self):
        self.wfc.disconnect()
        
        
    def set_mirror_flat(self):
        self.wfc.move_to_absolute_positions(self.flat_positions)
        
        
    def set_mirror_positions(self,
                             positions: np.ndarray):
        if positions.shape[0] != self.wfc.nb_actuators:
            raise ValueError(f"Positions array needs to have shape = {self.wfc.nb.actuators}")
        
        apply = True
        overage_acutators = np.sum(np.where(np.abs(positions) >= 0.99,1,0))
        if overage_acutators > 1:
            print('individual actuator voltage too high.')
            apply = False
        
        total_actuators = np.sum(np.abs(positions))
        if total_actuators >= 25:
            print('total voltage too high.')
            apply = False
        
        if apply:
            self.wfc.move_to_absolute_positions(positions)
            
        return apply        
        
        
    def set_modal_coefficients(self,
                               amps: ArrayLike):
        """
        """
        assert amps.shape[0]==self.n_modes, "amps array must have the same shape as the number of Zernike modes."
        # update modal data
        self.modal_coeff.set_data(
            coef_array = amps,
            index_array = np.arange(1, self.n_modes+1, 1),
            pupil = self.pupil
        ) 
        
        # create a new haso_slope from the new modal coeffiecients
        haso_slopes = wkpy.HasoSlopes(
            modalcoef = self.modal_coeff, 
            config_file_path=str(self.haso_config_file_path)
        )
        
        # SJS: When we calculate the deltas from "flat_positions" this means we need to define the position when the calibration was done.
        #      I could see spherical adding some tilt as well. Could be from misalignment during calibration.
        deltas = self.corr_data_manager.compute_delta_command_from_delta_slopes(delta_slopes=haso_slopes)
        new_positions = np.asarray(self.flat_positions) + np.asarray(deltas)
        
        apply = True
        overage_acutators = np.sum(np.where(np.abs(new_positions) >= 0.99,1,0))
        if overage_acutators > 1:
            print('individual actuator voltage too high.')
            apply = False
        
        total_actuators = np.sum(np.abs(new_positions))
        if total_actuators >= 25:
            print('total voltage too high.')
            apply = False
        
        if apply:
            self.wfc.move_to_absolute_positions(new_positions)
            self.current_positions = new_positions.copy()
            self.current_coeffs = amps
        time.sleep(.01)
        # self.update_mirror_positions()
        # print(self.flat_positions)
        # print(self.current_positions)
        
        return apply
        
        
    def update_mirror_positions(self):
        """
        """
        self.current_positions = np.array(self.wfc.get_current_positions())
        self.current_coeffs = np.asarray(self.modal_coeff.get_coefs_values()[0])
        self.deltas = self.current_positions - self.flat_positions
        
    def save_mirror_state(self, wfc_save_path: Path = None):
        self.wfc.save_current_positions_to_file(pmc_file_path=str(wfc_save_path))
    
    def save_mirror_positions(self, name: str = None):
        """
        """
        assert name is None, "saving mirror position requires a name!"
        
        self.update_mirror_positions()
        self.wfc_positions[name] = self.current_positions
        
        actuator_save_path = self.interaction_matrix_file_path.parent / Path(f"{name}_actuator_positions.wcs") 
        self.wfc.save_current_positions_to_file(pmc_file_path=str(actuator_save_path))
        
        # save last updated
        coeff_save_path = self.interaction_matrix_file_path.parent / Path(f"{name}_modalcoeffs.json")
        # copied from navigate
        coefs, coef_inds = self.current_coeffs
        mode_dict = {}
        for c in coef_inds:
            mode_dict[mode_names[c - 1]] = f"{coefs[c-1]:.4f}"

        import json

        with open(coeff_save_path, "w") as f:
            json.dump(mode_dict, f)
        f.close()
        
        if name=="flat":
            self.flat_positions = self.current_positions
            self.flat_coeffs = self.current_coeffs
            self.flat_positions_file_path = actuator_save_path

# NOTE: There can be a disconnect between what we show on the mirror and what the current modal_coeffs are. if the mirror actuators are set without reseting the current coeffs, as in directly, then the coeffs will not match the current mirror state. 



def DM_voltage_to_map(v):
    """
    Reshape the 52-long vector v into 2D matrix representing the actual DM aperture.
    Corners of the matrix are set to None for plotting.
    Parameters:
    v - double array of length 52
    
    Returns:
    output: 8x8 ndarray of doubles.
    -------
    Author: Nikita Vladimirov
    """
    import numpy as np
    M = np.zeros((8,8))
    M[:,:] = None
    M[2:6,0] = v[:4]
    M[1:7,1] = v[4:10]
    M[:,2] = v[10:18]
    M[:,3] = v[18:26]
    M[:,4] = v[26:34]
    M[:,5] = v[34:42]
    M[1:7,6] = v[42:48]
    M[2:6,7] = v[48:52]
    return M
    
# def plotDM(cmd, title="", 
#            cmap = "jet", 
#            vmin=-0.25,
#            vmax=0.25,
#            save_dir_path: Path = None):
#     """Plot deformable mirror (Mirao52e) commands arranged in 2D colormap, units: Volts"""
#     import numpy as np
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(1,1)
#     valmax = np.nanmax(cmd)
#     valmin = np.nanmin(cmd)
#     im = ax.imshow(DM_voltage_to_map(cmd), vmin=vmin, vmax=vmax,
#                     interpolation='nearest', cmap = cmap)
#     ax.text(0,-1, title + '\n min=' + "{:1.2f}".format(valmin) +
#            ', max=' + "{:1.2f}".format(valmax) + ' V',fontsize=12)
    
#     plt.colorbar(im)
#     if save_dir_path:
#         fig.savefig(save_dir_path / Path("mirror_positions.png"))
#     plt.show()
    