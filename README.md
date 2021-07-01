![image](https://user-images.githubusercontent.com/26783318/124163887-eb04cb00-da54-11eb-9db8-87c5269d3996.png)

# qi2lab-OPM | Control, reconstruction, analysis code for multiplexed imaging using oblique plane microscopy (OPM)
Control and analysis code for the qi2lab @ ASU OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Original instrument details, performance, and verification found in joint [eLife publication ](https://elifesciences.org/articles/57681) with York and Fiolka labs. Codebase and wavefront data from the as-publsihed "stage scanning" high NA OPM are in the "elife-publication-frozen" branch of this repo.

The tools developed here can be used with stage scanning or galvo scanning OPM designs.

# Important changes (06/21)
There have been large-scale, breaking changes and bug fixes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in Micro-manager 2.0. Ongoing work on fast 3D single-molecule tracking and iterative imaging with fluidics will continue to live in separate branches in this repo for now.

On the instrument side, we have added galvo scanning, returned to a cylindrical lens instead of DSLM, and swapped out the Triggerscope 3B for an NI DAQ card for more reliable high-speed triggering. In all our our experiments, the camera acts as the master clock. Laser blanking and galvo movements are synchronized to avoid motion blur due to sCMOS rolling shutters.

A new parts lists is in-progress.

# 3D (near) real time viewing
* run_opm_galvoscan_realtime_display.py
  * Run and display a single volume, multicolor galvo mirror scan using the qi2lab OPM on the OPM computer itself. This does not transfer data to the NAS / server and currently cannot save multiposition or timelapse data. It can save the current volume of interest.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup size of galvo sweep (max 250 micrometers for our design), exposure time (galvo stability and laser blanking verified to work down to 1.5 ms exposure time in 256x2304 ROI with OrcaFusion BT), laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Napari is work-in-progress. Once setup, call python script and it will open Napari. Click "start" to start data display.
  * Depends on: [Napari](https://napari.org/), [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), [Numba](http://numba.pydata.org/), and various standard Python libraries.

# Stage scan operation
* run_opm_stagescan.py
  * Run a multiposition, multicolor stage scan using the qi2lab OPM. Optional capability to push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Setup: Setup ROI cropping on camera in Micromanager 2.0. Setup stage positions, exposure time, laser lines, and laser powers in the initial block of the main() function. Once setup, call python script and it will execute scan.
* recon_opm_stagescan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for [BigStitcher](https://imagej.net/BigStitcher). Can be pointed to directory for an in-progress acquisition or completed acquisition. In qi2lab, we often push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), image_post_processing.py (in this repo), data_io.py (in this repo),  flat_field.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution: [Microvolution](https://www.microvolution.com/), which is commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) or [dexp](https://github.com/royerlab/dexp) for open-source GPU deconvolution.
  * Optional dependencies for GPU flatfield: [Cupy](https://docs.cupy.dev/en/stable/index.html) and [Scipy](https://www.scipy.org/).
  * Usage: python recon_opm_stagescan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flatfield (DEFAULT), 1: PyImageJ-FIJI based flatfield (will be removed soon), 2: Python GPU based flatfield>, -s <0: save as TIFF, 1: save as BDV H5 (DEFAULT), 2: save as Zarr> -z <integer level of coverslip z axis downsampling. 1 = no downsampling (DEFAULT)>

# Galvo scan operation
* run_opm_galvoscan.py
  * Run a single position, timelapse, multicolor galvo mirror scan using the qi2lab OPM. Optional capability to push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup size of galvo sweep (max 200 micrometers), exposure time (galvo stability and laser blanking verified to work down to 1.5 ms exposure time in 256x2304 ROI with OrcaFusion BT), laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Micro-manager 2.0 is work-in-progress and should be completed by 05/21. Once setup, call python script and it will execute scan.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
* recon_opm_galvoscan.py
  * Reconstruct an OPM acquisition created using 'run_opm_galvoscan.py' and create separate multichannel TIFF stacks for each timepoint. In qi2lab, we often push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), image_post_processing.py (in this repo), data_io.py (in this repo),  flat_field.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution: [Microvolution](https://www.microvolution.com/), which is commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) or [dexp](https://github.com/royerlab/dexp) for open-source GPU deconvolution.
  * Optional dependencies for GPU flatfield: [Cupy](https://docs.cupy.dev/en/stable/index.html) and [Scipy](https://www.scipy.org/).
  *  Usage: python recon_opm_galvoscan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flatfield (DEFAULT), 1: PyImageJ-FIJI based flatfield (will be removed soon), 2: Python GPU based flatfield>, -s <0: save as TIFF (DEFAULT), 1: save as BDV H5, 2: save as Zarr> -z <integer level of coverslip z axis downsampling. 1 = no downsampling (DEFAULT)>

# Contributions / Acknowledgements
Peter Brown (ASU), Franky Djutanta (ASU), Doug Shepherd (ASU), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).

# Contact
For questions, contact Doug Shepherd (douglas.shepherd (at) asu.edu).
