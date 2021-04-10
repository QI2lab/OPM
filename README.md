# High NA oblique plane microscopy
Control and analysis code for our OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Publication [here](https://elifesciences.org/articles/57681). Codebase and wavefront data from publication are in the "elife-publication-frozen" branch. 

# Important changes (04/21)
There have been large-scale, breaking changes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in Micro-manager 2.0. Ongoing work on fast 3D single-molecule tracking and iterative imaging with fluidics will continue to live in separate branches in this repo for now.

On the instrument side, we have added galvo scanning, changed the light sheet launcher to use an ETL for low NA remote focusing, returned to a cylindrical lens instead of DSLM, and integrated an NI DAQ card for high-speed triggering. In all our our experiments, the camera acts as the master clock. Laser blanking and galvo movements are synchronized to avoid motion blur due to sCMOS rolling shutters.

# Stage scan operation
* run_opm_stagescan.py
  * Run a multiposition, multicolor stage scan using our OPM. Push data during acquisition to our NAS for post-processing by server.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup stage positions, exposure time, laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Micro-manager 2.0 is work-in-progress and should be completed by 05/21. Once setup, call python script and it will execute scan.
* recon_opm_stagescan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for BigStitcher.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [Microvolution](https://www.microvolution.com/) (commerical software!), [npy2bdv](https://github.com/nvladimus/npy2bdv), [pyimagej](https://github.com/imagej/pyimagej), local FIJI w/ BaSiC plugin JAR, flatfield.py (in this repo), and various standard Python libraries.
  * Usage: python recon_opm_stagescan.py -i < inputdirectory > -d <0: no deconvolution, 1: deconvolution> -f <0: no flat-field 1: flat-field>

# Galvo scan operation
* run_opm_galvoscan.py
  * Run a single position, timelapse, multicolor galvo mirror scan using our OPM. Push data after acquisition to our NAS for post-processing by server.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup size of galvo sweep (max 200 micrometers), exposure time (verified to work down to 1.5 ms with OrcaFusion BT, our laser launch, and our galvo), laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Micro-manager 2.0 is work-in-progress and should be completed by 05/21. Once setup, call python script and it will execute scan.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
* recon_opm_galvocan.py
  * Reconstruct an OPM acquisition created using 'run_opm_galvoscan.py' and create a BDV H5 file for BDV viewing.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [Microvolution](https://www.microvolution.com/) (commerical software!), [npy2bdv](https://github.com/nvladimus/npy2bdv), [pyimagej](https://github.com/imagej/pyimagej), local FIJI w/ BaSiC plugin JAR, flatfield.py (in this repo), and various standard Python libraries.
  * Currently has option to reconstruct data acquired directly with Hamamatsu HCImage software, due to some debugging we are doing with an OrcaFusion BT. This will be removed once debugging is done.
  * Usage: python recon_opm_galvoscan.py -i < inputdirectory > -a <0: pycromanager, 1: hcimage> -d <0: no deconvolution, 1: deconvolution> -f <0: no flat-field 1: flat-field>

# Contributions / Acknowledgements
Peter Brown (ASU), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).
