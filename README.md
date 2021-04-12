# High NA oblique plane microscopy (OPM)
Control and analysis code for the qi2lab @ ASU OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Original instrument details, performance, and verification found in joint [eLife publication ](https://elifesciences.org/articles/57681) with York and Fiolka labs. Codebase and wavefront data from our high NA OPM are in the "elife-publication-frozen" branch of this repo.

# Important changes (04/21)
There have been large-scale, breaking changes and bug fixes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in Micro-manager 2.0. Ongoing work on fast 3D single-molecule tracking and iterative imaging with fluidics will continue to live in separate branches in this repo for now.

On the instrument side, we have added galvo scanning, changed the light sheet launcher to use an ETL for low NA remote focusing, returned to a cylindrical lens instead of DSLM, and integrated an NI DAQ card for high-speed triggering. In all our our experiments, the camera acts as the master clock. Laser blanking and galvo movements are synchronized to avoid motion blur due to sCMOS rolling shutters.

A new parts lists is in-progress.

# Stage scan operation
* run_opm_stagescan.py
  * Run a multiposition, multicolor stage scan using the qi2lab OPM. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup stage positions, exposure time, laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Micro-manager 2.0 is work-in-progress and should be completed by 05/21. Once setup, call python script and it will execute scan.
* recon_opm_stagescan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for [BigStitcher](https://imagej.net/BigStitcher). Can be pointed to directory for an in-progress acquisition or completed acquisition. In qi2lab, we push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), image_post_processing.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution and retrospective flatfield correction: [Microvolution](https://www.microvolution.com/) (commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) for open-source GPU deconvolution), [pyimagej](https://github.com/imagej/pyimagej), and local [FIJI](https://imagej.net/Fiji/Downloads) w/ [BaSiC](https://github.com/marrlab/BaSiC) plugin JAR.
  * Usage: python recon_opm_stagescan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>

# Galvo scan operation
* run_opm_galvoscan.py
  * Run a single position, timelapse, multicolor galvo mirror scan using the qi2lab OPM. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Usage: Setup ROI cropping on camera in Micromanager 2.0. Setup size of galvo sweep (max 200 micrometers), exposure time (galvo stability and laser blanking verified to work down to 1.5 ms exposure time in 256x2304 ROI with OrcaFusion BT), laser lines, and laser powers in the initial block of the main() function. Setting all of these up directly in Micro-manager 2.0 is work-in-progress and should be completed by 05/21. Once setup, call python script and it will execute scan.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
* recon_opm_galvoscan.py
  * Reconstruct an OPM acquisition created using 'run_opm_galvoscan.py' and create a BDV H5 file for BDV viewing. In qi2lab, we push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), image_post_processing.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution and retrospective flatfield correction: [Microvolution](https://www.microvolution.com/) (commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) for open-source GPU deconvolution), [pyimagej](https://github.com/imagej/pyimagej), and local [FIJI](https://imagej.net/Fiji/Downloads) w/ [BaSiC](https://github.com/marrlab/BaSiC) plugin JAR.
  * Currently has option to reconstruct data acquired directly with Hamamatsu HCImage software, due to some debugging we are doing with an OrcaFusion BT. This will be removed once debugging is done.
  * Usage: python recon_opm_galvoscan.py -i < inputdirectory > -a <0: pycromanager (DEFAULT), 1: hcimage> -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>

# Contributions / Acknowledgements
Peter Brown (ASU), Franky Djutanta (ASU), Doug Shepherd (ASU), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).

# Contact
For questions, contact Doug Shepherd (douglas.shepherd (at) asu.edu).