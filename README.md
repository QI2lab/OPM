# High NA oblique plane microscopy (OPM) with iterative labeling for multiplexed fluorescence imaging
Control and analysis code for the qi2lab @ ASU OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Original instrument details, performance, and verification found in joint [eLife publication ](https://elifesciences.org/articles/57681) with York and Fiolka labs. Codebase and wavefront data from our high NA OPM are in the "elife-publication-frozen" branch of this repo.

The tools developed here can be used with any OPM design.

# Important changes (05/21)
There have been large-scale, breaking changes and bug fixes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in Micro-manager 2.0. Ongoing work on fast 3D single-molecule tracking and iterative imaging with fluidics will continue to live in separate branches in this repo for now.

On the instrument side, we have added galvo scanning, changed the light sheet launcher to use an ETL for low NA remote focusing, returned to a cylindrical lens instead of DSLM, and swapped out the Triggerscope 3B for an NI DAQ card for more reliable high-speed triggering. In all our our experiments, the camera acts as the master clock. Laser blanking and galvo movements are synchronized to avoid motion blur due to sCMOS rolling shutters.

This branch incorporates fluidics control using a fluidics controller based on designs by the [Moffit Lab](https://moffittlab.github.io/). We are very thankful to them for their help.

A new parts lists is in-progress.

# Iterative OPM operation
* run_opm_iterative_stagescan_GUI.py
  * Run a multiround, multiposition, multicolor stage scan using the qi2lab OPM in light sheet mode. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Run python code with Micromanager open and fluidics program defined. Use the MM GUI to setup the corners of the stage scan, laser lines, laser powers, and exposure time. Software will prompt you to make sure everything looks ok and then execute the multiround imaging.
* recon_opm_iterative_stagescan.py 
  * Reconstruct an OPM acquisition created using 'run_opm_iterative_stagescan_GUI.py' and create a BDV H5 file for [BigStitcher](https://imagej.net/BigStitcher). Can be pointed to directory for an in-progress acquisition or completed acquisition. In qi2lab, we push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), data_io.py (in this repo), image_post_processing.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution and retrospective flatfield correction: [Microvolution](https://www.microvolution.com/) (commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) for open-source GPU deconvolution), [pyimagej](https://github.com/imagej/pyimagej), and local [FIJI](https://imagej.net/Fiji/Downloads) w/ [BaSiC](https://github.com/marrlab/BaSiC) plugin JAR.
  * Usage: python recon_opm_iterative_stagescan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>

# Iterative widefield bypass operation
* run_widefield_iterative_stagescan_GUI.py
  * Run a multiround, multiposition, multicolor stage scan using the qi2lab OPM in widefield bypass mode. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Run python code with Micromanager open and fluidics program defined. Use the MM GUI to setup the all stage positions for imaging, z stack size, laser lines, laser powers, and exposure time. Software will prompt you to make sure everything looks ok and then execute the multiround imaging.
* recon_widefield_iterative_stagescan_GUI.py 
  * Reconstruct an OPM acquisition created using 'run_widefield_iterative_stagescan_GUI.py' and create a BDV H5 file for [BigStitcher](https://imagej.net/BigStitcher). Can be pointed to directory for an in-progress acquisition or completed acquisition. In qi2lab, we push in-progress acquisitions to a NAS via 10G fiber and reconstruct during acquistion using a Linux server with a 10G fiber connection to the NAS, dual 12-core Xeon CPUs, 1TB of RAM, and a Titan RTX GPU.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), data_io.py (in this repo), image_post_processing.py (in this repo), and various standard Python libraries.
  * Optional dependencies for GPU deconvolution and retrospective flatfield correction: [Microvolution](https://www.microvolution.com/) (commerical software! Can replace with [pyCUDAdecon](https://pycudadecon.readthedocs.io/en/latest/) for open-source GPU deconvolution), [pyimagej](https://github.com/imagej/pyimagej), and local [FIJI](https://imagej.net/Fiji/Downloads) w/ [BaSiC](https://github.com/marrlab/BaSiC) plugin JAR.
  * Usage: python recon_opm_iterative_stagescan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>

# Contributions / Acknowledgements
Peter Brown (ASU), Franky Djutanta (ASU), Doug Shepherd (ASU), Jefff Moffitt (BCU & Harvard), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).

# Contact
For questions, contact Doug Shepherd (douglas.shepherd (at) asu.edu).
