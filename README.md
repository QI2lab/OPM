# High NA oblique plane microscopy (OPM) with iterative labeling for multiplexed fluorescence imaging
Control and analysis code for the qi2lab @ ASU OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Original instrument details, performance, and verification found in joint [eLife publication ](https://elifesciences.org/articles/57681) with York and Fiolka labs. Codebase and wavefront data from our as-published "stage scannning" high NA OPM are in the "elife-publication-frozen" branch of this repo.

The tools developed here can be used to run and reconstruct any OPM design using Micromanager, Pycromanager, and Python.

# Important changes (07/21)
There have been large-scale, breaking changes and bug fixes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in Micro-manager 2.0. Ongoing work on fast 3D single-molecule tracking and iterative imaging with fluidics will continue to live in separate branches in this repo for now.

Round-by-round localization using raw OPM data is currently happening in "localization" branch.

This branch incorporates fluidics control using a fluidics controller based on designs by the [Moffit Lab](https://moffittlab.github.io/). We are very thankful to them for their help.

A new parts lists is in-progress.

# Iterative OPM operation
* control/run_opm_iterative_stagescan_GUI.py
  * Run a multiround, multiposition, multicolor stage scan using the qi2lab OPM in light sheet mode. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Run python code with Micromanager open and fluidics program defined. Use the MM GUI to setup the corners of the stage scan, laser lines, laser powers, and exposure time. Software will prompt you to make sure everything looks ok and then execute the multiround imaging.

# Iterative widefield bypass operation
* control/run_widefield_iterative_stagescan_GUI.py
  * Run a multiround, multiposition, multicolor stage scan using the qi2lab OPM in widefield bypass mode. Push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Usage: Run python code with Micromanager open and fluidics program defined. Use the MM GUI to setup the all stage positions for imaging, z stack size, laser lines, laser powers, and exposure time. Software will prompt you to make sure everything looks ok and then execute the multiround imaging.

# Contributions / Acknowledgements
Peter Brown (ASU), Franky Djutanta (ASU), Doug Shepherd (ASU), Jeff Moffitt (BCU & Harvard), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).

# Contact
For questions, contact Doug Shepherd (douglas.shepherd (at) asu.edu).
