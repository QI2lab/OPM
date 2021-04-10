# High NA oblique plane microscopy
Control and analysis code for our OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Publication [here](https://elifesciences.org/articles/57681). Codebase and wavefront data from publication are in the "elife-publication-frozen" branch. 

# Important changes (04/21)
There have been large-scale, breaking changes since publication to the instrument design, control code, and reconstruction code. There will be additional refactors in the common weeks to modularize the code for shared functions and streamline acquisition setup in micromanager. Iterative imaging with fluidics will continue to live in it's own branch (MERFISH) for now.

# Stage scan operation
* run_opm_stagescan.py
  * Run a multiposition, multicolor stage scan using our OPM. Push data during acquisition to NAS, which is deskewed and placed into BigDataViewer H5 file for BigStitcher on the fly.
  * Depends on: Micromanager 2.0, Pycro-manager,  PyDAQmx, and various standard Python libraries.
* recon_opm_stagescan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for BigStitcher.
  * Depends on: Pycro-manager, Numba, Microvolution (commerical software!), npy2bdv, pyimagej, local FIJI w/ BaSiC plugin JAR,  and various standard Python libraries.

# Galvo scan operation
* run_opm_galvoscan.py
  * Run a single position, multicolor galvo mirror scan using our OPM. Push data after acquisition to NAS.
  * Depends on: Micromanager 2.0, Pycro-manager,  PyDAQmx, and various standard Python libraries.
* recon_opm_galvocan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for BDV viewing.
  * Depends on: Pycro-manager, Numba, Microvolution (commerical software!), npy2bdv, pyimagej, local FIJI w/ BaSiC plugin JAR,  and various standard Python libraries.

# Contributions / Acknowledgements
Peter Brown (ASU), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).
