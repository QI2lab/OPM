# High NA oblique plane microscopy
Control and analysis code for our OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Preprint [here](https://www.biorxiv.org/content/10.1101/2020.04.07.030569v2).

* /Control-MM
  * [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/) code to run multicolor stage scan using an OPM built with ASI Tiger controller, ASI constant speed scan optimized XY stage, ASI FTP focusing stage, Coherent OBIS Laser Box with 5 Coherent OBIS lasers, custom DAC, and Teledyne Photometrics Prime BSI.
* /Reconstruction-python
  * Python code to execute stage deskew using orthogonal interpolation and create a [BigStitcher](https://github.com/PreibischLab/BigStitcher/) compatible HDF5 file. Orthogonal interpolation algorithm directly inspired by [Vincent Maioli PhD thesis](https://doi.org/10.25560/68022). Depends on [npy2bdv](https://github.com/nvladimus/npy2bdv/), [Numba](http://numba.pydata.org/), [scikit-image](https://scikit-image.org/), [natsort](https://natsort.readthedocs.io/en/master/index.html), and standard Python libraries.
* /wavefront
  * Raw data of O3 (Snouty) pupil in various configurations acquired using Shack-Hartmann sensor. Python code to load, crop, and analyze wavefront data. Uses [prysm](https://prysm.readthedocs.io/en/stable/index.html) and standard Python libraries to perform analysis.

This is a work in progress as we work towards fully automated imaging and a release. Integration of our fluidics setup, [pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), and real time data deskewing are next on the roadmap.

# Contributions / Acknowledgements
Peter Brown (ASU), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).
