# High NA orthogonal plane microscope
Control and analysis code for our OPM using bespoke solid immersion objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Preprint [here](https://www.biorxiv.org/content/10.1101/2020.04.07.030569v2).

* /Control-MM
  * [Micromanager 2.0](https://micro-manager.org/wiki/Version_2.0) script to run multicolor stage scan using an OPM built with ASI Tiger, ASI XY stage, ASI objective focusing LS-50 stage, Coherent OBIS Box, and Teledyne Photometrics Prime BSI Express.
* /Reconstruction-python
  * Python code to execute stage deskew using orthogonal interpolation and create a [BigStitcher](https://github.com/PreibischLab/BigStitcher/) compatible HDF5 file. Orthogonal interpolation algorithm directly inspired by [Vincent Maioli PhD thesis](https://doi.org/10.25560/68022). Depends on [npy2bdv](https://github.com/nvladimus/npy2bdv/), [Numba](http://numba.pydata.org/), [scikit-image](https://scikit-image.org/), [natsort](https://natsort.readthedocs.io/en/master/index.html), and standard Python libraries.

This is a work in progress as we work towards fully automated imaging and a release.

# Contributions / Acknowledgements
Peter Brown (ASU), Adam Galser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millet-Sikking (Calico), and Andrew York (Calico).
