# OPM
Control and analysis code for OPM using Snouty objective

* /Control-MM
  * Micromanager 2.0 script to run multicolor stage scan using an OPM built with ASI Tiger, Coherent OBIS Box, and Prime BSI.
* /Reconstruction-python
  * Python code to execute stage deskew using orthogonal interpolation and create a [BigStitcher](https://github.com/PreibischLab/BigStitcher/) compatible HDF5 file. Orthogonal interpolation algorithm directly inspired by [Vincent Maioli PhD thesis](https://doi.org/10.25560/68022). Depends on [npy2bdv](https://github.com/nvladimus/npy2bdv/) by Nikita Vladimirov and [Numba](http://numba.pydata.org/).

This is a work in progress as we work towards fully automated imaging and a release.
