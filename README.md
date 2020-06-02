# High NA oblique plane microscopy
Control and analysis code for our OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Preprint [here](https://www.biorxiv.org/content/10.1101/2020.04.07.030569v2).

* /Control-MM
  * [Micromanager 2.0](https://micro-manager.org/wiki/Version_2.0) script to run multicolor stage scan using an OPM built with ASI Tiger controller, ASI constant speed scan optimized XY stage, ASI objective focusing LS-50 stage, Coherent OBIS Laser Box with 5 Coherent OBIS lasers, and Teledyne Photometrics Prime BSI Express.
* /Reconstruction-python
  * Python code to execute stage deskew using orthogonal interpolation and create a [BigStitcher](https://github.com/PreibischLab/BigStitcher/) compatible HDF5 file. Orthogonal interpolation algorithm directly inspired by [Vincent Maioli PhD thesis](https://doi.org/10.25560/68022). Depends on [npy2bdv](https://github.com/nvladimus/npy2bdv/), [Numba](http://numba.pydata.org/), [scikit-image](https://scikit-image.org/), [natsort](https://natsort.readthedocs.io/en/master/index.html), and standard Python libraries.
* /probedesign-python (work in progress)
 * Python code to design MERFISH encoding probes, readout probes, amplifier scheme, and codebook from a set of genes. Encoding probe seqeuences are based on the [Oligopaints] (https://oligopaints.hms.harvard.edu/) framework. General MERFISH encoding probe design is a synthesis of the existing MERFISH literature, which are referenced in the Jupyter notebook.
* /fluidics-python (stub)
 * Python code to control fluidics unit
* /Control-pyMM (stub)
 * pycro-manager control software that integrates existing Micromanager 2.0 control code with fluidics control for fully automated acquisition.

This is a work in progress as we work towards fully automated imaging and a release. Integration of our fluidics setup, [pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), and real time data deskewing are next on the roadmap.

# Contributions / Acknowledgements
Peter Brown (ASU), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Jeffrey Moffitt (Harvard Medical School), Brian Long (Allen Brain Institute), Alfred Millet-Sikking (Calico), and Andrew York (Calico).
