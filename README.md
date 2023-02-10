![image](https://user-images.githubusercontent.com/26783318/124163887-eb04cb00-da54-11eb-9db8-87c5269d3996.png)

# qi2lab-OPM | Control, reconstruction, analysis code for multiplexed imaging using oblique plane microscopy (OPM)
Control and analysis code for the qi2lab @ ASU OPM using a solid immersion tertiary objective (aka [Mr. Snouty](https://andrewgyork.github.io/high_na_single_objective_lightsheet/)). Original instrument details, performance, and verification found in joint [eLife publication ](https://elifesciences.org/articles/57681) with York and Fiolka labs. Codebase and wavefront data from the as-publsihed "stage scanning" high NA OPM are in the "elife-publication-frozen" branch of this repo.

The tools developed here can be used with **any** stage scanning or galvo scanning skewed light sheet designs, from diSPIM to lattice light sheet to OPM.

# Important changes (02/22)
There have been large-scale, breaking changes and bug fixes since publication to the instrument design, control code, and reconstruction code. The codebase is moving towards Napari GUI control of the instrument and reconstruction. The pycro-manager based acquistion approach remains for now.

# Napari-GUI control with real-time deskewing
* /napari-control/opm_timelapse_control.py
  * Run and display a multicolor galvo mirror using the qi2lab OPM on the OPM computer itself.
  * Usage: Run and display OPM time-lapse experiments in real-time.
  * Depends on: [Napari](https://napari.org/),  [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus), [PyDAQmx](https://github.com/clade/PyDAQmx), [Numba](http://numba.pydata.org/), [dexp](https://github.com/royerlab/dexp), and various standard Python libraries.
* /napari-control/opm_iterative_control.py
  * Run and display an iterative labeling stage scan experiment (e.g. MERFISH) using the qi2lab OPM on the OPM computer itself.
  * Usage: Setup and execute iterative labeing experiments using qi2lab OPM .
  * Depends on: [Napari](https://napari.org/),  [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus), [PyDAQmx](https://github.com/clade/PyDAQmx), [Numba](http://numba.pydata.org/), [dexp](https://github.com/royerlab/dexp), and various standard Python libraries.
* /napari-control/opm_timelapse_reconstruction.py
  * Example of chaining together qi2lab and other OA tools to deskew, rotate, deconvolution, flat-field, and analyze skewed data. 
  * Depends on: [Napari](https://napari.org/), [PyDAQmx](https://github.com/clade/PyDAQmx), [Numba](http://numba.pydata.org/), [dexp](https://github.com/royerlab/dexp), and various standard Python libraries.

# Micromanager-GUI stage scan controland reconstruction
* pycromanager-control/run_opm_stagescan.py
  * Run a multiposition, multicolor stage scan using the qi2lab OPM. Optional capability to push data to a network drive during acquisition using 10G fiber connection for post-processing.
  * Depends on: [Micro-manager 2.0 gamma](https://micro-manager.org/wiki/Download_Micro-Manager_Latest_Release), [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/),  [PyDAQmx](https://github.com/clade/PyDAQmx), and various standard Python libraries.
  * Setup: Setup ROI cropping on camera in Micromanager 2.0. Setup stage positions, exposure time, laser lines, and laser powers in the initial block of the main() function. Once setup, call python script and it will execute scan.
* reconstruction/recon_opm_stagescan.py
  * Reconstruct an OPM acquisition created using 'run_opm_stagescan.py' and create a BDV H5 file for [BigStitcher](https://imagej.net/BigStitcher). Can be pointed to directory for an in-progress acquisition or completed acquisition.
  * Depends on: [Pycro-manager](https://pycro-manager.readthedocs.io/en/latest/), [Numba](http://numba.pydata.org/), [npy2bdv](https://github.com/nvladimus/npy2bdv), [scikit-image](https://scikit-image.org/), and various standard Python libraries.
  * Optional dependencies for raw data GPU deconvolution: [clij2-fft](https://github.com/clij/clij2-fft).
  * Optional dependencies for raw data GPU flatfield: [Cupy](https://docs.cupy.dev/en/stable/index.html) and [Scipy](https://www.scipy.org/).
  * Usage: python recon_opm_stagescan.py -i < inputdirectory > -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flatfield (DEFAULT), 1: PyImageJ-FIJI based flatfield (will be removed soon), 2: Python GPU based flatfield>, -s <0: save as TIFF, 1: save as BDV H5 (DEFAULT), 2: save as Zarr> -z <integer level of coverslip z axis downsampling. 1 = no downsampling (DEFAULT)>

# Contributions / Acknowledgements
Peter Brown (ASU), Franky Djutanta (ASU), Doug Shepherd (ASU), Nikita Vladimirov (BIMSB_MDC),  Henry Pinkard (UCB), Adam Glaser (UW), Jon Daniels (ASI), Reto Fiolka (UTSW), Kevin Dean (UTSW), Alfred Millett-Sikking (Calico), and Andrew York (Calico).

# Contact
For questions, contact Doug Shepherd (douglas.shepherd (at) asu.edu).
