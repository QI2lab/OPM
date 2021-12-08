import numpy as np
from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from functools import partial
import dask.array as da

def _c(array):
    """
    :param array: dexp array
    :return array: numpy array
    """

    return Backend.to_numpy(array)

def tiled_lr_decon(image,psf,num_iterations,padding,internal_dtype):

    result = lucy_richardson_deconvolution(
        image=image,
        psf=psf,
        num_iterations=num_iterations,
        padding=padding,
        internal_dtype=internal_dtype
    )

    return _c(result)

def lr_deconvolution_cupy(image,psf,iterations=50):
    """
    Lucy-Richardson deconvolution using dexp library
    :param image: raw OPM data
    :param skewed_psf: theoretical PSF skewed into OPM coordinates
    :param iterations: number of iterations to run 
    :return deconvolved:
    """

    lr_dask = partial(tiled_lr_decon,psf=psf,num_iterations=50,padding=16,internal_dtype=np.float16)
    dask_raw = da.from_array(image.astype(np.float16),chunks=(256,256,256))
    dask_decon = da.map_overlap(lr_dask,dask_raw,depth=20,boundary=None,trim=True,dtype=np.float16)

    with CupyBackend():
        decon = dask_decon.compute(scheduler='single-threaded')
        
    return decon.astype(np.uint16)