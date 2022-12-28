from opm_psf import generate_skewed_psf
from tifffile import imwrite
import numpy as np
from pathlib import Path

range_pz = np.arange(0,145,15)
dxy = .115
ds = .400
em_wvls = [.420,.520,.580,.670,.780]

root_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\psfs')

for em_wvl in em_wvls:
    for pz in range_pz:
        skewed_psf = generate_skewed_psf(.01,.488,em_wvl,pz,plot=False)
        tiff_filename= 'opm_psf_w'+str(int(100*em_wvl))+'_p'+str(int(pz))+'.tiff'
        tiff_output_path = root_path / Path(tiff_filename)
        imwrite(tiff_output_path, skewed_psf.astype(np.float32), imagej=True, resolution=(1/dxy, 1/dxy),
                metadata={'spacing': (ds), 'unit': 'um', 'axes': 'ZYX'})