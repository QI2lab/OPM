import dask.array as da
from tifffile import imwrite
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

input_zarr_path = Path(r'Z:\20230623_molm14_bafilomyocin\deskew_decon_output\OPM_processed.zarr')
output_tiff_path = Path(r'Z:\20230623_molm14_bafilomyocin\deskew_decon_output\tiff')
output_tiff_path.mkdir(parents=False,exist_ok=True)
metadata_path = Path(r'Z:\20230623_molm14_bafilomyocin\scan_metadata.csv')

# load zarr as dask array
datastore = da.from_zarr(input_zarr_path)

# load metadata as pandas dataframe
metadata = pd.read_csv(metadata_path)

# extract key metadata information
num_t = metadata['num_t'].values[0].astype(int)
galvo_scan_range_um = metadata['galvo_scan_range_um'].values[0].astype(float)
scan_step = metadata['scan_step'].values[0].astype(float)
exposure_ms = metadata['exposure_ms'].values[0].astype(float)
time_delay_s = metadata['time_delay'].values[0].astype(float)
pixel_size = metadata['pixel_size'].values[0].astype(float)

# calculate TIFF metadata
f_interval_s = .001 * exposure_ms * (galvo_scan_range_um / scan_step) + time_delay_s
fps = np.round(1/f_interval_s,4)

for t_idx in tqdm(range(0,num_t)):
    filename = output_tiff_path / Path('OPM_data_t' + str(t_idx).zfill(5)+'.tiff')
    data_to_write = datastore[t_idx,:,:].compute()
    data_to_write = np.swapaxes(data_to_write,0,1)
    imwrite(filename,
            np.squeeze(data_to_write),
            imagej=True,
            resolution = (1/pixel_size,1/pixel_size),
            metadata = {'spacing' : pixel_size,
                        'unit' : 'um',
                        'finterval' : f_interval_s,
                        'fps': fps,
                        'axes': 'ZCYX'})