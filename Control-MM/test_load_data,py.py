from pycromanager import Dataset
from pathlib import Path
import napari

# This path is to the top level of the dataset
data_path = Path('F:\\20201018\\scan_test_13\\')

# construct dataset
dataset = Dataset(data_path)

# create dask array for dataset
dask_array = dataset.as_array(stitched=False)
print(dask_array.shape)