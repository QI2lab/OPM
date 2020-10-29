from pycromanager import Dataset
from pathlib import Path
import napari

# This path is to the top level of the dataset
data_path = Path('E:\\20201024\\restrepo_z000_1\\')

# construct dataset
dataset = Dataset(data_path)

dask_array = dataset.as_array(verbose=False)
print(dask_array.shape)