from basicpy import BaSiC
from hyperactive import Hyperactive
import zarr
import dask.array as da
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

zarr_path = Path(r'D:\20230630_cells_MERFISH\processed\raw_zarr\cells.zarr')
channel_path = Path(r'r000\x000_y001_z000\ch488')
test_channel = zarr.open_group(zarr_path,mode='r',path=channel_path)
img_data = da.from_zarr(test_channel['raw_data'])
img_data_mean = da.mean(img_data,axis=[1,2])
top_brightness_idx = np.argsort(img_data_mean.compute())
brightest_flatfield_data = img_data[top_brightness_idx[-500:-1],:]

basic = BaSiC(
    get_darkfield=True, 
    working_size=[128,425],
    smoothness_flatfield=2,
    smoothness_darkfield=10
) 
to_fit = brightest_flatfield_data.compute().astype(np.uint16)
basic.fit(to_fit)

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
im = axes[0].imshow(basic.flatfield)
fig.colorbar(im, ax=axes[0])
axes[0].set_title("Flatfield")
im = axes[1].imshow(basic.darkfield)
fig.colorbar(im, ax=axes[1])
axes[1].set_title("Darkfield")
fig.tight_layout()

result = ((to_fit.astype(np.float32)-basic.darkfield)/basic.flatfield).astype(np.uint16)

for i in range(0, result.shape[0], 25):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    im = axes[0].imshow(to_fit[i])
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Original")
    im = axes[1].imshow(result[i])
    fig.colorbar(im, ax=axes[1])
    axes[1].set_title("Corrected")
    fig.suptitle(f"frame {i}")
    fig.tight_layout()

plt.show()