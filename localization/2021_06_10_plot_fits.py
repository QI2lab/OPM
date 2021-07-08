
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import tifffile
import localize
import localize_skewed


rounds = list(range(8))
tiles = list(range(9))
channels = [1, 2]

root_dir = r"\\10.206.26.21\opm2\20210610\output\fused_tiff"
# data_dir = os.path.join(root_dir, "2021_06_27_14;02;39_localization")
data_dir = os.path.join(root_dir, "2021_06_27_16;10;01_localization")

for round in rounds:
    for tile in tiles:
        for channel in channels:
            # img_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\img_TL0_Ch0_Tile0.tif" # cell outlines
            img_fname = os.path.join(root_dir, "img_TL%d_Ch%d_Tile%d.tif" % (round, channel, tile))
            # data_fname = r"\\10.206.26.21\opm2\20210610\output\fused_tiff\2021_06_27_11;08;01_localization\img_TL0_Ch1_Tile0_round=0_ch=1_tile=0_vol=0.pkl"
            data_fname = os.path.join(data_dir, "img_TL%d_Ch%d_Tile%d_round=%d_ch=%d_tile=%d_vol=0.pkl" % (round, channel, tile, round, channel, tile))

            imgs = tifffile.imread(img_fname)

            # get coordinates
            dc = 0.065
            dz = 0.25
            x, y, z = localize.get_coords(imgs.shape, dc, dz)

            with open(data_fname, "rb") as f:
                data = pickle.load(f)

            fps = data["fit_params"]
            ips = data["init_params"]
            rois = data["rois"]
            to_keep = data["to_keep"]


            for ind in range(len(fps)):
                if to_keep[ind]:
                    figa = localize.plot_gauss_roi(fps[ind], rois[ind], imgs, x, y, z,
                                                    init_params=ips[ind],
                                                    figsize=(16, 8), same_color_scale=True, prefix="TL%d_Ch%d_Tile%d_ROI%d_plot" % (round, channel, tile, ind),
                                                    save_dir=data_dir)
                    plt.close(figa)
