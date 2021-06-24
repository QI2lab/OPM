"""
Updated results saved in *.pkl files with new filters
"""
import copy
import os
import glob
import pickle
import numpy as np
import localize

fdirs = [r"\\10.206.26.21\opm2\20210622a\glycerol90_1\2021_06_22_23;18;29_localization",
         r"\\10.206.26.21\opm2\20210622b\glycerol90_1\2021_06_23_01;17;29_localization",
         r"\\10.206.26.21\opm2\20210622c\glycerol80_1\2021_06_23_03;16;34_localization",
         r"\\10.206.26.21\opm2\20210622d\glycerol80_1\2021_06_23_05;33;50_localization",
         r"\\10.206.26.21\opm2\20210622e\glycerol60_1\2021_06_23_07;42;08_localization",
         r"\\10.206.26.21\opm2\20210622f\glycerol60_1\2021_06_23_09;56;27_localization"]

for dir in fdirs:
    files = glob.glob(os.path.join(dir, "*.pkl"))
    for f in files:
        with open(f, "rb") as fl:
            data = pickle.load(fl)

        # filter
        fp = data["fit_params"]
        ip = data["init_params"]
        filters = data["localization_settings"]
        x, y, z = localize.get_skewed_coords((25, 252, 1272), 0.115, 0.4, 30*np.pi/180)
        to_keep, conditions, condition_names, filter_settings = \
            localize.filter_localizations(fp, ip, (z, y, x), (0.6604, 0.1331), filters["min_dists_um"][0:2],
                                      (filters["sigmas_min_um"], filters["sigmas_max_um"]), filters["threshold"] * 0.5,
                                      dist_boundary_min=(0.5, 0.25), mode="skewed")

        data_updated = copy.deepcopy(data)
        data_updated["filter_settings"] = filter_settings
        data_updated["to_keep"] = to_keep
        data_updated["conditions"]: conditions
        data_updated["condition_names"]: condition_names

        with open(f, "wb") as fl:
            pickle.dump(data_updated, fl)

