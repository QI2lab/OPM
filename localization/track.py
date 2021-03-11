"""
Load data analyzed by the script localize_diffusion.py and use trackpy to track beads and extract the mean square displacement
"""
import glob
import os
import re
import numpy as np
import pickle
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt

# root_dir = r"\\10.206.26.21\opm2\20210203\beads_50glyc_1000dilution_1\2021_02_21_12;56;10_localization"
# root_dir = r"\\10.206.26.21\opm2\20210203\beads_0glyc_1000dilution_1\2021_02_23_08;07;45_localization"
# root_dir = r"\\10.206.26.21\opm2\20210305a\crowders_densest_50glycerol\2021_03_07_08;12;02_localization"
root_dir = r"\\10.206.26.21\opm2\20210309\crowders-10x-50glyc\2021_03_10_17;57;53_localization"
data_files = glob.glob(os.path.join(root_dir, "localization_results*.pkl"))
inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_files])

centers = []
frame_inds = []
for ii, ind in enumerate(inds):
    with open(data_files[ind], "rb") as f:
        dat = pickle.load(f)
    centers.append(dat["centers"])
    frame_inds.append(np.ones(len(dat["centers"])) * ii)

frame_inds = np.concatenate(frame_inds)
centers = np.concatenate(centers, axis=0)
data = np.concatenate((frame_inds[:, None], centers, centers), axis=1)

df = pd.DataFrame(data, columns=["frame", "z", "y", "x", "zum", "yum", "xum"])

linked = tp.link_df(df, search_range=(1.0, 1.0, 1.0), memory=3, pos_columns=["xum", "yum", "zum"])
# nmin_traj = 100
nmin_traj = 20
linked = tp.filter_stubs(linked, nmin_traj)

# frames per second = 1 / (frames per volume * frame rate)
frames_per_sec = 1 / (25 * 2e-3)
msd = tp.emsd(linked, mpp=1.0, fps=frames_per_sec, max_lagtime=nmin_traj, pos_columns=["xum", "yum", "zum"])

# linear fit
slope = np.linalg.lstsq(msd.index[:, np.newaxis], msd)[0][0]

# plot results
ax = msd.plot(style='o', label="MSD", figsize=(16, 8))
figh = plt.gcf()

t_interp = np.linspace(0, msd.index.max(), 300)
ax.plot(t_interp, slope * t_interp, 'orange', label='linear fit, D=%0.5g' % (slope / 6))

ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
ax.set_xlabel('lag time $t$')
ax.legend(loc='upper left')

# save figures
fname = os.path.join(root_dir, "msd.png")
figh.savefig(fname)

plt.figure()
tp.plot_traj(linked)

# save track data
fname = os.path.join(root_dir, "tracks.pkl")
with open(fname, "wb") as f:
    pickle.dump(linked, f)