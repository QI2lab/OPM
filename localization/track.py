"""
Load data analyzed by the script localize_diffusion.py and use trackpy to track beads and extract the mean square displacement
"""
import glob
import os
import re
import numpy as np
import pickle
import scipy.optimize
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
import data_io

use_every_nth_frame = 1
figsize = (16, 8)
nmin_traj = 5

# root_dir = r"\\10.206.26.21\opm2\20210203\beads_50glyc_1000dilution_1\2021_02_21_12;56;10_localization"
# root_dir = r"\\10.206.26.21\opm2\20210203\beads_0glyc_1000dilution_1\2021_02_23_08;07;45_localization"
# root_dir = r"\\10.206.26.21\opm2\20210305a\crowders_densest_50glycerol\2021_03_07_08;12;02_localization"
# load all localization data
# data_dirs = glob.glob(os.path.join(root_dir, r"vol_*"))
# inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_dirs])

# root_dir = r"\\10.206.26.21\opm2\20210309\crowders-10x-50glyc\2021_03_10_17;57;53_localization"
# root_dir = r"\\10.206.26.21\opm2\20210309\crowders-1x-50glyc\2021_03_11_15;50;33_localization"
# root_dir = r"\\10.206.26.21\opm2\20210309\no_crowders\2021_03_14_13;21;57_localization"
# data_files = glob.glob(os.path.join(root_dir, "localization_results*.pkl"))
# inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_files])

# define dataset
# root_dir = r"\\10.206.26.21\opm2\20210408m\glycerol50x_1\2021_04_09_11;40;36_localization"
# root_dir = r"\\10.206.26.21\opm2\20210408m\glycerol50x_1\2021_04_09_19;23;24_localization"
# data_files = glob.glob(os.path.join(root_dir, "localization_results*.pkl"))
# inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_files])

# root_dir = r"\\10.206.26.21\opm2\20210408n\glycerol60x_1\2021_04_12_10;37;46_localization"
# root_dir = r"\\10.206.26.21\opm2\20210408n\glycerol60x_1\2021_04_20_16;15;27_localization"
# root_dir = r"\\10.206.26.21\opm2\20210408n\glycerol60x_1\2021_04_20_18;15;17_localization"
# root_dir = r"\\10.206.26.21\opm2\20210408m\glycerol50x_1\2021_04_20_17;09;52_localization"
# data_files = glob.glob(os.path.join(root_dir, "localization_results*.pkl"))
# inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_files])

# root_dirs = [r"\\10.206.26.21\opm2\20210622a\glycerol90_1\2021_06_22_23;18;29_localization",
#             r"\\10.206.26.21\opm2\20210622b\glycerol90_1\2021_06_23_01;17;29_localization",
#             r"\\10.206.26.21\opm2\20210622c\glycerol80_1\2021_06_23_03;16;34_localization",
#             r"\\10.206.26.21\opm2\20210622d\glycerol80_1\2021_06_23_05;33;50_localization",
#             r"\\10.206.26.21\opm2\20210622e\glycerol60_1\2021_06_23_07;42;08_localization",
#             r"\\10.206.26.21\opm2\20210622f\glycerol60_1\2021_06_23_09;56;27_localization",
#             r"\\10.206.26.21\opm2\20210622g\glycerol50_1\2021_06_23_17;45;54_localization",
#             r"\\10.206.26.21\opm2\20210622h\glycerol50_1\2021_06_23_20;09;39_localization"]

# root_dirs = [r"\\10.206.26.21\opm2\20210624a\glycerol80_1\2021_06_24_16;37;59_localization",
root_dirs = [r"\\10.206.26.21\opm2\20210624b\glycerol80_1\2021_06_24_17;19;23_localization",
             r"\\10.206.26.21\opm2\20210624c\glycerol80_1\2021_06_24_17;53;35_localization",
             r"\\10.206.26.21\opm2\20210624d\glycerol80_1\2021_06_24_18;27;45_localization",
             r"\\10.206.26.21\opm2\20210624e\glycerol60_1\2021_06_24_18;56;45_localization",
             r"\\10.206.26.21\opm2\20210624f\glycerol60_1\2021_06_24_19;26;29_localization",
             r"\\10.206.26.21\opm2\20210624g\glycerol60_1\2021_06_24_19;53;54_localization",
             r"\\10.206.26.21\opm2\20210624h\glycerol60_1\2021_06_24_20;28;11_localization",
             r"\\10.206.26.21\opm2\20210624i\glycerol50_1\2021_06_24_21;08;45_localization",
             r"\\10.206.26.21\opm2\20210624j\glycerol50_1\2021_06_24_21;41;37_localization",
             r"\\10.206.26.21\opm2\20210624k\glycerol50_1\2021_06_24_22;14;02_localization",
             r"\\10.206.26.21\opm2\20210624l\glycerol50_1\2021_06_24_22;48;05_localization"]

for root_dir in root_dirs:
    print(root_dir)
    data_files = glob.glob(os.path.join(root_dir, "localization_results*.pkl"))
    inds = np.argsort([int(re.match(".*_(\d+)", d).group(1)) for d in data_files])

    scan_data_path = os.path.join(root_dir, "..", "..", "scan_metadata.csv")

    # read on file for useful info
    with open(data_files[0], "rb") as f:
        d = pickle.load(f)

    # read useful metadata
    scan_data = data_io.read_metadata(scan_data_path)

    frames_per_vol = scan_data["scan_axis_positions"]
    # frames_per_vol = 25
    try:
        raw_frame_period = d["frame_time_ms"] * 1e-3
    except KeyError:
        raw_frame_period = d["physical_data"]["frame_time_ms"] * 1e-3

    frames_per_sec = 1 / (frames_per_vol * raw_frame_period * use_every_nth_frame)
    vol = d["volume_um3"]

    vol = (scan_data["pixel_size"] / 1e3 * scan_data["x_pixels"]) * \
          (scan_data["scan_step"] / 1e3 * scan_data["scan_axis_positions"]) * \
          (scan_data["pixel_size"] / 1e3 * np.sin(scan_data["theta"] * np.pi/180) * scan_data["y_pixels"])

    # load all localization data
    print("loading data")
    centers = []
    frame_inds = []
    for ii, ind in enumerate(inds):
        with open(data_files[ind], "rb") as f:
            dat = pickle.load(f)

        fp = dat["fit_params"][dat["to_keep"]]
        if len(fp) < 500 and np.mod(ii, use_every_nth_frame) == 0:
            centers.append(np.stack((fp[:, 3], fp[:, 2], fp[:, 1]), axis=1))
            frame_inds.append(np.ones(len(fp)) * ii // use_every_nth_frame)
        else:
            pass

    frame_inds = np.concatenate(frame_inds)
    centers = np.concatenate(centers, axis=0)
    data = np.concatenate((frame_inds[:, None], centers, centers), axis=1)

    # put data in dataframe format expected by TrackPy
    df = pd.DataFrame(data, columns=["frame", "z", "y", "x", "zum", "yum", "xum"])

    # do linking
    print("linking")
    linked = tp.link_df(df, search_range=(1.0, 1.0, 1.0), memory=3, pos_columns=["xum", "yum", "zum"])
    # filter trajectories shorter than a certain length
    linked = tp.filter_stubs(linked, nmin_traj)

    # #######################
    # MSD
    # #######################
    # per trajectory
    # mpp = microns per pixel
    msd_per_traj = tp.imsd(linked, mpp=1.0, fps=frames_per_sec, max_lagtime=np.inf, pos_columns=["xum", "yum", "zum"])
    msd_per_traj_arr = msd_per_traj.to_numpy()

    # get step sizes
    pos_x = linked.set_index(['frame', 'particle'])['xum'].unstack() # particles as columns
    pos_y = linked.set_index(['frame', 'particle'])['yum'].unstack()
    pos_z = linked.set_index(['frame', 'particle'])['zum'].unstack()
    # vhove_x = tp.vanhove(pos_x, 1, mpp=1, ensemble=True)
    # vhove_y = tp.vanhove(pos_y, 1, mpp=1, ensemble=True)
    # vhove_z = tp.vanhove(pos_z, 1, mpp=1, ensemble=True)

    nbins = 1000
    bin_edges = np.linspace(-2, 2, nbins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    def gauss(p, x): return p[0] * np.exp(-(x - p[1])**2 / 2 / p[2]**2)

    # x-Van Hove function
    pos_x_arr = pos_x.to_numpy()
    step_sizes = pos_x_arr[1:] - pos_x_arr[: -1]
    step_sizes = step_sizes[np.logical_not(np.isnan(step_sizes))]
    steps_x_hist, _ = np.histogram(step_sizes, bin_edges)

    def fit_fn(p): return gauss(p, bin_centers) - steps_x_hist

    first_moment = np.mean(steps_x_hist * bin_centers) / np.sum(steps_x_hist)
    second_moment = np.mean(steps_x_hist * bin_centers**2) / np.sum(steps_x_hist)
    sd = np.sqrt(second_moment - first_moment**2)
    init_params_x = [np.max(steps_x_hist), first_moment, sd]

    results_x = scipy.optimize.least_squares(fit_fn, init_params_x)
    gauss_fit_x = gauss(results_x["x"], bin_centers)

    # y
    pos_y_arr = pos_y.to_numpy()
    step_sizes = pos_y_arr[1:] - pos_y_arr[: -1]
    step_sizes = step_sizes[np.logical_not(np.isnan(step_sizes))]
    steps_y_hist, _ = np.histogram(step_sizes, bin_edges)

    def fit_fn(p): return gauss(p, bin_centers) - steps_y_hist

    first_moment = np.mean(steps_y_hist * bin_centers) / np.sum(steps_y_hist)
    second_moment = np.mean(steps_y_hist * bin_centers**2) / np.sum(steps_y_hist)
    sd = np.sqrt(second_moment - first_moment**2)
    init_params_y = [np.max(steps_y_hist), first_moment, sd]

    results_y = scipy.optimize.least_squares(fit_fn, init_params_y)
    gauss_fit_y = gauss(results_y["x"], bin_centers)

    # z
    pos_z_arr = pos_z.to_numpy()
    step_sizes = pos_z_arr[1:] - pos_z_arr[: -1]
    step_sizes = step_sizes[np.logical_not(np.isnan(step_sizes))]
    steps_z_hist, _ = np.histogram(step_sizes, bin_edges)

    def fit_fn(p): return gauss(p, bin_centers) - steps_z_hist

    first_moment = np.mean(steps_z_hist * bin_centers) / np.sum(steps_z_hist)
    second_moment = np.mean(steps_z_hist * bin_centers**2) / np.sum(steps_z_hist)
    sd = np.sqrt(second_moment - first_moment**2)
    init_params_z = [np.max(steps_z_hist), first_moment, sd]

    results_z = scipy.optimize.least_squares(fit_fn, init_params_z)
    gauss_fit_z = gauss(results_z["x"], bin_centers)


    # ensemble averaged
    msd = tp.emsd(linked, mpp=1.0, fps=frames_per_sec, max_lagtime=np.inf, pos_columns=["xum", "yum", "zum"])

    n_traj_contributing = np.sum(np.logical_not(np.isnan(msd_per_traj_arr)), axis=1)

    # linear fit of ensemble averaged MSD
    full_msd_slope = np.linalg.lstsq(msd.index[:, np.newaxis], msd, rcond=None)[0][0]

    tshort_cutoff = 2
    to_use = msd.index < tshort_cutoff
    short_msd_slope = np.linalg.lstsq(msd.index[to_use, None], msd[to_use], rcond=None)[0][0]

    # power law fit
    t_plaw_cutoff = 10
    to_use = msd.index < t_plaw_cutoff
    def line_fn(p, t): return p[0] * t + p[1]
    def fit_fn(p): return line_fn(p, np.log(msd.index[to_use].to_numpy())) - np.log(msd[to_use].to_numpy())
    results_plaw = scipy.optimize.least_squares(fit_fn, [1, 1])

    t_plaw_fit = msd.index[to_use]
    plaw_fit = np.exp(results_plaw["x"][1]) * t_plaw_fit ** results_plaw["x"][0]

    # #######################
    # plot results
    # #######################
    figh = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 2, hspace=0.5)
    nparticles_start = len(df[df["frame"] == 0])
    density_um3 = nparticles_start / vol
    plt.suptitle("%s\n" % root_dir + r"N=%d in first frame, $\rho$~%0.3g particles/$\mu m^3$ = %0.3g particle/ml $\to$ %0.3g $\mu m^3/particle$"
                 % (nparticles_start, density_um3, density_um3 * 1e12, 1/density_um3))

    # ensemble average MSD
    ax = plt.subplot(grid[0, 0])
    ax.plot(msd.index, msd, 'o')

    ax.plot(t_plaw_fit, plaw_fit, "magenta", label=r"$At^\alpha$, $\alpha$=%0.3f, A=%0.3f" % (results_plaw["x"][0], results_plaw["x"][1]))

    t_interp_short = np.linspace(0, tshort_cutoff, 300)
    ax.plot(t_interp_short, short_msd_slope * t_interp_short, "red", label=r"short time, D=%0.3g $\mu m^2/s$" % (short_msd_slope / 6))

    ax.set_title("Ensemble average MSD")
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    ax.set_xlabel('lag time (s)')
    ax.legend(loc='upper left')

    # individual msds
    ax = plt.subplot(grid[0, 1])
    ax.plot(msd_per_traj.index, msd_per_traj)

    ax.set_title("Trajectory MSD's")
    ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
    ax.set_xlabel('lag time (s)')

    # number of contributing trajectories
    ax = plt.subplot(grid[1, 0])
    plt.plot(msd.index, n_traj_contributing)

    ax.set_title("Contributing trajectories")
    ax.set_ylabel("number of trajs")
    ax.set_xlabel('lag time (s)')

    # step size distribution
    ax = plt.subplot(grid[1, 1])
    plt.plot(bin_centers, steps_x_hist, '.', color="pink", label="x")
    plt.plot(bin_centers, gauss_fit_x, 'r')

    plt.plot(bin_centers, steps_y_hist, '.', color=np.array([0, 0, 0.5]), label="y")
    plt.plot(bin_centers, gauss_fit_y, 'b')

    plt.plot(bin_centers, steps_z_hist, '.', color=np.array([0, 0.25, 0]), label="z")
    plt.plot(bin_centers, gauss_fit_z, 'g')

    ax.set_xlim([-0.5, 0.5])

    ax.set_title("Step size distribution (ensemble avg)\n" +
                 r"$\mu_x$=%0.3f $\mu m$, $\sigma_x$=%0.3f $\mu m$" % (results_x["x"][1], results_x["x"][2]) +
                 "\t" + r"$\mu_y$=%0.3f $\mu m$, $\sigma_y$=%0.3f $\mu m$" % (results_y["x"][1], results_y["x"][2]) +
                 "\n" + r"$\mu_z$=%0.3f $\mu m$, $\sigma_z$=%0.3f $\mu m$" % (results_z["x"][1], results_z["x"][2]))
    ax.set_ylabel("number")
    ax.set_xlabel('size (um)')
    ax.legend(loc='upper left')

    amp = np.max([results_x["x"][0], results_y["x"][0], results_z["x"][0]])
    ax.set_ylim([-0.2 * amp, 1.2 * amp])

    # plt.figure(figsize=figsize)
    # tp.plot_traj(linked)

    # #######################
    # save results
    # #######################
    fname = os.path.join(root_dir, "msd.png")
    figh.savefig(fname)

    # save track data
    fname = os.path.join(root_dir, "tracks.pkl")
    with open(fname, "wb") as f:
        pickle.dump(linked, f)