import pygpufit.gpufit as gf
import localize
import numpy as np
import matplotlib.pyplot as plt


# set scan parameters
theta = 30 * np.pi/180
normal = np.array([0, -np.sin(theta), np.cos(theta)]) # normal of camera pixel
# size and pixel size
nx = 15
ny = 15
npos = 11
dc = 0.115
dstep = dc * np.cos(theta) * 1
stage_pos = dstep * np.arange(npos)

# set physical parameters
na = 1.3
ni = 1.4
emission_wavelength = 0.605
sxy = 0.22 * emission_wavelength / na
sz = np.sqrt(6) / np.pi * ni * emission_wavelength / na ** 2

# coordinates
x, y, z = localize.get_lab_coords(nx, ny, dc, theta, stage_pos)
x, y, z = np.broadcast_arrays(x, y, z)

# simulated image
gt, c_gt = localize.simulate_img({"dc": dc, "dstep": dstep, "theta": theta, "shape": (npos, ny, nx)},
                           {"na": na, "ni": ni, "peak_photons": 1000, "background": 0.1, "emission_wavelength": emission_wavelength},
                           ncenters=1)
# camera and photon shot noise
img, _, _ = localize.simulate_img_noise(gt, 1, cam_gains=2, cam_offsets=100, cam_readout_noise_sds=5)

data = np.expand_dims(img.ravel(), axis=0)
data = np.concatenate((data, data), axis=0)
data = data.astype(np.float32)

user_info = np.concatenate((x.ravel(), y.ravel(), z.ravel()))
user_info = np.concatenate((user_info, user_info))
user_info = user_info.astype(np.float32)

init_p = np.expand_dims(np.array([1000, c_gt[0, 2], c_gt[0, 1], c_gt[0, 0], sxy, sz, 100]), axis=0)
init_p += np.random.uniform(-0.2, 0.2, size=init_p.shape)
init_p = np.concatenate((init_p, init_p))
init_p = init_p.astype(np.float32)

params, states, chi_sqrs, niter, time = gf.fit(data, None, gf.ModelID.GAUSS_3D_ARB, init_p, max_number_iterations=100,
                                               estimator_id=gf.EstimatorID.LSE, user_info=user_info)