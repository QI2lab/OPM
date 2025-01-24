import psfmodels as psfm
import numpy as np
#import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# ROI tools
def get_skewed_roi_size(sizes, theta, dc, dstep, ensure_odd=True):
    """
    Get ROI size in OPM matrix that includes sufficient xy and z points
    :param sizes: [z-size, y-size, x-size] in same units as dc, dstep
    :param theta: angle in radians
    :param dc: camera pixel size
    :param dstep: step size
    :param bool ensure_odd:
    :return [no, n1, n2]: integer size of roi in skewed coordinates
    """

    # x-size determines n2 size
    n2 = int(np.ceil(sizes[2] / dc))

    # z-size determines n1
    n1 = int(np.ceil(sizes[0] / dc / np.sin(theta)))

    # set so that @ top and bottom z-points, ROI includes the full y-size
    n0 = int(np.ceil((0.5 * (n1 + 1)) * dc * np.cos(theta) + sizes[1]) / dstep)

    if ensure_odd:
        if np.mod(n2, 2) == 0:
            n2 += 1

        if np.mod(n1, 2) == 0:
            n1 += 1

        if np.mod(n0, 2) == 0:
            n0 += 1

    return [n0, n1, n2]

# coordinate transformations between OPM and coverslip frames
def get_skewed_coords(sizes, dc, ds, theta, scan_direction="lateral"):
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a stage-scanning OPM set
    :param sizes: (n0, n1, n2)
    :param dc: camera pixel size
    :param ds: stage step size
    :param theta: in radians
    :return x, y, z:
    """
    nimgs, ny_cam, nx_cam = sizes

    if scan_direction == "lateral":
        x = dc * np.arange(nx_cam)[None, None, :]
        # y = stage_pos[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        y = ds * np.arange(nimgs)[:, None, None] + dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    elif scan_direction == "axial":
        x = dc * np.arange(nx_cam)[None, None, :]
        y = dc * np.cos(theta) * np.arange(ny_cam)[None, :, None]
        z = ds * np.arange(nimgs)[:, None, None] + dc * np.sin(theta) * np.arange(ny_cam)[None, :, None]
    else:
        raise ValueError("scan_direction must be `lateral` or `axial` but was `%s`" % scan_direction)

    return x, y, z

def create_psf_silicone_100x(dxy, dz, nxy, nz, ex_NA,ex_wvl,em_wvl):
    """
    Create OPM PSF in coverslip coordinates
    :param dxy: spacing of xy pixels
    :param dz: spacing of z planes
    :param nxy: number of xy pixels on a side
    :param nz: number of z planes
    :param ex_NA: excitaton NA
    :param ex_wvl: excitation wavelength in microns
    :param em_wvl: emission wavelength in microns 
    :return tot_psf:
    """

    silicone_lens = {
        'ni0': 1.4, # immersion medium RI design value
        'ni': 1.4,  # immersion medium RI experimental value
        'ns': 1.45,  # specimen refractive index
        'tg': 170, # microns, coverslip thickness
        'tg0': 170 # microns, coverslip thickness design value
    }
    ex_lens = {**silicone_lens, 'NA': ex_NA}
    em_lens = {**silicone_lens, 'NA': 1.35}

    # The psf model to use
    # can be any of {'vectorial', 'scalar', or 'microscpsf'}
    func = 'vectorial'

    # the main function
    _, _, tot_psf = psfm.tot_psf(nx=nxy, nz=nz, dxy=dxy, dz=dz, pz=15,
                                        x_offset=0, z_offset=0,
                                        ex_wvl = ex_wvl, em_wvl = em_wvl,
                                        ex_params=ex_lens, em_params=em_lens,
                                        psf_func=func)    
    return tot_psf

def generate_skewed_psf(ex_NA,ex_wvl,em_wvl,dstage=0.4):
    """
    Create OPM PSF in skewed coordinates
    :param ex_NA: excitaton NA
    :param ex_wvl: excitation wavelength in microns
    :param em_wvl: emission wavelength in microns 
    :return skewed_psf:
    """
    
    dc = 0.115
    na = 1.35
    ni = 1.4
    theta = 30 * np.pi/180

    xy_res = 1.6163399561827614 / np.pi * em_wvl / na
    z_res = 2.355*(np.sqrt(6) / np.pi * ni * em_wvl / na ** 2)

    roi_skewed_size = get_skewed_roi_size([z_res * 9, xy_res * 9, xy_res * 9],
                                          theta, dc, dstage, ensure_odd=True)
    # make square
    roi_skewed_size[2]= roi_skewed_size[1]

    # get tilted coordinates
    x, y, z = get_skewed_coords(roi_skewed_size, dc, dstage, theta)
    dx = x[0, 0, 1] - x[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dz = z[0, 1, 0] - z[0, 0, 0]


    z -= z.mean()
    x -= x.mean()
    y -= y.mean()

    # get on grid of coordinates
    dxy = 0.5 * np.min([dx, dy])
    dz = 0.5 * dz
    print(dxy,dz)
    
    nxy = np.max([int(2 * ((x.max() - x.min()) // dxy) + 1),
                  int(2 * ((y.max() - y.min()) // dxy) + 1)])
    nz = z.size

    xg = np.arange(nxy) * dxy
    xg -= xg.mean()
    yg = np.arange(nxy) * dxy
    yg -= yg.mean()
    print(xg[1]-xg[0])

    psf_grid = create_psf_silicone_100x(dxy, dz, nxy, nz, ex_NA,ex_wvl,em_wvl)

    # get value from interpolation
    skewed_psf = np.zeros(roi_skewed_size)
    for ii in range(nz):
        skewed_psf[:, ii, :] = interp2d(xg, yg, psf_grid[ii], kind="linear")(x.ravel(), y[:, ii].ravel())

    return skewed_psf / np.sum(skewed_psf,axis=(0,1,2))