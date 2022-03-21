#!/usr/bin/env python

'''
Galvo scanning OPM post-processing using numpy, numba, skimage, pyimagej, and npy2bdv.
Orthgonal interpolation method adapted from original description by Vincent Maioli (http://doi.org/10.25560/68022)

Last updated: Shepherd 06/21
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import sys
import argparse
from skimage.measure import block_reduce
from image_post_processing import deskew
from itertools import compress
from itertools import product
import data_io
import tifffile
import gc
import zarr

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Process raw OPM data.")
    parser.add_argument("-i", "--ipath", type=str, nargs="+", help="supply the directories to be processed")
    parser.add_argument("-d", "--decon", type=int, default=0, help="0: no deconvolution (DEFAULT), 1: deconvolution")
    parser.add_argument("-f", "--flatfield", type=int, default=0, help="0: No flat field (DEFAULT), 1: flat field")
    parser.add_argument("-k", "--deskew", type=int, default=1, help="0: no deskewing, 1: deskewing (DEFAULT)")
    parser.add_argument("-s", "--save_type", type=int, default=0, help="0: TIFF stack output (DEFAULT), 1: BDV output, 2: Zarr output")
    parser.add_argument("-t", "--tilt_orientation",type=str, default='new', help="new: new orientation (DEFAULT), prev: previous orientation")
    parser.add_argument("--time_steps", nargs='+', type=int, default=-1, help="-1: all time steps (DEFAULT), else list of time steps")
    parser.add_argument("--channels", nargs='+', type=int, default=-1, help="-1: all channels (DEFAULT), else list of all channels")
    parser.add_argument("--overwrite", type=int, default=0, help="0: do not overwrite existing folder (DEFAULT), 1: overwrite")
    args = parser.parse_args()

    input_dir_strings = args.ipath
    decon_flag = args.decon
    flatfield_flag = args.flatfield
    deskew_flag = args.deskew
    save_type= args.save_type
    tilt_orientation = args.tilt_orientation
    overwrite_flag = args.overwrite == 1


    # Loop over all user supplied directories for batch reconstruction
    for ii, input_dir_string in enumerate(input_dir_strings):
        print("Processing directory %d/%d" % (ii + 1, len(input_dir_strings)))

        # https://docs.python.org/3/library/pathlib.html
        # Create Path object to directory
        input_dir_path=Path(input_dir_string)

        # create parameter array from scan parameters saved by acquisition code
        df_metadata = data_io.read_metadata(input_dir_path.resolve().parents[0] / 'scan_metadata.csv')
        root_name = df_metadata['root_name']
        scan_type = df_metadata['scan_type']
        theta = df_metadata['theta']
        scan_step = df_metadata['scan_step']
        pixel_size = df_metadata['pixel_size']
        num_t = df_metadata['num_t']
        num_y = df_metadata['num_y']
        num_z  = df_metadata['num_z']
        num_ch = df_metadata['num_ch']
        num_images = df_metadata['scan_axis_positions']
        excess_images = 0
        y_pixels = df_metadata['y_pixels']
        x_pixels = df_metadata['x_pixels']
        chan_405_active = df_metadata['405_active']
        chan_488_active = df_metadata['488_active']
        chan_561_active = df_metadata['561_active']
        chan_635_active = df_metadata['635_active']
        chan_730_active = df_metadata['730_active']
        active_channels = [chan_405_active,chan_488_active,chan_561_active,chan_635_active,chan_730_active]
        channel_idxs = [0,1,2,3,4]
        channels_in_data = list(compress(channel_idxs, active_channels))
        n_active_channels = len(channels_in_data)
        if not (num_ch == n_active_channels):
            print('Channel setup error. Check metatdata file and directory names.')
            sys.exit()

        # calculate pixel sizes of deskewed image in microns
        deskewed_x_pixel = pixel_size / 1000.
        deskewed_y_pixel = pixel_size / 1000.
        deskewed_z_pixel = pixel_size / 1000.
        print('Deskewed pixel sizes before downsampling (um). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

        # amount of down sampling in z
        z_down_sample = 1

        # load dataset
        if str(input_dir_path).endswith('zarr'):
            dataset = zarr.open(input_dir_path, mode='r')
            im_type = 'zarr'
        else:
            dataset = Dataset(str(input_dir_path))
            im_type = 'pycro'

        # create output directory
        im_processes = []
        if decon_flag == 1:
            im_processes.append('decon')
        if flatfield_flag == 1:
            im_processes.append('flatfield')
        if deskew_flag == 1:
            im_processes.append('deskew')
        if len(im_processes) == 0:
            str_processes = 'original_output'
        else:
            str_processes = '_'.join(im_processes) + '_output'
        input_dir_path=Path(input_dir_string)
        output_dir_path = input_dir_path.resolve().parents[0] / str_processes
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # initialize counters
        timepoints_in_data = list(range(num_t))
        ch_in_BDV = list(range(n_active_channels))
        em_wavelengths=[.450,.520,.580,.670,.780]

        # if specific time steps or channels are provided, we use them only
        # by default -1, list of int if provided by user
        if not isinstance(args.time_steps, int):
            timepoints_in_data = args.time_steps
            num_t = len(timepoints_in_data)
        if not isinstance(args.channels, int):
            ch_in_BDV = args.channels
            num_ch = len(ch_in_BDV)

        # Create TIFF if requested
        if (save_type==0):
            # create directory for data type
            tiff_output_dir_path = output_dir_path / Path('tiff')
            tiff_output_dir_path.mkdir(parents=True, exist_ok=overwrite_flag)
        # Create BDV if requested
        elif (save_type == 1):
            # create directory for data type
            bdv_output_dir_path = output_dir_path / Path('bdv')
            bdv_output_dir_path.mkdir(parents=True, exist_ok=overwrite_flag)

            # https://github.com/nvladimus/npy2bdv
            # create BDV H5 file with sub-sampling for BigStitcher
            bdv_output_path = bdv_output_dir_path / Path(root_name+'_bdv.h5')
            bdv_writer = npy2bdv.BdvWriter(str(bdv_output_path), nchannels=num_ch, ntiles=1, subsamp=((1,1,1),),blockdim=((16, 16, 16),))

            # create blank affine transformation to use for stage translation
            unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                                    (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                                    (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)
        # Create Zarr if requested
        elif (save_type == 2):
            # create directory for data type
            zarr_output_dir_path = output_dir_path / Path('zarr')
            zarr_output_dir_path.mkdir(parents=True, exist_ok=overwrite_flag)

            # create name for zarr directory
            zarr_output_path = zarr_output_dir_path / Path(root_name + '_zarr.zarr')

            # calculate size of one volume
            # change step size from physical space (nm) to camera space (pixels)
            pixel_step = scan_step/pixel_size    # (pixels)

            # calculate the number of pixels scanned during stage scan 
            scan_end = num_images * pixel_step  # (pixels)

            # calculate properties for final image
            ny = np.int64(np.ceil(scan_end+y_pixels*np.cos(theta*np.pi/180))) # (pixels)
            nz = np.int64(np.ceil(y_pixels*np.sin(theta*np.pi/180)))          # (pixels)
            nx = np.int64(x_pixels)                                           # (pixels)

            # create and open zarr file
            root = zarr.open(str(zarr_output_path), mode="w")
            opm_data = root.zeros("opm_data", shape=(num_t, num_ch, nz, ny, nx), chunks=(1, 1, 32, 256, 256), dtype=np.uint16)
            root = zarr.open(str(zarr_output_path), mode="rw")
            opm_data = root["opm_data"]
        
        # if retrospective flatfield is requested, import and open pyimagej in interactive mode
        # TO DO: need to fix for new call
        if flatfield_flag == 1:
            from image_post_processing import manage_flat_field

        # if decon is requested, import microvolution wrapper
        if decon_flag == 1:
            from image_post_processing import lr_deconvolution, mv_lr_decon

        # loop over all timepoints and channels
        for (t_idx, ch_BDV_idx) in product(timepoints_in_data,ch_in_BDV):

            ch_idx = channels_in_data[ch_BDV_idx]

            # pull data stack into memory
            print('Process timepoint '+str(t_idx)+'; channel '+str(ch_BDV_idx) +'.')
            if im_type == 'pycro':
                raw_data = data_io.return_data_numpy(dataset, t_idx, ch_BDV_idx, num_images, excess_images, y_pixels, x_pixels)
            elif im_type == 'zarr':
                raw_data = dataset[t_idx, ch_BDV_idx, :, :, :]
            
            # run deconvolution on skewed image
            if decon_flag == 1:
                print('Deconvolve.')
                em_wvl = em_wavelengths[ch_idx]
                channel_opm_psf = data_io.return_opm_psf(em_wvl)
                if tilt_orientation == 'new':
                    channel_opm_psf = np.flip(channel_opm_psf, axis=1)
                decon = mv_lr_decon(image=raw_data,psf=channel_opm_psf,iterations=50)
                #decon = lr_deconvolution(image=raw_data,psf=channel_opm_psf,iterations=30)
            else:
                decon = raw_data
            del raw_data
            gc.collect()

            # perform flat-fielding
            if flatfield_flag == 0:
                corrected_stack = decon
            else:
                print('Flatfield.')
                corrected_stack, flat_field, dark_field = manage_flat_field(decon, ij)
            del decon
            gc.collect()

            # deskew raw_data
            if deskew_flag == 1:
                print('Deskew.')
                if tilt_orientation == 'new':
                    deskewed = deskew(data=np.flip(corrected_stack, axis=1),theta=theta,distance=scan_step,pixel_size=pixel_size)
                else:
                    deskewed = deskew(data=np.flip(corrected_stack, axis=0),theta=theta,distance=scan_step,pixel_size=pixel_size)
            else:
                if tilt_orientation == 'new':
                    deskewed = np.flip(corrected_stack, axis=1)
                else:
                    deskewed = np.flip(corrected_stack, axis=0)
            del corrected_stack
            gc.collect()

            # downsample in z due to oversampling when going from OPM to coverslip geometry
            if z_down_sample==1:
                downsampled = deskewed
            else:
                print('Downsample.')
                downsampled = block_reduce(deskewed, block_size=(z_down_sample,1,1), func=np.mean)
            del deskewed
            gc.collect()

            # save deskewed image into TIFF stack
            if (save_type==0):
                print('Write TIFF stack')
                tiff_filename= 'f_'+root_name+'_c'+str(ch_idx).zfill(3)+'_t'+str(t_idx).zfill(5)+'.tiff'
                tiff_output_path = tiff_output_dir_path / Path(tiff_filename)
                tifffile.imwrite(str(tiff_output_path), downsampled.astype(np.int16), imagej=True, resolution=(1/deskewed_x_pixel, 1/deskewed_y_pixel),
                                metadata={'unit': 'um', 'axes': 'ZYX'})
            # save tile in BDV H5 with actual stage positions
            elif (save_type==1):
                print('Write data into BDV H5.')
                bdv_writer.append_view(downsampled, time=t_idx, channel=ch_BDV_idx,
                                        tile=0,
                                        voxel_size_xyz=(deskewed_y_pixel, deskewed_y_pixel, z_down_sample*deskewed_z_pixel),
                                        voxel_units='um')

                # save deskewed image into Zarr container
            elif (save_type==2):
                print('Write data into Zarr container')
                opm_data[t_idx, ch_BDV_idx, :, :, :] = downsampled

            # free up memory
            del downsampled
            gc.collect()

        if (save_type==1):
            # write BDV xml file
            # https://github.com/nvladimus/npy2bdv
            bdv_writer.write_xml()
            bdv_writer.close()

    # shut down pyimagej
    if flatfield_flag==1:
        ij.getContext().dispose()

    # exit
    print('Finished.')
    sys.exit()

# run
if __name__ == "__main__":
    main(sys.argv[1:])
