#!/usr/bin/env python

'''
Stage scanning OPM post-processing using numpy, numba, skimage, pyimagej, and npy2bdv.
Places all tiles in actual stage positions and places iterative rounds into the time axis of BDV H5 for alignment
Orthgonal interpolation method adapted from Vincent Maioli (http://doi.org/10.25560/68022)

Last updated: Shepherd 04/21
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import sys
import gc
import argparse
import time
from skimage.measure import block_reduce
from image_post_processing import deskew
from itertools import compress, product
import data_io
import zarr
import tifffile

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument 
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Process raw OPM data.")
    parser.add_argument("-i", "--ipath", type=str, help="supply the directory to be processed")
    parser.add_argument("-d", "--decon", type=int, default=0,
                        help="0: no deconvolution (DEFAULT), 1: deconvolution")
    parser.add_argument("-f", "--flatfield", type=int, default=0, help="0: No flat field (DEFAULT), 1: flat field (FIJI) 2: flat field (python)")
    parser.add_argument("-s", "--save_type", type=int, default=1, help="0: TIFF stack output, 1: BDV output (DEFAULT), 2: Zarr output")
    parser.add_argument("-z", "--z_down_sample",type=int, default=1, help="1: No downsampling (DEFAULT), n: Nx downsampling")
    args = parser.parse_args()

    input_dir_string = args.ipath
    decon_flag = args.decon
    flatfield_flag = args.flatfield
    save_type= args.save_type
    z_down_sample = args.z_down_sample

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # create parameter array from scan parameters saved by acquisition code
    df_metadata = data_io.read_metadata(input_dir_path / Path('scan_metadata.csv'))
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

    # create output directory
    if decon_flag == 0 and flatfield_flag == 0:
        output_dir_path = input_dir_path / 'deskew_output'
    elif decon_flag == 0 and flatfield_flag > 0 :
        output_dir_path = input_dir_path / 'deskew_flatfield_output'
    elif decon_flag == 1 and flatfield_flag == 0:
        output_dir_path = input_dir_path / 'deskew_decon_output'
    elif decon_flag == 1 and flatfield_flag > 1:
        output_dir_path = input_dir_path / 'deskew_flatfield_decon_output'
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Create TIFF if requested
    if (save_type==0):
        # create directory for data type
        tiff_output_dir_path = output_dir_path / Path('tiff')
        tiff_output_dir_path.mkdir(parents=True, exist_ok=True)
    # Create BDV if requested
    elif (save_type == 1):
        # create directory for data type
        bdv_output_dir_path = output_dir_path / Path('bdv')
        bdv_output_dir_path.mkdir(parents=True, exist_ok=True)

        # https://github.com/nvladimus/npy2bdv
        # create BDV H5 file with sub-sampling for BigStitcher
        bdv_output_path = bdv_output_dir_path / Path(root_name+'_bdv.h5')
        bdv_writer = npy2bdv.BdvWriter(str(bdv_output_path), 
                                       nchannels=num_ch, 
                                       ntiles=num_y*num_z, 
                                       subsamp=((1,1,1),(4,8,8),(8,16,16)),
                                       blockdim=((32, 128, 128),),
                                       compression=None)

        # create blank affine transformation to use for stage translation
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                                (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (   0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)
    # Create Zarr if requested
    elif (save_type == 2):
        # create directory for data type
        zarr_output_dir_path = output_dir_path / Path('zarr')
        zarr_output_dir_path.mkdir(parents=True, exist_ok=True)

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
        opm_data = root.zeros("opm_data", shape=(num_t, num_y*num_z, num_ch, nz, ny, nx), chunks=(1, 1, 1, 32, 128, 128), dtype=np.uint16)
        root = zarr.open(str(zarr_output_path), mode="rw")
        opm_data = root["opm_data"]
    
    # if retrospective flatfield is requested, import and open pyimagej in interactive mode
    # because BaSiC flat-fielding plugin cannot run in headless mode
    if flatfield_flag == 1:
        from image_post_processing import manage_flat_field
        import imagej
        import scyjava

        scyjava.config.add_option('-Xmx12g')
        plugins_dir = Path('/home/dps/Fiji.app/plugins')
        scyjava.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
        ij_path = Path('/home/dps/Fiji.app')
        ij = imagej.init(str(ij_path), headless=False)
        ij.ui().showUI()
        print('PyimageJ approach to flat fielding will be removed soon. Switch to GPU accelerated python BASIC code (-f 2).')
    elif flatfield_flag == 2:
        from image_post_processing import manage_flat_field_py

    # if decon is requested, import microvolution wrapper
    if decon_flag == 1:
        from image_post_processing import mv_decon

    # initialize counters
    timepoints_in_data = list(range(num_t)) 
    y_tile_in_data = list(range(num_y))
    z_tile_in_data = list(range(num_z)) 
    ch_in_BDV = list(range(n_active_channels))
    tile_idx=0

    # loop over all directories. Each directory will be placed as a "tile" into the BigStitcher file
    for (y_idx, z_idx) in product(y_tile_in_data,z_tile_in_data):
        for (t_idx, ch_BDV_idx) in product(timepoints_in_data, ch_in_BDV):

            ch_idx = channels_in_data[ch_BDV_idx]

            # open stage positions file
            stage_position_filename = Path('t'+str(t_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_stage_positions.csv')
            stage_position_path = input_dir_path / stage_position_filename
            # check to see if stage poisition file exists yet
            while(not(stage_position_filename.exists())):
                time.sleep(60)

            df_stage_positions = data_io.read_metadata(stage_position_path)

            stage_x = np.round(float(df_stage_positions['stage_x']),2)
            stage_y = np.round(float(df_stage_positions['stage_y']),2)
            stage_z = np.round(float(df_stage_positions['stage_z']),2)
            print('y tile '+str(y_idx+1)+' of '+str(num_y)+'; z tile '+str(z_idx+1)+' of '+str(num_z)+'; channel '+str(ch_BDV_idx+1)+' of '+str(n_active_channels))
            print('Stage location (um): x='+str(stage_x)+', y='+str(stage_y)+', z='+str(stage_z)+'.')

            # construct directory name
            current_tile_dir_path = Path(root_name+'_t'+str(t_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_1')
            tile_dir_path_to_load = input_dir_path / current_tile_dir_path

            # https://pycro-manager.readthedocs.io/en/latest/read_data.html
            dataset = Dataset(str(tile_dir_path_to_load))
            raw_data = data_io.return_data_numpy(dataset=dataset, time_axis=None, channel_axis=None, num_images=num_images, y_pixels=y_pixels,x_pixels=x_pixels)
    
            # perform flat-fielding
            if flatfield_flag == 1:
                print('Flatfield.')
                corrected_stack = manage_flat_field(raw_data,ij)
            elif flatfield_flag == 2:
                corrected_stack = manage_flat_field_py(raw_data)
            else:
                corrected_stack = raw_data
            del raw_data

            # deskew
            print('Deskew.')
            deskewed = deskew(data=np.flipud(corrected_stack),theta=theta,distance=scan_step,pixel_size=pixel_size)
            del corrected_stack

            # downsample in z due to oversampling when going from OPM to coverslip geometry
            if z_down_sample > 1:
                print('Downsample.')
                deskewed_downsample = block_reduce(deskewed, block_size=(z_down_sample,1,1), func=np.mean)
            else:
                deskewed_downsample = deskewed
            del deskewed

            # run deconvolution on deskewed image
            if decon_flag == 1:
                print('Deconvolve.')
                deskewed_downsample_decon = mv_decon(deskewed_downsample,ch_idx,deskewed_y_pixel,z_down_sample*deskewed_z_pixel)
            else:
                deskewed_downsample_decon = deskewed_downsample
            del deskewed_downsample

            # save deskewed image into TIFF stack
            if (save_type==0):
                print('Write TIFF stack')
                tiff_filename= root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.tiff'
                tiff_output_path = tiff_output_dir_path / Path(tiff_filename)
                tifffile.imwrite(str(tiff_output_path), deskewed_downsample_decon, imagej=True, resolution=(1/deskewed_x_pixel, 1/deskewed_y_pixel),
                                metadata={'spacing': (z_down_sample*deskewed_z_pixel), 'unit': 'um', 'axes': 'ZYX'})
                
                metadata_filename = root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.csv'
                metadata_output_path = tiff_output_dir_path / Path(metadata_filename)
                tiff_stage_metadata = [{'stage_x': float(stage_x),
                                        'stage_y': float(stage_y),
                                        'stage_z': float(stage_z)}]
                data_io.write_metadata(tiff_stage_metadata[0], metadata_output_path)

            elif (save_type==1):
                # create affine transformation for stage translation
                # swap x & y from instrument to BDV
                affine_matrix = unit_matrix
                affine_matrix[0,3] = (stage_y)/(deskewed_y_pixel)  # x-translation 
                affine_matrix[1,3] = (stage_x)/(deskewed_x_pixel)  # y-translation
                affine_matrix[2,3] = (-1*stage_z) / (z_down_sample*deskewed_z_pixel)  # z-translation

                # save tile in BDV H5 with actual stage positions
                print('Write into BDV H5.')
                bdv_writer.append_view(deskewed_downsample_decon, time=0, channel=ch_BDV_idx, 
                                        tile=tile_idx,
                                        voxel_size_xyz=(deskewed_x_pixel, deskewed_y_pixel, z_down_sample*deskewed_z_pixel), 
                                        voxel_units='um',
                                        calibration=(1,1,(z_down_sample*deskewed_z_pixel)/deskewed_y_pixel),
                                        m_affine=affine_matrix,
                                        name_affine = 'tile '+str(tile_idx)+' translation')

            elif (save_type==2):
                print('Write data into Zarr container')
                opm_data[t_idx, tile_idx, ch_BDV_idx, :, :, :] = deskewed_downsample_decon
                metadata_filename = root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.csv'
                metadata_output_path = zarr_output_dir_path / Path(metadata_filename)
                zarr_stage_metadata = [{'stage_x': float(stage_x),
                                        'stage_y': float(stage_y),
                                        'stage_z': float(stage_z)}]
                data_io.write_metadata(zarr_stage_metadata[0], metadata_output_path)

            # free up memory
            del deskewed_downsample_decon
            gc.collect()

        tile_idx=tile_idx+1

    if (save_type==2):
        # write BDV xml file
        # https://github.com/nvladimus/npy2bdv
        # bdv_writer.write_xml(ntimes=num_t)
        bdv_writer.write_xml()
        bdv_writer.close()

    # shut down pyimagej
    if (flatfield_flag == 1):
        ij.getContext().dispose()

    # exit
    print('Finished.')
    sys.exit()

# run
if __name__ == "__main__":
    main(sys.argv[1:])
    
