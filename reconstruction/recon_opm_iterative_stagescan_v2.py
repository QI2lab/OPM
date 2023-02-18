#!/usr/bin/env python
'''
QI2lab OPM suite
Reconstruction tools

Stage scanning iterative OPM post-processing using numpy, numba, skimage, cupy, dexp, and npy2bdv.
Places all tiles in actual stage positions and places time points OR iterative rounds into the time axis of BDV H5 for alignment
Orthgonal interpolation method adapted from Vincent Maioli (http://doi.org/10.25560/68022)

Change log: 
Shepherd 01/23 - breaking change for v2.0 acquisition approach.
Shepherd 01/23 - remove z downsampling and add option to select individual y & z tiles
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import sys
import gc
import argparse
from image_post_processing import deskew
from itertools import compress, product
import data_io
import zarr
import tifffile
import time
from skimage.measure import block_reduce

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Process raw OPM data.")
    parser.add_argument("-i", "--ipath", type=str, help="supply the directory to be processed")
    parser.add_argument("-o", "--opath", type=str, default="default", help="supply output directory (DEFAULT is input directory)")
    parser.add_argument("-d", "--decon", type=int, default=0,
                        help="0: no deconvolution (DEFAULT), 1: deconvolution")
    parser.add_argument("-f", "--flatfield", type=int, default=0, help="0: No flat field (DEFAULT), 1: flat field (python)")
    parser.add_argument("-s", "--save_type", type=int, default=1, help="0: TIFF stack output, 1: BDV output (DEFAULT), 2: Zarr output")
    parser.add_argument("-c", "--channel",type=int, default=5, help="5: All channels (DEFAULT), n: channel index (starting from 0)")
    parser.add_argument("-r", "--round",type=int, default=-1, help="0: All rounds (DEFAULT), n: round index (starting from 1)")
    parser.add_argument("-x", "--x_pos",type=int, default=-1, help="-1: All x positions (DEFAULT), n: x tile index (starting from 0)")
    parser.add_argument("-y", "--y_pos",type=int, default=-1, help="-1: All y positions (DEFAULT), n: y tile index (starting from 0)")
    parser.add_argument("-z", "--z_pos",type=int, default=-1, help="-1: All z positions (DEFAULT), n: z tile index (starting from 0)")
    
    args = parser.parse_args()

    input_dir_string = args.ipath
    output_dir_string = args.opath
    decon_flag = args.decon
    flatfield_flag = args.flatfield
    save_type= args.save_type
    ch_idx_to_use = args.channel
    r_idx_to_use = args.round
    x_idx_to_use = args.x_pos
    y_idx_to_use = args.y_pos
    z_idx_to_use = args.z_pos

    # Load data
    # Data must be generated by QI2lab pycromanager iterative OPM control code
    # https://www.github.com/qi2lab/OPM/

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # read metadata for this experiment
    df_metadata = data_io.read_metadata(input_dir_path / 'scan_metadata.csv')
    root_name = df_metadata['root_name']
    scan_type = df_metadata['scan_type']
    interleaved = df_metadata['interleaved']
    theta = df_metadata['theta']
    scan_step = df_metadata['scan_step']
    pixel_size = df_metadata['pixel_size']
    num_t = df_metadata['num_t']
    num_r = df_metadata['num_r']
    num_x = df_metadata['num_x']
    num_y = df_metadata['num_y']
    num_z = df_metadata['num_z']
    num_ch = df_metadata['num_ch']
    #num_images = df_metadata['scan_positions']
    excess_images = df_metadata['excess_scan_positions']
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

    if not(ch_idx_to_use==5):
        active_channels = [False,False,False,False,False]
        active_channels[ch_idx_to_use] = True
        channels_in_data = list(compress(channel_idxs, active_channels))
        n_active_channels = len(channels_in_data)
        one_channel_flag = True
    else:
        one_channel_flag = False

    if (r_idx_to_use == -1):
        num_r = num_r - 1
        rounds_in_data = list(range(num_r))
    else:
        num_r=1
        rounds_in_data = [r_idx_to_use]

    if (x_idx_to_use == -1):
        num_x = num_x
        x_tile_in_data = list(range(num_x))
    else:
        num_x = 1
        x_tile_in_data = [x_idx_to_use]

    if (y_idx_to_use == -1):
        num_y = num_y
        y_tile_in_data = list(range(num_y))
    else:
        num_y = 1
        y_tile_in_data = [y_idx_to_use]

    if (z_idx_to_use == -1):
        num_z = num_z
        z_tile_in_data = list(range(num_z))
    else:
        num_z = 1
        z_tile_in_data = [z_idx_to_use]

    # calculate pixel sizes of deskewed image in microns
    deskewed_x_pixel = pixel_size / 1000.
    deskewed_y_pixel = pixel_size / 1000.
    deskewed_z_pixel = pixel_size / 1000.
    print('Deskewed pixel sizes before downsampling (um). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

    # create output directory
    if output_dir_string == "default":
        output_dir_path_base = input_dir_path
    else:
        output_dir_path_base = Path(output_dir_string)

    if decon_flag == 0 and flatfield_flag == 0:
        output_dir_path = output_dir_path_base / 'deskew_output'
    elif decon_flag == 0 and flatfield_flag > 0 :
        output_dir_path = output_dir_path_base / 'deskew_flatfield_output'
    elif decon_flag == 1 and flatfield_flag == 0:
        output_dir_path = output_dir_path_base / 'deskew_decon_output'
    elif decon_flag == 1 and flatfield_flag == 1:
        output_dir_path = output_dir_path_base / 'deskew_flatfield_decon_output'
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
                                       nchannels=n_active_channels+1,
                                       ntiles=num_x*num_y*num_z,
                                       subsamp=((1,1,1), (4,8,8),(8,16,16)),
                                       blockdim=((8,256,256),(4,384,384),(2,512,512)))

        # create blank affine transformation to use for stage translation
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                                (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                                (0.0, 0.0, 1.0, 0.0))) # change the 4. value for z_translation (px)
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
        # TO DO: this is a tricky now, since the number of scan pixels varies
        #        need to think about how to deal with this. My guess is to write
        #        the largest scan pixel to disk in the metadata and use that? It
        #        should be fine if we don't fill the array. Will just have to be
        #        careful with indexing.
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

    # if retrospective flatfield is requested, import GPU flatfield code
    if flatfield_flag==1:
        from image_post_processing import manage_flat_field_py
    # if decon is requested, import GPU deconvolution code
    if decon_flag==1:
        from image_post_processing import lr_deconvolution

    # initialize tile counter
    timepoints_in_data = list(range(num_t))
    ch_in_BDV = list(range(n_active_channels))
    em_wavelengths=[.420,.520,.580,.670,.780]
    tile_idx=0

    channel_ids = ['ch405','ch488','ch561','ch635','ch730']
    first_flatfield = True

    # loop over all rounds.
    for r_idx in rounds_in_data:
        tile_idx = 0
        # loop over all tiles. Each unique round/tile will be placed as a "tile" into the BigStitcher file 
        for (x_idx,y_idx, z_idx) in product(x_tile_in_data,y_tile_in_data,z_tile_in_data):
            # open stage positions file
            stage_position_filename = Path(root_name+'_r'+str(r_idx+1).zfill(4)+'_x'+str(x_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_stage_positions.csv')
            stage_position_path = input_dir_path / stage_position_filename

            read_metadata = False
            retry_flag = 0
            # check if this is the first tile. If so, wait longer for fluidics
            if x_idx == 0 and y_idx == 0 and z_idx == 0:
                while(not(read_metadata) and (retry_flag <4)):
                    try:
                        df_stage_positions = data_io.read_metadata(stage_position_path)
                    except:
                        read_metadata = False
                        retry_flag = retry_flag + 1
                        print(data_io.time_stamp(), "New round, initial stage position not found. Wait 20 minutes and try again.")
                        time.sleep(1)
                    else:
                        read_metadata = True
            else:
                while(not(read_metadata) and (retry_flag <4)):
                    try:
                        df_stage_positions = data_io.read_metadata(stage_position_path)
                    except:
                        read_metadata = False
                        retry_flag = retry_flag + 1
                        print(data_io.time_stamp(), "Stage position not found. Wait 1 minute and try again.")
                        time.sleep(1)
                    else:
                        read_metadata = True
                    
            if retry_flag == 4:
                skip_tile = True
                print(data_io.time_stamp(), "Timeout occurred. Skipping this stage position.")
            else:
                skip_tile = False

            if not(skip_tile):

                # grab recorded stage positions
                stage_x = np.round(float(df_stage_positions['stage_x']),2)
                stage_y = np.round(float(df_stage_positions['stage_y']),2)
                stage_z = np.round(float(df_stage_positions['stage_z']),2)

                # construct directory name
                current_tile_dir_path = Path(root_name+'_r'+str(r_idx+1).zfill(4)+'_x'+str(x_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_1')
                tile_dir_path_to_load = input_dir_path / current_tile_dir_path

                # https://pycro-manager.readthedocs.io/en/latest/read_data.html
                dataset = Dataset(str(tile_dir_path_to_load))

                for (t_idx, ch_BDV_idx) in product(timepoints_in_data, ch_in_BDV):

                    ch_idx = channels_in_data[ch_BDV_idx]
                    channel_id = channel_ids[ch_idx]
                    # deal with last round channel idx change via brute force right now
                    # need to fix more elegantly!
                    #if r_idx == (num_r-1) and one_channel_flag:
                    #    ch_idx = ch_idx + 1

                    print(data_io.time_stamp(), 'round '+str(r_idx+1)+' of '+str(num_r)+'; x tile '+str(x_idx+1)+' of '+str(num_x)+'; y tile '+str(y_idx+1)+' of '+str(num_y)+'; z tile '+str(z_idx+1)+' of '+str(num_z)+'; channel '+str(ch_BDV_idx+1)+' of '+str(n_active_channels))
                    print(data_io.time_stamp(), 'Stage location (um): x='+str(stage_x)+', y='+str(stage_y)+', z='+str(stage_z)+'.')

                    # load raw data using Dask interface to NDTIFF
                    raw_data = data_io.return_data_dask(dataset,excess_images,channel_id)

                    # run deconvolution on skewed image
                    if decon_flag == 1:
                        print(data_io.time_stamp(), 'Deconvolve.')
                        em_wvl = em_wavelengths[ch_idx]
                        channel_opm_psf = np.flip(data_io.return_opm_psf(em_wvl,z_idx),axis=1)
                        decon = lr_deconvolution(image=raw_data,psf=channel_opm_psf,iterations=30)
                    else:
                        decon = raw_data
                    del raw_data
                    gc.collect()

                    # perform flat-fielding
                    if flatfield_flag == 1:
                        print(data_io.time_stamp(), 'Flatfield.')

                        if (first_flatfield):
                            flat_field = np.zeros([num_y,num_z,n_active_channels,y_pixels,x_pixels])
                            dark_field = np.zeros([num_y,num_z,n_active_channels,y_pixels,x_pixels])
                            first_flatfield = False

                        corrected_stack, flat_field[y_idx,z_idx,ch_BDV_idx,:], dark_field[y_idx,z_idx,ch_BDV_idx,:] = manage_flat_field_py(decon)

                    else:
                        corrected_stack = decon
                    del decon
                    gc.collect()

                    # deskew
                    print(data_io.time_stamp(), 'Deskew.')
                    # maybe just skip np.flipud, but have to check if it doesn't flip the major axis, which is z
                    deskewed = deskew(data=corrected_stack,theta=theta,distance=scan_step,pixel_size=pixel_size)
                    del corrected_stack
                    gc.collect()

                    # print('Downsample')
                    # downsampled = block_reduce(image=deskewed,
                    #                           block_size=(2,2,2),
                    #                           func=np.mean,
                    #                           func_kwargs={'dtype': np.uint16})
                    # del deskewed
                    # gc.collect()

                    # save deskewed image into TIFF stack
                    if (save_type==0):
                        print(data_io.time_stamp(), 'Write TIFF stack')
                        tiff_filename= root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.tiff'
                        tiff_output_path = tiff_output_dir_path / Path(tiff_filename)
                        tifffile.imwrite(str(tiff_output_path), deskewed, imagej=True, resolution=(1/deskewed_x_pixel, 1/deskewed_y_pixel),
                                        metadata={'spacing': (deskewed_z_pixel), 'unit': 'um', 'axes': 'ZYX'})

                        metadata_filename = root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.csv'
                        metadata_output_path = tiff_output_dir_path / Path(metadata_filename)
                        tiff_stage_metadata = [{'stage_x': float(stage_x),
                                                'stage_y': float(stage_y),
                                                'stage_z': float(stage_z)}]
                        data_io.write_metadata(tiff_stage_metadata[0], metadata_output_path)

                    elif (save_type==1):
                        corner_crop = 445

                        # create affine transformation for stage translation
                        # swap x & y from instrument to BDV
                        affine_matrix = unit_matrix
                        affine_matrix[0,3] = (stage_y)/(deskewed_y_pixel)  # BDV x-translation (tile axis)
                        if interleaved:
                            affine_matrix[1,3] = (stage_x)/(deskewed_x_pixel)-((scan_step/1000)/deskewed_x_pixel)/(num_ch)*ch_BDV_idx  # BDV y-translation (scan axis)
                        else:
                            affine_matrix[1,3] = (stage_x)/(deskewed_x_pixel)  # BDV y-translation (scan axis)
                        affine_matrix[2,3] = (-1*stage_z) / (deskewed_z_pixel)  # BDV z-translation (height axis)

                        # save tile in BDV H5 with actual stage positions
                        print(data_io.time_stamp(), 'Write into BDV H5.')
                        #print('Channel:' + str(ch_idx))
                        bdv_writer.append_view(deskewed[:,corner_crop:-corner_crop,:], time=r_idx, channel=ch_idx,
                                                tile=tile_idx,
                                                voxel_size_xyz=(deskewed_x_pixel, deskewed_y_pixel, deskewed_z_pixel),
                                                voxel_units='um',
                                                calibration=(1,1,deskewed_z_pixel/deskewed_y_pixel),
                                                m_affine=affine_matrix,
                                                name_affine = 'tile '+str(tile_idx)+' translation')

                    elif (save_type==2):
                        print(data_io.time_stamp(), 'Write data into Zarr container')
                        opm_data[t_idx, tile_idx, ch_BDV_idx, :, :, :] = deskewed
                        metadata_filename = root_name+'_t'+str(t_idx).zfill(3)+'_p'+str(tile_idx).zfill(4)+'_c'+str(ch_idx).zfill(3)+'.csv'
                        metadata_output_path = zarr_output_dir_path / Path(metadata_filename)
                        zarr_stage_metadata = [{'stage_x': float(stage_x),
                                                'stage_y': float(stage_y),
                                                'stage_z': float(stage_z)}]
                        data_io.write_metadata(zarr_stage_metadata[0], metadata_output_path)

                    # free up memory
                    del deskewed
                    gc.collect()
                
                dataset.close()
                del dataset
                gc.collect()

            tile_idx=tile_idx+1
            bdv_writer.write_xml() # try to update XML on the fly for viewing


    if (save_type==1):
        # created downsampled views with compression and write BDV xml file
        # https://github.com/nvladimus/npy2bdv
        bdv_writer.write_xml()
        bdv_writer.close()

    if (flatfield_flag==1):
        flatfield_filename= root_name+'flat_field.tiff'
        flatfield_output_path = output_dir_path / Path(flatfield_filename)
        tifffile.imwrite(str(flatfield_output_path), flat_field.astype(np.float32), imagej=True)

        darkfield_filename= root_name+'dark_field.tiff'
        darkfield_output_path = output_dir_path / Path(darkfield_filename)
        tifffile.imwrite(str(darkfield_output_path), dark_field.astype(np.float32), imagej=True)

    # exit
    print('Finished.')
    sys.exit()

# run
if __name__ == "__main__":
    main(sys.argv[1:])
