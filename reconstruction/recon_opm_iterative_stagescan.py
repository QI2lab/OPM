#!/usr/bin/env python

'''
Stage scanning iterative labeling OPM post-processing using numpy, numba, skimage, pyimagej, and npy2bdv.
Places all tiles in actual stage positions and places iterative rounds into the time axis of BDV H5 for alignment
Orthgonal interpolation method adapted from Vincent Maioli (http://doi.org/10.25560/68022)

Last updated: Shepherd 05/21 - changes for iterative labeling experiments and ongoing issues with loading Pycromanager data into Dask
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import sys
import getopt
from skimage.measure import block_reduce
from image_post_processing import deskew
import pandas as pd
from itertools import compress
from itertools import product
import data_io

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument 
    input_dir_string = ''
    output_dir_string = ''
    decon_flag = 0
    flatfield_flag = 0

    try:
        arguments, values = getopt.getopt(argv,"hi:d:f:",["help","ipath=","decon=","flatfield="])
    except getopt.GetoptError:
        print('Error. recon_opm_stagescan.py -i <inputdirectory> -d <0: no deconvolution, 1: deconvolution> -f <0: no flat-field 1: flat-field>')
        sys.exit(2)
    for current_argument, current_value in arguments:
        if current_argument == '-h':
            print('Usage. recon_opm_stagescan.py -i <inputdirectory> -d <0: no deconvolution, 1: deconvolution> -f <0: no flat-field 1: flat-field>')
            sys.exit()
        elif current_argument in ("-i", "--ipath"):
            input_dir_string = str(current_value)
        elif current_argument in ("-d", "--decon"):
            decon_flag = int(current_value)
        elif current_argument in ("-f", "--flatfield"):
            flatfield_flag = int(current_value)

    if (input_dir_string == ''):
        print('Input directory parse error.')
        sys.exit(2)

    if not(decon_flag==0 or decon_flag==1):
        print('Deconvolution setting parse error.')
        sys.exit(2)

    if not(flatfield_flag==0 or flatfield_flag==1):
        print('Flatfield setting parse error.')
        sys.exit(2)

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
    theta = df_metadata['theta']
    scan_step = df_metadata['scan_step']
    pixel_size = df_metadata['pixel_size']
    num_r = df_metadata['num_r']
    num_y = df_metadata['num_y']
    num_z = df_metadata['num_z']
    num_ch = df_metadata['num_ch']
    num_images = df_metadata['scan_axis_positions']
    y_pixels = df_metadata['y_pixels']
    x_pixels = df_metadata['x_pixels']
    chan_405_active = df_metadata['405_active']
    chan_488_active = df_metadata['488_active']
    chan_561_active = df_metadata['561_active']
    chan_635_active = df_metadata['635_active']
    chan_730_active = df_metadata['730_active']

    # determine active channels in experiment
    active_channels = [chan_405_active,chan_488_active,chan_561_active,chan_635_active,chan_730_active]
    channel_idxs = [0,1,2,3,4]
    channels_in_data = list(compress(channel_idxs, active_channels))
    
    n_active_channels = len(channels_in_data)
    if not (num_ch == n_active_channels):
        print('Channel setup error. Check metatdata file and directory names.')
        sys.exit()
    num_ch = 1
    n_active_channels = 1
    channels_in_data = [1]

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)

    # https://github.com/nvladimus/npy2bdv
    # create BDV H5 file with sub-sampling for BigStitcher
    if decon_flag == 0 and flatfield_flag == 0:
        output_path = output_dir_path / 'full_deskew_only.h5'
    elif decon_flag == 0 and flatfield_flag == 1:
        output_path = output_dir_path / 'full_deskew_flatfield.h5'
    elif decon_flag == 1 and flatfield_flag == 0:
        output_path = output_dir_path / 'full_deskew_decon.h5'
    elif decon_flag == 1 and flatfield_flag == 1:
        output_path = output_dir_path / 'full_deskew_flatfield_decon.h5'

    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_ch, ntiles=(num_y*num_z), \
                                   subsamp=((1,1,1),(2,4,4),(4,8,8)),blockdim=((4, 256, 256),))

    # calculate pixel sizes of deskewed image in microns
    deskewed_x_pixel = pixel_size / 1000.
    deskewed_y_pixel = pixel_size / 1000.
    deskewed_z_pixel = pixel_size / 1000.
    
    print('Deskewed pixel sizes before downsampling (nm). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

    # amount of down sampling in z
    z_down_sample = 2

    # set up parameters for deskew parameters
    deskew_parameters = np.empty([3])
    deskew_parameters[0] = theta             # (degrees)
    deskew_parameters[1] = scan_step         # (nm)
    deskew_parameters[2] = pixel_size        # (nm)

    # create blank affine transformation to use for stage translation
    unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                            (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

    # if retrospective flatfield is requested, import and open pyimagej in interactive mode 
    # because BaSiC flat-fielding plugin cannot run in headless mode
    if flatfield_flag==1:
        pass
    # if decon is requested, import microvolution wrapper
    # this file is private and does not follow the same license as the rest of our code.
    if decon_flag==1:
        from image_post_processing import mv_decon

    # initialize tile counter
    rounds_in_data = list(range(num_r))
    y_in_data = list(range(num_y))
    z_in_data = list(range(num_z))
    ch_in_BDV = list(range(n_active_channels))

    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file
    for r_idx in rounds_in_data:
        
        #reset tile counter
        tile_idx = 0
        
        for (y_idx,z_idx) in product(y_in_data,z_in_data):
            for ch_BDV_idx in (ch_in_BDV):

                ch_idx = channels_in_data[ch_BDV_idx]

                # construct directory name
                current_tile_dir_path = Path(root_name+'_r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_1')
                tile_dir_path_to_load = input_dir_path / current_tile_dir_path
                
                # open stage positions file
                stage_position_filename = Path(root_name+'_r'+str(r_idx).zfill(4)+'_y'+str(y_idx).zfill(4)+'_z'+str(z_idx).zfill(4)+'_ch'+str(ch_idx).zfill(4)+'_stage_positions.csv')
                df_stage_positions = pd.read_csv(input_dir_path / stage_position_filename)

                stage_x = np.round(float(df_stage_positions['stage_x']),2)
                stage_y = np.round(float(df_stage_positions['stage_y']),2)
                stage_z = np.round(float(df_stage_positions['stage_z']),2)
                print('round '+str(r_idx+1)+' of '+str(num_r)+'; y tile '+str(y_idx+1)+' of '+str(num_y)+'; z tile '+str(z_idx+1)+' of '+str(num_z)+'; channel '+str(ch_BDV_idx+1)+' of '+str(n_active_channels))
                print('Stage location (um): x='+str(stage_x)+', y='+str(stage_y)+', z='+str(stage_z)+'.')

                # https://pycro-manager.readthedocs.io/en/latest/read_data.html
                dataset = Dataset(str(tile_dir_path_to_load))
                data_array = data_io.return_data_numpy(dataset, channel_axis=None, num_images=num_images,y_pixels=y_pixels,x_pixels=x_pixels)
            
                # perform flat-fielding
                if flatfield_flag == 1:
                    print('Flatfield.')
                    corrected_stack = manage_flat_field(data_array,ij)
                else:
                    corrected_stack=data_array
                del data_array

                # deskew
                print('Deskew.')
                deskewed = deskew(data=np.flipud(corrected_stack),parameters=deskew_parameters)
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
                
                # create affine transformation for stage translation
                # swap x & y from instrument to BDV
                affine_matrix = unit_matrix
                affine_matrix[0,3] = (stage_y)/(deskewed_y_pixel)  # x-translation 
                affine_matrix[1,3] = (stage_x)/(deskewed_x_pixel)  # y-translation
                affine_matrix[2,3] = (-1*stage_z) / (z_down_sample*deskewed_z_pixel)  # z-translation

                # save tile in BDV H5 with actual stage positions
                print('Write into BDV H5.')
                bdv_writer.append_view(deskewed_downsample_decon,
                                        time=r_idx, 
                                        channel=ch_BDV_idx, 
                                        tile=tile_idx,
                                        voxel_size_xyz=(deskewed_y_pixel, deskewed_y_pixel, z_down_sample*deskewed_z_pixel), 
                                        voxel_units='um',
                                        calibration=(1,1,z_down_sample*deskewed_z_pixel/deskewed_x_pixel),
                                        m_affine=affine_matrix,
                                        name_affine = 'tile '+str(tile_idx)+' translation')

                # free up memory
                del deskewed_downsample_decon
            
            tile_idx = tile_idx +1 
        
    # write BDV xml file
    # https://github.com/nvladimus/npy2bdv
    bdv_writer.write_xml()
    bdv_writer.close()

        # TO DO
        # Write function and macros to:
        # 1. Stitch in BigStitcher using polyA channel in all timepoints
        # 2. Run ICP for chromatic aberration with <5 pixel shift
        # 3. Run ICP for tile correction <1 pixel shift
        # 4. Run timepoint stabilization for polyA and optimize across all rounds
        # 5. Use npy2bdv to append transform from polyA registration within each timepoint
        # 6. Perform fusion into new H5 if requested
        #
        # Write function to:
        # 1. localize spots in raw data using Peter's GPU fitting
        # 2. Apply transformations to take raw fits -> deskewed, downsampled, stitched, and stabilized data
        # 3. Optimize localizations across rounds for barcoding if codebook is used


    # exit
    print('Finished.')
    sys.exit()

# run
if __name__ == "__main__":
    main(sys.argv[1:])