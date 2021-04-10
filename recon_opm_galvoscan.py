#!/usr/bin/env python

'''
Galvo scanning OPM post-processing using numpy, numba, skimage, pyimagej, and npy2bdv.
Orthgonal interpolation method adapted from original description by Vincent Maioli (http://doi.org/10.25560/68022)

Last updated: Shepherd 04/21
'''

# imports
import numpy as np
from pathlib import Path
from pycromanager import Dataset
import npy2bdv
import sys
import getopt
import re
from skimage.measure import block_reduce
import skimage.io
from image_post_processing import deskew
import pandas as pd
import dask.array as da
import glob
from itertools import compress
from itertools import product

# parse experimental directory, load data, perform orthogonal deskew, and save as BDV H5 file
def main(argv):

    # parse directory name from command line argument 
    input_dir_string = ''
    output_dir_string = ''
    acq_type = 0
    decon_flag = 0
    flatfield_flag = 0

    try:
        arguments, values = getopt.getopt(argv,"hi:a:d:f:",["help","ipath=","acq=","decon=","flatfield="])
    except getopt.GetoptError:
        print('Error. recon_opm_galvoscan.py -i <inputdirectory> -a <0: pycromanager (DEFAULT), 1: hcimage> -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>')
        sys.exit(2)
    for current_argument, current_value in arguments:
        if current_argument == '-h':
            print('Usage. recon_opm_galvoscan.py -i <inputdirectory> -a <0: pycromanager (DEFAULT), 1: hcimage> -d <0: no deconvolution (DEFAULT), 1: deconvolution> -f <0: no flat-field (DEFAULT), 1: flat-field>')
            sys.exit()
        elif current_argument in ("-i", "--ipath"):
            input_dir_string = str(current_value)
        elif current_argument in ("-a", "--acq"):
            acq_type = int(current_value)
        elif current_argument in ("-d", "--decon"):
            decon_flag = int(current_value)
        elif current_argument in ("-f", "--flatfield"):
            flatfield_flag = int(current_value)
        
    if (input_dir_string == ''):
        print('Input directory parse error.')
        sys.exit(2)

    if not(acq_type==0 or acq_type==1):
        print('Acquisiton type parse error.')
        sys.exit(2)

    if not(decon_flag==0 or decon_flag==1):
        print('Deconvolution setting parse error.')
        sys.exit(2)

    if not(flatfield_flag==0 or flatfield_flag==1):
        print('Flatfield setting parse error.')
        sys.exit(2)


    # Load data
    # Data must be generated by QI2lab pycromanager control code
    # https://www.github.com/qi2lab/OPM/

    # https://docs.python.org/3/library/pathlib.html
    # Create Path object to directory
    input_dir_path=Path(input_dir_string)

    # create parameter array from scan parameters saved by acquisition code
    # [timepoints, channels, scan positions, y pixels, x pixels, theta, stage move distance, camera pixel size]
    # units are [degrees,nm,nm]
    if acq_type==0:
        df_metadata = pd.read_csv(input_dir_path.resolve().parents[0] / 'scan_metadata.csv')
    else:
        df_metadata = pd.read_csv(input_dir_path / 'scan_metadata.csv')

    root_name = str(df_metadata['root_name'][0])
    scan_type = str(df_metadata['scan_type'][0])
    theta = float(df_metadata['theta'][0])
    scan_step = float(df_metadata['scan_step'][0])
    pixel_size = float(df_metadata['pixel_size'][0])
    num_t = int(df_metadata['num_t'][0])
    num_y = int(df_metadata['num_y'][0])
    num_z  = int(df_metadata['num_z'][0])
    num_ch = int(df_metadata['num_ch'][0])
    num_images = int(df_metadata['scan_axis_positions'][0])
    y_pixels = int(df_metadata['y_pixels'][0])
    x_pixels = int(df_metadata['x_pixels'][0])
    chan_405_active = df_metadata['405_active'][0]
    chan_488_active = df_metadata['488_active'][0]
    chan_561_active = df_metadata['561_active'][0]
    chan_635_active = df_metadata['635_active'][0]
    chan_730_active = df_metadata['730_active'][0]
    active_channels = [chan_405_active,chan_488_active,chan_561_active,chan_635_active,chan_730_active]
    channel_idxs = [0,1,2,3,4]
    channels_in_data = list(compress(channel_idxs, active_channels))
    n_active_channels = len(channels_in_data)
    if not (num_ch == n_active_channels):
        print('Channel setup error. Check metatdata file and directory names.')
        sys.exit()

    # check if user provided output path
    if (output_dir_string==''):
        output_dir_path = input_dir_path
    else:
        output_dir_path = Path(output_dir_string)
  
    # load data
    if acq_type==0:
        dataset = Dataset(str(input_dir_path))
        dask_array_raw = dataset.as_array()
        dask_array_raw = np.squeeze(dask_array_raw)
        # check which axes match experimental metadata
        dataset_shape = dask_array_raw.shape
        for i in range(len(dataset_shape)):
            if not(num_t==1) and (dataset_shape[i]==num_t):
                timepoint_axis = i
            elif not(num_ch==1) and (dataset_shape[i]==num_ch):
                channel_axis = i
            elif  dataset_shape[i]==num_images:
                scan_axis = i
            elif  dataset_shape[i]==y_pixels:
                y_axis = i
            elif  dataset_shape[i]==x_pixels:
                x_axis = i
        # reorder dask array to [time,channel,scan,y,x]
        if len(dataset_shape) == 5:
            dask_array = da.moveaxis(dask_array_raw,[timepoint_axis,channel_axis,scan_axis,y_axis,x_axis],[0,1,2,3,4])
        elif len(dataset_shape) == 4:
            if (num_ch==1):
                dask_array = da.moveaxis(dask_array_raw,[timepoint_axis,scan_axis,y_axis,x_axis],[0,1,2,3])
            elif (num_t==1):
                dask_array = da.moveaxis(dask_array_raw,[channel_axis,scan_axis,y_axis,x_axis],[0,1,2,3])
        elif len(dataset_shape) == 3:
            dask_array = da.moveaxis(dask_array_raw,[scan_axis,y_axis,x_axis],[0,1,2])

        del dataset
        del dask_array_raw

    else:
        all_images = sorted(glob.glob(str(input_dir_path / '*.tif')))
        array_images = np.zeros((len(all_images), y_pixels,x_pixels), dtype=np.uint16)
        for idx, image in enumerate(all_images):
            array_images[idx] = skimage.io.imread(image)
        dask_array = da.squeeze(da.from_array(np.reshape(array_images,[num_t,num_ch,num_images,y_pixels,x_pixels]),chunks=[1,1,y_pixels,x_pixels]))

    # check if user provided output path
    if acq_type==0:
        if (output_dir_string==''):
            output_dir_path = input_dir_path.resolve().parents[0]
        else:
            output_dir_path = Path(output_dir_string)
    else:
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

    bdv_writer = npy2bdv.BdvWriter(str(output_path), nchannels=num_ch, ntiles=1, subsamp=((1,1,1),),blockdim=((16, 16, 16),))

    # calculate pixel sizes of deskewed image in microns
    # Cannot use a different pixel size in (x,y) in BigStitcher, so calculate here for posterity
    deskewed_x_pixel = pixel_size / 1000.
    deskewed_y_pixel = pixel_size / 1000.
    deskewed_z_pixel = pixel_size / 1000.
    print('Deskewed pixel sizes before downsampling (nm). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

    # set up parameters for deskew parameters
    deskew_parameters = np.empty([3])
    deskew_parameters[0] = theta             # (degrees)
    deskew_parameters[1] = scan_step         # (nm)
    deskew_parameters[2] = pixel_size        # (nm)

    # amount of down sampling in z
    z_down_sample = 1

    # create blank affine transformation to use for stage translation
    unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                            (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)

    # if retrospective flatfield is requested, import and open pyimagej in interactive mode 
    # because BaSiC flat-fielding plugin cannot run in headless mode
    if flatfield_flag==1:
        from image_post_processing import manage_flat_field
        import imagej
        import scyjava
        from scyjava import jimport

        scyjava.config.add_option('-Xmx12g')
        plugins_dir = Path('/home/dps/Fiji.app/plugins')
        scyjava.config.add_option(f'-Dplugins.dir={str(plugins_dir)}')
        ij_path = Path('/home/dps/Fiji.app')
        ij = imagej.init(str(ij_path), headless=False)
        ij.ui().showUI()

    # if decon is requested, import microvolution wrapper
    if decon_flag==1:
        from image_post_processing import mv_decon

    # initialize tile counter
    tile_idx = 0
    timepoints_in_data = list(range(num_t))
    y_in_data = list(range(num_y))
    z_in_data = list(range(num_z))
    ch_in_BDV = list(range(n_active_channels))
    
    # loop over each directory. Each directory will be placed as a "tile" into the BigStitcher file
    for (t_idx, ch_BDV_idx) in product(timepoints_in_data,ch_in_BDV):

        ch_idx = channels_in_data[ch_BDV_idx]

        # pull data stack
        print('Process timepoint '+str(t_idx)+'; channel '+str(ch_BDV_idx) +'.')
        if (num_t == 1) and (num_ch==1):
            sub_stack = dask_array
        elif (num_t == 1) and not(num_ch==1):
            sub_stack = dask_array[ch_BDV_idx,:,:,:]
        elif not(num_t == 1) and (num_ch==1):
            sub_stack = dask_array[t_idx,:,:,:]
        elif not(num_t == 1) and not(num_ch==1):
            sub_stack = dask_array[t_idx,ch_BDV_idx,:,:,:]

        # perform
        #  flat-fielding
        if flatfield_flag == 0:
            corrected_stack=sub_stack.compute()
        else:
            print('Flatfield.')
            corrected_stack = manage_flat_field(sub_stack,ij)
        del sub_stack

        # deskew
        print('Deskew.')
        deskewed = deskew(data=np.flipud(corrected_stack),parameters=deskew_parameters)
        del corrected_stack

        # run deconvolution on deskewed image
        if decon_flag == 0:
            deskewed_decon = deskewed
        else:
            print('Deconvolve.')
            deskewed_decon = mv_decon(deskewed,ch_idx,deskewed_y_pixel,deskewed_z_pixel)
        del deskewed
                
        # downsample in z due to oversampling when going from OPM to coverslip geometry
        if z_down_sample==1:
            deskewed_downsample = deskewed_decon
        else:
            print('Downsample.')
            deskewed_downsample = block_reduce(deskewed_decon, block_size=(z_down_sample,1,1), func=np.mean)
        del deskewed_decon

        # save tile in BDV H5 with actual stage positions
        print('Write data into BDV H5.')
        bdv_writer.append_view(deskewed_downsample, time=t_idx, channel=ch_BDV_idx, 
                                tile=0,
                                voxel_size_xyz=(deskewed_y_pixel, deskewed_y_pixel, z_down_sample*deskewed_z_pixel), 
                                voxel_units='um')

        # free up memory
        del deskewed_downsample

    # write BDV xml file
    # https://github.com/nvladimus/npy2bdv
    bdv_writer.write_xml_file(ntimes=num_t)
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