import zarr
from numcodecs import Blosc
import dask.array as da
from pathlib import Path
from _imageprocessing import deskew, run_bigstitcher, perform_local_registration, generate_flatfield, apply_flatfield
import numpy as np
from itertools import product
import gc
import _dataio as data_io
import npy2bdv
import sys

ch_BDV_ids = [0,1,2]
channels=['ch488','ch561','ch635']
num_r = 8
num_x = 1
num_y = 7
num_z = 1
pixel_size = 115
scan_step = 400
theta = 30
bdv_pixel_size = .115

root_name = 'cells'
output_dir_path = Path(r'D:\20230630_cells_MERFISH_alldeskew\processed')
zarr_path = Path(r'D:\20230630_cells_MERFISHV2\processed\raw_zarr\cells.zarr')
n5_path = Path(r'D:\20230630_cells_MERFISH_alldeskew\processed\deskewed_bdv')

# create BDV N5
bdv_output_dir_path = n5_path
bdv_output_dir_path.mkdir(parents=True, exist_ok=True)
bdv_output_path = bdv_output_dir_path / Path(root_name+'.n5')
bdv_xml_path = bdv_output_dir_path / Path(root_name+'.xml')
# nchannels = 3
# bdv_writer = npy2bdv.BdvWriter(str(bdv_output_path),
#                                 nchannels=nchannels,
#                                 ntiles=num_x*(num_y-1)*num_z,
#                                 subsamp=((1,1,1),(4,4,4),(8,8,8),(16,16,16),(32,32,32)),
#                                 blockdim=((64,64,64),(64,64,64),(64,64,64),(64,64,64),(64,64,64)),
#                                 compression='blosc')
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# # create blank affine transformation to use for stage translation
# unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
#                         (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
#                         (0.0, 0.0, 1.0, 0.0))) # change the 4. value for z_translation (px)

# # loop over all rounds.
# for r_idx in range(num_r):
#     # create group for this round in Zarr
#     round_name = 'r'+str(r_idx).zfill(3)
#     tile_idx = 0
#     flatfield_done = False
#     for (x_idx,y_idx,z_idx) in product(range(num_x),range(1,num_y),range(num_z)):
#         for channel_id, ch_BDV_idx in zip(channels,ch_BDV_ids):
        
#             tile_name = 'x'+str(x_idx).zfill(3)+'_y'+str(y_idx).zfill(3)+'_z'+str(z_idx).zfill(3)
#             print(data_io.time_stamp(), 'Round/Tile: '+str(round_name)+'/'+str(tile_name)+'/'+channel_id)
#             current_channel = zarr.open_group(zarr_path,mode='r',path=round_name+'/'+tile_name+'/'+channel_id)
#             raw_data = da.from_zarr(current_channel['raw_data']).compute().astype(np.uint16)
#             stage_x, stage_y, stage_z = da.from_zarr(current_channel['stage_position']).compute().astype(np.float32)

#             print(data_io.time_stamp(), 'Flatfield.')
#             if not(flatfield_done) and channel_id == 'ch488':
#                 img_data_mean = np.mean(raw_data,axis=(1,2)) # might need to randomly sample for bigger data
#                 top_brightness_idx = np.argsort(img_data_mean)
#                 brightest_flatfield_data = raw_data[top_brightness_idx[-500:-1],:]
#                 del img_data_mean, top_brightness_idx
#                 gc.collect()

#                 flatfield, darkfield = generate_flatfield(brightest_flatfield_data)
#                 del brightest_flatfield_data
#                 gc.collect()
#                 flatfield_done=True

#             corrected_stack = apply_flatfield(raw_data,flatfield,darkfield)

#             print(data_io.time_stamp(), 'Deskew.')
#             deskewed = deskew(data=corrected_stack,
#                                 pixel_size=pixel_size,
#                                 scan_step=scan_step,
#                                 theta=theta)

#             # create affine transformation for stage translation
#             # swap x & y from instrument to BDV
#             print(data_io.time_stamp(), 'Write BDV N5 tile.')
#             affine_matrix = unit_matrix
#             affine_matrix[0,3] = (stage_y)/(bdv_pixel_size)  # BDV x-translation (tile axis)
#             affine_matrix[1,3] = (stage_x)/(bdv_pixel_size)  # BDV y-translation (scan axis)
#             affine_matrix[2,3] = (-1*stage_z) / (bdv_pixel_size)  # BDV z-translation (height axis)

#             bdv_writer.append_view(deskewed, 
#                                    time=r_idx, 
#                                    channel=ch_BDV_idx,
#                                    tile=tile_idx,
#                                    voxel_size_xyz=(bdv_pixel_size, bdv_pixel_size, bdv_pixel_size),
#                                    voxel_units='um',
#                                    calibration=(1,1,bdv_pixel_size/bdv_pixel_size),
#                                    m_affine=affine_matrix,
#                                    name_affine = 'tile '+str(tile_idx)+' translation')
#             bdv_writer.write_xml()

#             del deskewed
#             gc.collect()

#         tile_idx = tile_idx+1

# bdv_writer.write_xml()
    
# run BigStitcher translation registration
# Both the BigStitcher and local affine results are placed into raw data Zarr array for use in decoding.
# TO DO: load Fiji related paths from .json
print(data_io.time_stamp(),'Starting BigStitcher registration and poly-dT fusion (can take >12 hours).')
fiji_path = Path(r'C:\Fiji.app\ImageJ-win64.exe')

print(data_io.time_stamp(),'Calculate and filter initial rigid registrations.')
macro_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\rigid_registration.ijm')
run_bigstitcher(output_dir_path,fiji_path,macro_path,bdv_xml_path)

# print(data_io.time_stamp(),'Calculate interest point registrations.')
# macro_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\ip_registration.ijm')
# run_bigstitcher(output_dir_path,fiji_path,macro_path,bdv_xml_path)

print(data_io.time_stamp(),'Optimize global alignment of all tiles and rounds.')    
macro_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\align.ijm')
run_bigstitcher(output_dir_path,fiji_path,macro_path,bdv_xml_path)

# print(data_io.time_stamp(),'Calculate affine registration for each aligned tile across rounds.')    
# macro_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\affine.ijm')
# run_bigstitcher(output_dir_path,fiji_path,macro_path,bdv_xml_path)

print(data_io.time_stamp(),'Generate 4x downsampled fusion of first round poly-dT.')    
macro_path = Path(r'C:\Users\qi2lab\Documents\GitHub\OPM\reconstruction\fusion.ijm')
run_bigstitcher(output_dir_path,fiji_path,macro_path,bdv_xml_path)

print(data_io.time_stamp(),'Finished BigStitcher registration and poly-dT fusion.')

# run affine local registration across rounds.
# Both the BigStitcher and local affine results are placed into raw data Zarr array for use in decoding.
print(data_io.time_stamp(), 'Starting DEEDS local affine registration (can take >12 hours).')
perform_local_registration(bdv_output_path,
                            bdv_xml_path,
                            zarr_path,
                            num_r,
                            num_x,
                            num_y,
                            num_z,
                            bdv_pixel_size,
                            compressor)
print(data_io.time_stamp(), 'Finished local affine translations.')

# exit
print(data_io.time_stamp(), 'Finished processing dataset.')
sys.exit()