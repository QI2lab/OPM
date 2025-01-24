'''
Napari interface to process OPM timelapse data

TO DO: - Change to OME-Zarr output
       - Add OME-tiff resave option
       - Add ability to load old data
       - Add option for number of iterations & TV setting for clij2-fft

Last update: 12/2022; Update to use clij2-fft, remove flatfield (due to Powell lens), remove dexp dependence
D. Shepherd - 12/2021
'''

from magicclass import magicclass, MagicTemplate
from magicgui import magicgui
from magicgui.tqdm import trange
from pathlib import Path
import numpy as np
from src.utils.image_post_processing import deskew
from napari.qt.threading import thread_worker
import zarr
import dask.array as da
from src.utils.data_io import read_metadata, return_opm_psf, time_stamp
from skimage.measure import block_reduce
from itertools import compress
import gc
from tifffile import TiffWriter
from numcodecs import Blosc, blosc

try:
    import cupy as cp
    CP_AVAILABLE = True
except:
    CP_AVAILABLE = False

if CP_AVAILABLE:
    try:
        from cucim.skimage.exposure import match_histograms
        CUCIM_AVAILABLE = True
    except:
        from skimage.exposure import match_histograms
        CUCIM_AVAILABLE = False
else:
    from skimage.exposure import match_histograms
    CUCIM_AVAILABLE = False

# OPM control UI element            
@magicclass(labels=False)
class OPMMirrorReconstruction(MagicTemplate):

    def __init__(self):
        self.decon = False
        self.match_histograms = False
        self.debug = False
        self.channel_idxs=[0,1,2,3,4]
        self.active_channels=[False,False,False,False,False]
        self._create_batch_processing_worker()
        self._create_tiff_conversion_worker()
        self.z_downsample = 1
        
        self.background = 100
        self.ADU_to_e = .24
        self.QE = [.75,.9,.95,.85,.65]

    # set viewer
    def _set_viewer(self,viewer):
        self.viewer = viewer

    @thread_worker
    def _process_data(self):
        
        # create parameter array from scan parameters saved by acquisition code
        df_metadata = read_metadata(self.data_path / Path('scan_metadata.csv'))
        root_name = df_metadata['root_name']
        scan_type = df_metadata['scan_type']
        theta = df_metadata['theta']
        exposure_ms = df_metadata['exposure_ms']
        scan_step = df_metadata['scan_step']
        pixel_size = df_metadata['pixel_size']
        num_t = df_metadata['num_t']
        time_delay_s = df_metadata['time_delay']
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

        self.active_channels = active_channels
        self.channels_in_data = channels_in_data

        # calculate pixel sizes of deskewed image in microns
        deskewed_x_pixel = pixel_size
        deskewed_y_pixel = pixel_size
        deskewed_z_pixel = pixel_size
        if self.debug:
            print('Deskewed pixel sizes before downsampling (um). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

        # deskew parameters
        deskew_parameters = np.empty([3])
        deskew_parameters[0] = theta            # (degrees)
        deskew_parameters[1] = scan_step*100    # (nm)
        deskew_parameters[2] = pixel_size*100   # (nm)

        # amount of down sampling in z
        z_down_sample = self.z_downsample

        # load dataset
        dataset_zarr = zarr.open(self.data_path / Path(root_name+'.zarr'),mode='r')

        # create output directory
        if self.decon == 0:
            output_dir_path = self.data_path / 'deskew_output'
        elif self.decon == 1:
            output_dir_path = self.data_path / 'deskew_decon_output'
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # create name for zarr directory
        zarr_output_path = output_dir_path / Path('OPM_processed.zarr')

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
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        blosc.use_threads=True
        blosc.set_nthreads(20)
        opm_data = zarr.open(str(zarr_output_path), 
                             mode="w", 
                             shape=(num_t, num_ch, nz//z_down_sample, ny, nx), 
                             chunks=(1, 1, int(nz//z_down_sample), int(ny), int(nx)//4), 
                             dimension_separator='/',
                             compressor = compressor,
                             dtype=np.uint16)
        
        if self.decon:
            from src.utils.image_post_processing import lr_deconvolution

            skewed_psf = []

            for ch_idx in np.flatnonzero(active_channels):
                skewed_psf.append(return_opm_psf(ch_idx))

        if self.match_histograms:
            reference_images = np.zeros((num_ch, nz, ny, nx), dtype=np.uint16)
        
        for t_idx in trange(num_t,desc='t',position=0):
            for ch_idx in trange(n_active_channels,desc='c',position=1, leave=False):

                raw_data = np.array(dataset_zarr[t_idx,ch_idx,0:num_images,:]).astype(np.float32)
                raw_data = raw_data - self.background
                raw_data[raw_data<0.]=0
                raw_data = ((raw_data * self.ADU_to_e) / self.QE[ch_idx]) .astype(np.uint16)

                # pull data stack into memory
                if self.debug:
                    print('Process timepoint '+str(t_idx)+'; channel '+str(ch_idx) +'.')

                # run deconvolution on deskewed image
                if self.decon:
                    if self.debug:
                        print('Deconvolve.')
                    psf = np.asarray(skewed_psf[ch_idx],dtype=np.float32).copy()
                    stride = int(np.round(scan_step/.4,0))
                    psf_strided = psf[::stride,:,:]
                    psf_strided = psf_strided / np.sum(psf_strided,axis=(0,1,2))
                    decon = lr_deconvolution(raw_data,psf_strided,iterations=100)
                    del psf, psf_strided
                else:
                    decon = raw_data
                del raw_data

                # deskew
                if self.debug:
                    print('Deskew.')
                deskewed = deskew(decon,*deskew_parameters)
                del decon

                # downsample in z due to oversampling when going from OPM to coverslip geometry
                if z_down_sample==1:
                    deskewed_downsample = deskewed
                else:
                    if self.debug:
                        print('Downsample.')
                    deskewed_downsample = block_reduce(deskewed, block_size=(z_down_sample,1,1), func=np.mean)
                del deskewed

                if self.match_histograms:
                    if self.debug:
                        print('Match histogram.')

                    if t_idx == 0:
                        reference_images[ch_idx,:] = deskewed_downsample
                        deskewed_matched = deskewed_downsample
                    else:
                        if CUCIM_AVAILABLE:
                            reference_image_cp = cp.asarray(reference_images[ch_idx,:],dtype=cp.uint16)
                            deskewed_downsample_cp = cp.asarray(deskewed_downsample,dtype=cp.uint16)
                            deskewed_matched = cp.asnumpy(match_histograms(deskewed_downsample_cp,reference_image_cp)).astype(np.uint16)
                            del reference_image_cp, deskewed_downsample_cp
                            gc.collect()
                            cp.clear_memo()
                            cp._default_memory_pool.free_all_blocks()
                        else:
                            deskewed_matched = match_histograms(reference_images[ch_idx,:],deskewed_downsample)
                        
                else:
                    deskewed_matched = deskewed_downsample
                del deskewed_downsample
                
                opm_data[t_idx,ch_idx,:] = deskewed_matched.astype(np.uint16)

                opm_data.attrs['dx_um'] = deskewed_x_pixel
                opm_data.attrs['dy_um'] = deskewed_y_pixel
                opm_data.attrs['dz_um'] = deskewed_z_pixel * z_down_sample
                opm_data.attrs['volume_time_ms'] = np.round(num_images * exposure_ms,3)
                opm_data.attrs['volume_interval_s'] = time_delay_s
                opm_data.attrs['405_state'] = chan_405_active
                opm_data.attrs['488_state'] = chan_488_active
                opm_data.attrs['561_state'] = chan_561_active
                opm_data.attrs['635_state'] = chan_635_active
                opm_data.attrs['730_state'] = chan_730_active
            
                # free up memory
                del deskewed_matched
                gc.collect()

                if self.debug:
                    print('Write data into Zarr container')

        # exit
        self.dataset_zarr = zarr_output_path
        self.scale = [1,deskewed_z_pixel* z_down_sample,deskewed_y_pixel,deskewed_x_pixel]

    @thread_worker
    def _batch_process_data(self):

        for path in Path(self.root_path).iterdir():
            for sub_path in Path(path).iterdir():
                test_metadata_path = sub_path / Path('scan_metadata.csv')

                if test_metadata_path.exists():
                    self.data_path = sub_path
                else:
                    self.data_path = path
                    
                print(time_stamp(),'Processing path: '+str(self.data_path.name))
                
                # create parameter array from scan parameters saved by acquisition code
                df_metadata = read_metadata(self.data_path / Path('scan_metadata.csv'))
                root_name = df_metadata['root_name']
                scan_type = df_metadata['scan_type']
                theta = df_metadata['theta']
                scan_step = df_metadata['scan_step']
                pixel_size = df_metadata['pixel_size']
                exposure_ms = df_metadata['exposure_ms']
                num_t = df_metadata['num_t']
                time_delay_s = df_metadata['time_delay']
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

                self.active_channels = active_channels
                self.channels_in_data = channels_in_data

                # calculate pixel sizes of deskewed image in microns
                deskewed_x_pixel = pixel_size
                deskewed_y_pixel = pixel_size
                deskewed_z_pixel = pixel_size
                if self.debug:
                    print('Deskewed pixel sizes before downsampling (um). x='+str(deskewed_x_pixel)+', y='+str(deskewed_y_pixel)+', z='+str(deskewed_z_pixel)+'.')

                # deskew parameters
                deskew_parameters = np.empty([3])
                deskew_parameters[0] = theta            # (degrees)
                deskew_parameters[1] = scan_step*100    # (nm)
                deskew_parameters[2] = pixel_size*100   # (nm)

                # amount of down sampling in z
                z_down_sample = self.z_downsample

                # load dataset
                dataset_zarr = zarr.open(self.data_path / Path(root_name+'.zarr'),mode='r')

                # create output directory
                if self.decon == 0:
                    output_dir_path = self.data_path / 'deskew_output'
                elif self.decon == 1:
                    output_dir_path = self.data_path / 'deskew_decon_output'
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # create name for zarr directory
                zarr_output_path = output_dir_path / Path('OPM_processed.zarr')
                
                if zarr_output_path.exists():
                    continue

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
                compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
                blosc.use_threads=True
                blosc.set_nthreads(20)
                if z_down_sample==1:
                    nz_on_disk = int(nz)
                else:
                    nz_on_disk = int(nz / int(z_down_sample))
                
                opm_data = zarr.open(str(zarr_output_path), 
                                    mode="w", 
                                    shape=(num_t, num_ch, nz_on_disk, ny, nx), 
                                    chunks=(1, 1, 1, int(ny), int(nx)), 
                                    dimension_separator='/',
                                    compressor = compressor,
                                    dtype=np.uint16)
                
                # if decon is requested, try to import microvolution wrapper or clij2-fft library
                if self.decon:
                    #from src.utils.opm_psf import generate_skewed_psf
                    from src.utils.image_post_processing import lr_deconvolution

                    skewed_psf = []

                    for ch_idx in np.flatnonzero(active_channels):
                        skewed_psf.append(return_opm_psf(ch_idx))

                if self.match_histograms:
                    reference_images = np.zeros((num_ch, nz, ny, nx), dtype=np.uint16)
                
                for t_idx in trange(num_t,desc='t',position=0):
                    for ch_idx in trange(n_active_channels,desc='c',position=1, leave=False):

                        raw_data = np.asarray(dataset_zarr[t_idx,ch_idx,0:num_images,:]).astype(np.float32)
                        raw_data = raw_data - self.background
                        raw_data[raw_data<0.] = 0
                        raw_data = ((raw_data * self.ADU_to_e)/self.QE[ch_idx]).astype(np.uint16)

                        # pull data stack into memory
                        if self.debug:
                            print('Process timepoint '+str(t_idx)+'; channel '+str(ch_idx) +'.')
                        #raw_data = return_data_from_zarr_to_numpy(dataset_zarr, t_idx, ch_idx, num_images, y_pixels,x_pixels)

                        # run deconvolution on deskewed image
                        if self.decon:
                            if self.debug:
                                print('Deconvolve.')
                            psf = np.asarray(skewed_psf[ch_idx],dtype=np.float32)
                            stride = int(np.round(scan_step/.4,0))
                            psf_strided = psf[::stride,:,:]
                            psf_strided = psf_strided / np.sum(psf_strided,axis=(0,1,2))
                            decon = lr_deconvolution(raw_data,psf_strided,iterations=100)
                        else:
                            decon = raw_data
                            pass
                        del raw_data

                        # deskew
                        if self.debug:
                            print('Deskew.')
                        #deskewed = deskew(np.flipud(decon),*deskew_parameters)
                        deskewed = deskew(decon,*deskew_parameters)
                        del decon

                        # downsample in z due to oversampling when going from OPM to coverslip geometry
                        if z_down_sample==1:
                            deskewed_downsample = deskewed
                        else:
                            if self.debug:
                                print('Downsample.')
                            deskewed_downsample = block_reduce(deskewed, block_size=(z_down_sample,1,1), func=np.mean)
                        del deskewed

                        if self.match_histograms:
                            if self.debug:
                                print('Match histogram.')

                            if t_idx == 0:
                                reference_images[ch_idx,:] = deskewed_downsample
                                deskewed_matched = deskewed_downsample
                            else:
                                if CUCIM_AVAILABLE:
                                    reference_image_cp = cp.asarray(reference_images[ch_idx,:],dtype=cp.uint16)
                                    deskewed_downsample_cp = cp.asarray(deskewed_downsample,dtype=cp.uint16)
                                    deskewed_matched = cp.asnumpy(match_histograms(deskewed_downsample_cp,reference_image_cp)).astype(np.uint16)
                                    del reference_image_cp, deskewed_downsample_cp
                                    gc.collect()
                                    cp.clear_memo()
                                    cp._default_memory_pool.free_all_blocks()
                                else:
                                    deskewed_matched = match_histograms(reference_images[ch_idx,:],deskewed_downsample)
                                
                        else:
                            deskewed_matched = deskewed_downsample
                        del deskewed_downsample

                        opm_data[t_idx,ch_idx,:] = deskewed_matched.astype(np.uint16)
                    
                        # free up memory
                        del deskewed_matched
                        gc.collect()

                        if self.debug:
                            print('Write data into Zarr container')

                # exit
                self.scale = [1,deskewed_z_pixel* z_down_sample,deskewed_y_pixel,deskewed_x_pixel]
               # opm_data[:] = opm_data_np[:]

                opm_data.attrs['dx_um'] = np.round(deskewed_x_pixel,3)
                opm_data.attrs['dy_um'] = np.round(deskewed_y_pixel,3)
                opm_data.attrs['dz_um'] = np.round(deskewed_z_pixel * z_down_sample,3)
                opm_data.attrs['exp_ms'] = np.round(exposure_ms,2)
                opm_data.attrs['volume_time_ms'] = np.round(num_images * exposure_ms,3)
                opm_data.attrs['volume_interval_s'] = np.round(time_delay_s,2)
                opm_data.attrs['405_state'] = chan_405_active
                opm_data.attrs['488_state'] = chan_488_active
                opm_data.attrs['561_state'] = chan_561_active
                opm_data.attrs['635_state'] = chan_635_active
                opm_data.attrs['730_state'] = chan_730_active

                del opm_data   
                gc.collect() 
        print(time_stamp(),'Finished.') 

    @thread_worker
    def _tiff_convert_data(self):
         for path in Path(self.tiff_path).iterdir():
            if path.is_dir():
                print(time_stamp(),'Processing path: '+str(path.name))
                self.data_path = path

                if self.decon == 0:
                    input_dir_path = self.data_path / 'deskew_output'
                elif self.decon == 1:
                    input_dir_path = self.data_path / 'deskew_decon_output'
                zarr_input_path = input_dir_path / Path('OPM_processed.zarr')
                data_zarr = zarr.open(zarr_input_path,mode='r')

                tiff_output_path = input_dir_path / Path('OPM_processed_tiff')
                tiff_output_path.mkdir(parents=True, exist_ok=True)

                max_tiff_output_path = tiff_output_path / Path('max_projections')
                max_tiff_output_path.mkdir(parents=True, exist_ok=True)
                
                for t_idx in trange(data_zarr.shape[0],desc='t'):
                    # filename = 'opm_data_'+str(path.name)+'_t'+str(t_idx).zfill(4)+'.ome.tiff'
                    # filename_path = tiff_output_path / Path(filename)
                    # if not(filename_path.exists()):
                    #     with TiffWriter(filename_path, bigtiff=True) as tif:
                    #         metadata={'axes': 'ZCYX',
                    #                 'SignificantBits': 16,
                    #                 #'TimeIncrement': data_zarr.attrs['volume_time_ms'],
                    #                 #'TimeIncrementUnit': 'ms',
                    #                 'PhysicalSizeX': data_zarr.attrs['dx_um'],
                    #                 'PhysicalSizeXUnit': 'µm',
                    #                 'PhysicalSizeY': data_zarr.attrs['dy_um'],
                    #                 'PhysicalSizeYUnit': 'µm',
                    #                 'PhysicalSizeZ': data_zarr.attrs['dz_um'],
                    #                 'PhysicalSizeZUnit': 'µm',
                    #                 }
                    #         options = dict(compression='zlib',
                    #                         compressionargs={'level': 8},
                    #                         predictor=True,
                    #                         photometric='minisblack',
                    #                         resolutionunit='CENTIMETER',
                    #                         tile=(64,64),
                    #                         )
                    #         tif.write(np.swapaxes(np.array(data_zarr[t_idx,:]),0,1),
                    #                     resolution=(1e4 / data_zarr.attrs['dy_um'],
                    #                                 1e4 / data_zarr.attrs['dx_um']),
                    #                     **options,
                    #                     metadata=metadata)
                        
                    max_filename = 'max_z_opm_data_'+str(path.name)+'_t'+str(t_idx).zfill(4)+'.ome.tiff'
                    max_filename_path = max_tiff_output_path / Path(max_filename)
                    if not(max_filename_path.exists()):
                        with TiffWriter(max_filename_path, bigtiff=True) as tif:
                            metadata={'axes': 'CYX',
                                    'SignificantBits': 16,
                                    #'TimeIncrement': data_zarr.attrs['volume_time_ms'],
                                    #'TimeIncrementUnit': 'ms',
                                    'PhysicalSizeX': data_zarr.attrs['dx_um'],
                                    'PhysicalSizeXUnit': 'µm',
                                    'PhysicalSizeY': data_zarr.attrs['dy_um'],
                                    'PhysicalSizeYUnit': 'µm'
                                    }
                            options = dict(compression='zlib',
                                            compressionargs={'level': 8},
                                            predictor=True,
                                            photometric='minisblack',
                                            resolutionunit='CENTIMETER',
                                            tile=(64,64),
                                            )
                            tif.write(np.max(np.swapaxes(np.array(data_zarr[t_idx,:]),0,1),0),
                                        resolution=(1e4 / data_zarr.attrs['dy_um'],
                                                    1e4 / data_zarr.attrs['dx_um']),
                                        **options,
                                        metadata=metadata)  
                            
    def _create_processing_worker(self):
        worker_processing = self._process_data()
        self._set_worker_processing(worker_processing)

    # set 3D timelapse acquistion thread worker
    def _set_worker_processing(self,worker_processing):
        self.worker_processing = worker_processing

    def _create_batch_processing_worker(self):
        batch_worker_processing = self._batch_process_data()
        self._set_batch_worker_processing(batch_worker_processing)

    # set 3D timelapse acquistion thread worker
    def _set_batch_worker_processing(self,batch_worker_processing):
        self.batch_worker_processing = batch_worker_processing

    def _create_tiff_conversion_worker(self):
        tiff_worker_conversion = self._tiff_convert_data()
        self._set_tiff_worker_conversion(tiff_worker_conversion)

    # set 3D timelapse acquistion thread worker
    def _set_tiff_worker_conversion(self,tiff_worker_conversion):
        self.tiff_worker_conversion = tiff_worker_conversion

    # update viewer
    def _update_viewer(self,display_data):

        # clean up viewer
        self.viewer.layers.clear()

        # channel names and colormaps to match control software
        channel_names = ['405nm','488nm','561nm','635nm','730nm']
        colormaps = ['bop purple','bop blue','bop orange','red','grey']

        active_channel_names=[]
        active_colormaps=[]

        dataset = da.from_zarr(zarr.open(self.dataset_zarr,mode='r'))

        # iterate through active channels and populate viewer
        for channel in self.channels_in_data:
            active_channel_names.append(channel_names[channel])
            active_colormaps.append(colormaps[channel])
        self.viewer.add_image(
            dataset, 
            channel_axis=1, 
            name=active_channel_names, 
            scale = self.scale,
            blending='additive', 
            colormap=active_colormaps)
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'

    # set deconvoluton option
    @magicgui(
        auto_call=True,
        use_decon = {"widget_type": "CheckBox", "label": "Deconvolution"},
        layout="horizontal"
    )
    def set_deconvolution_option(self,use_decon = False):
        self.decon = use_decon

    # set histogram matching option
    @magicgui(
        auto_call=True,
        use_match_histograms = {"widget_type": "CheckBox", "label": "Match histograms"},
        layout="horizontal"
    )
    def set_histogram_option(self,use_match_histograms = False):
        self.match_histograms = use_match_histograms

     # set z downsample option
    @magicgui(
        auto_call=True,
        z_downsample = {"widget_type": "SpinBox", "label": "Z downsample ratio"},
        layout="horizontal"
    )
    def set_z_downsample_option(self,z_downsample = 1):
        self.z_downsample = z_downsample
        
    # set path to dataset for procesing
    @magicgui(
        auto_call=False,
        data_path={"widget_type": "FileEdit","mode": "d", "label": 'Folder to process (one acq):'},
        layout='vertical', 
        call_button="Set"
    )
    def set_data_processing(self, data_path='d:/'):
        self.data_path = data_path
        df_metadata = read_metadata(self.data_path / Path('scan_metadata.csv'))
        self.time_points = df_metadata['num_t']

    # control single folder data processing
    @magicgui(
        auto_call=True,
        start_processing={"widget_type": "PushButton", "label": 'Run reconstruction'},
        layout='horizontal'
    )
    def run_processing(self,start_processing):
        if not(self.data_path is None):
            self.worker_processing.start()
            self.worker_processing.returned.connect(self._create_processing_worker)
            self.worker_processing.returned.connect(self._update_viewer)

    # set path for batch processing
    @magicgui(
        auto_call=False,
        root_path={"widget_type": "FileEdit","mode": "d", "label": 'Folder to batch process:'},
        layout='vertical', 
        call_button="Set"
    )
    def set_batch_processing(self, root_path='d:/'):
        self.root_path = root_path

    # control multiple folder data processing
    @magicgui(
        auto_call=True,
        start_batch_processing={"widget_type": "PushButton", "label": 'Run batch reconstruction'},
        layout='horizontal'
    )
    def run_batch_processing(self,start_batch_processing):
        self.batch_worker_processing.start()
        self.batch_worker_processing.returned.connect(self._create_batch_processing_worker)
        self.batch_worker_processing.returned.connect(self._update_viewer)

    # set path for tiff processing
    @magicgui(
        auto_call=False,
        tiff_path={"widget_type": "FileEdit","mode": "d", "label": 'Folder to batch convert zarr to tiff:'},
        layout='vertical', 
        call_button="Set"
    )
    def set_tiff_conversion(self, tiff_path='d:/'):
        self.tiff_path = tiff_path

    # convert deskewed zarr to tiff
    @magicgui(
        auto_call=True,
        start_tiff_conversion={"widget_type": "PushButton", "label": 'Run tiff conversion'},
        layout='horizontal'
    )
    def run_tiff_conversion(self,start_tiff_conversion):
        self.tiff_worker_conversion.start()
        self.tiff_worker_conversion.returned.connect(self._create_tiff_conversion_worker)
        self.tiff_worker_conversion.returned.connect(self._update_viewer)