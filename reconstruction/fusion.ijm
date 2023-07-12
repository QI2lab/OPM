// read XML path
args = getArgument();
args = split(args, " ");

xmlpath = args[0];
n5datapath = args[1];
n5xmlpath = args[2];

run("Fuse dataset ...", "select="+xmlpath+" process_angle=[All angles] process_channel=[Single channel (Select from List)] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[Single Timepoint (Select from List)] processing_channel=[channel 0] processing_timepoint=[Timepoint 0] bounding_box=[Currently Selected Views] downsampling=4 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend produce=[Each timepoint & channel] fused_image=[ZARR/N5/HDF5 export using N5-API] define_input=[Auto-load from input data (values shown below)] export=N5 create n5_dataset_path="+n5datapath+" xml_output_file="+n5xmlpath+" viewid_timepointid=0 viewid_setupid=0");
eval("script","System.exit(0);");