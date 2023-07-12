// read XML path
args = getArgument();
args = split(args, " ");

xmlpath = args[0];
n5datapath = args[1];
n5xmlpath = args[2];

run("Flip Axes", "select="+xmlpath+" flip_z");
run("Calculate pairwise shifts ...", "select="+xmlpath+" process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] show_expert_grouping_options how_to_treat_timepoints=[treat individually] how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat individually] how_to_treat_tiles=compare downsample_in_x=16 downsample_in_y=16 downsample_in_z=16");
run("Calculate pairwise shifts ...", "select="+xmlpath+" process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] show_expert_grouping_options how_to_treat_timepoints=compare how_to_treat_channels=[treat individually] how_to_treat_illuminations=[treat individually] how_to_treat_angles=[treat individually] how_to_treat_tiles=[treat individually] downsample_in_x=16 downsample_in_y=16 downsample_in_z=16");
run("Filter pairwise shifts ...", "select="+xmlpath+" filter_by_link_quality min_r=.6 max_r=1 max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");
run("Filter pairwise shifts ...", "select"+xmlpath+" min_r=0 max_r=1 filter_by_shift_in_each_dimension max_shift_in_x=150 max_shift_in_y=150 max_shift_in_z=150 max_displacement=0");
eval("script","System.exit(0);");