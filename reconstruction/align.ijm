// read XML path
args = getArgument();
args = split(args, " ");

xmlpath = args[0];
n5datapath = args[1];
n5xmlpath = args[2];

run("Optimize globally and apply shifts ...", "select="+xmlpath+" process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.5 absolute=3.5 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles] show_expert_grouping_options how_to_treat_timepoints=compare how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat individually] how_to_treat_tiles=compare fix_group_0-0");
eval("script","System.exit(0);");