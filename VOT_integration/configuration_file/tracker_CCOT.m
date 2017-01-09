% Copy this template configuration file to your VOT workspace.
% Enter the full path to the CCOT repository root folder.

CCOT_repo_path = ########

tracker_label = 'CCOT';
tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''CCOT'', ''VOT2016_settings'', true)', {[CCOT_repo_path '/VOT_integration/benchmark_wrapper']});
tracker_interpreter = 'matlab';
tracker_trax = false;