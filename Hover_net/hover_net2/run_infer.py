from pathlib import Path
import subprocess

# Get the absolute path to the directory where this script is located
dir_path = Path(__file__).resolve().parent

# Navigate to parent directories as needed
repo_path = dir_path.parent
parent_path = repo_path.parent

# Construct the full paths to the necessary files and directories
path_to_Hover_net_run_infer = parent_path / "Hover_net/hover_net2/infer.py"
#path_to_Hover_net_weights = parent_path / "Hover_net/hover_net2/checkpoint/00/net_epoch=4.tar"
path_to_Hover_net_weights = parent_path / "pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar"
hover_results_folder = parent_path / "Results/hover_results/"
directory_test = parent_path / "Datasets/Pannuke/data/test/images_png/"

# Build the command array, ensuring all path objects are converted to strings
command = [
    'python', path_to_Hover_net_run_infer,
    '--model_path=' + str(path_to_Hover_net_weights),
    '--model_mode=fast',
    'tile',
    '--input_dir=' + str(directory_test),
    '--output_dir=' + str(hover_results_folder),
    '--draw_dot'
]

# Execute the command
subprocess.run(command)