from pathlib import Path
import subprocess

# Get the absolute path to the directory where this script is located
dir_path = Path(__file__).resolve().parent

# Navigate to parent directories as needed
repo_path = dir_path.parent
parent_path = repo_path.parent

# Construct the full paths to the necessary files and directories

path_to_compute_stats = parent_path / "Hover_net/hover_net2/compute_stats.py"
true_dir = parent_path / "Datasets/Pannuke/data/test/mat"
pred_dir = parent_path / "Results/hover_results/mat"




# Build the command array, ensuring all path objects are converted to strings
command = [
    'python', path_to_compute_stats,
    '--mode=instance',
    '--pred_dir=' + str(pred_dir),
    '--true_dir=' + str(true_dir)
]

# Execute the command
subprocess.run(command)
