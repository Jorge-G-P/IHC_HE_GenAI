import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

CHECKPOINT_GEN_HE = parent_path / "pretrained-models/genHE_epochs.pth.tar"
print(CHECKPOINT_GEN_HE)