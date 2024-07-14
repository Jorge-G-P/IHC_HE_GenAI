import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from datetime import datetime

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

path_to_endonuke_data_folder = parent_path / "IHC_HE_GenAI/Datasets/Endonuke/data"
path_to_Hover_net_run_infer = parent_path / "IHC_HE_GenAI/Hover_net/hover_net2/run_infer.py"
path_to_Hover_net_weights = parent_path / "IHC_HE_GenAI/pretrained_models/hovernet_fast_pannuke_type_tf2pytorch.tar"
path_to_genHE_weights = parent_path / "IHC_HE_GenAI/pretrained_models/generator_HE.tar"
results_gan_folder = parent_path/ "IHC_HE_GenAI/Results"
results_hover_folder = parent_path/ "IHC_HE_GenAI/Results"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IN_CH = 3
N_RES_BLOCKS = 9
PREPROCESS_ENDONUKE = True

SUBSET_PERCENTAGE = 15
SHUFFLE_DATA = False
EARLY_STOP = 35
FID_FREQUENCY = 5
FID_BATCH_SIZE = 32

transforms = A.Compose([
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
        )
