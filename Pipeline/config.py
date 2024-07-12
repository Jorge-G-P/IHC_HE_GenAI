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

PREPROCESS_ENDONUKE = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.5
LAMBDA_CYCLE = 10
NUM_EPOCHS = 150
NUM_WORKERS = 4
D_FEATURES = [64, 128, 256, 512]
IN_CH = 3
N_RES_BLOCKS = 9
TRAIN_DIR_IHC = parent_path / "BCI_dataset/IHC/train"
TEST_DIR_IHC = parent_path / "BCI_dataset/IHC/test"
TRAIN_DIR_HE = parent_path / "BCI_dataset/HE/train"
TEST_DIR_HE = parent_path / "BCI_dataset/HE/test"
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

test_transforms = A.Compose([
                A.Resize(width=256, height=256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
        )

LOAD_MODEL = True
SAVE_MODEL = True

if not LOAD_MODEL:    # If LOAD_MODEL = True, must define manually current_time variable name to match an existing file with model learned parameters
    current_time = datetime.now().strftime("%Y%m%d") 
else:
    current_time = ""

CHECKPOINT_GEN_HE = repo_path / f"genHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_GEN_IHC = repo_path / f"genIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_HE = repo_path / f"discHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_IHC = repo_path / f"discIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"


# if __name__ == "__main__":
#     pass
