import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from datetime import datetime

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR_IHC = parent_path / "BCI_dataset/IHC/train"
TEST_DIR_IHC = parent_path / "BCI_dataset/IHC/val"
TRAIN_DIR_HE = parent_path / "BCI_dataset/HE/train"
TEST_DIR_HE = parent_path / "BCI_dataset/HE/val"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = True
SAVE_MODEL = True
DISCRIMINATOR_FEATURES = [64, 128, 256, 512]    # Not implemented yet

# If LOAD_MODEL = True, must define manually current_time variable name to match an existing file with model learned parameters,
if not LOAD_MODEL:
    current_time = datetime.now().strftime("%Y%m%d") 
else:
    current_time = "20240621"

CHECKPOINT_GEN_HE = f"genHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_GEN_IHC = f"genIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_HE = f"discHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_IHC = f"discIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"


transforms = A.Compose([
        A.Resize(width=64, height=64),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

if __name__ == "__main__":
    print(CHECKPOINT_GEN_HE.split('_')[0])
