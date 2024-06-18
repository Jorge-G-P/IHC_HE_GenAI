import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

#TRAIN_DIR_IHC = "C:\\Users\\jorge\\Escritorio\\UPC\\12final_project\\BCI_dataset\IHC\\train"
#TRAIN_DIR_HE = "C:\\Users\\jorge\\Escritorio\\UPC\\12final_project\\BCI_dataset\HE\\train"
#TEST_DIR_IHC = "C:\\Users\\jorge\\Escritorio\\UPC\\12final_project\\BCI_dataset\IHC\\test"
#TEST_DIR_HE = "C:\\Users\\jorge\\Escritorio\\UPC\\12final_project\\BCI_dataset\HE\\test"

TRAIN_DIR_IHC = parent_path / "BCI_dataset/IHC/train"
TEST_DIR_IHC = parent_path / "BCI_dataset/IHC/val"
TRAIN_DIR_HE = parent_path / "BCI_dataset/HE/train"
TEST_DIR_HE = parent_path / "BCI_dataset/HE/val"

#TRAIN_DIR_HE = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/HE/train'
#VAL_DIR_HE = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/HE/test'        # Use for testing not validation
#TRAIN_DIR_IHC = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/IHC/train'
#VAL_DIR_IHC = r'/home/jotapv98/coding/MyProjects/JOAO_HE_IHC/BCI_dataset/IHC/test'      # Use for testing not validation
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
DISCRIMINATOR_FEATURES = [64, 128, 256, 512]    # Not implemented yet
CHECKPOINT_GEN_HE = "genh.pth.tar"
CHECKPOINT_GEN_IHC = "genz.pth.tar"
CHECKPOINT_CRITIC_HE = "critich.pth.tar"
CHECKPOINT_CRITIC_IHC = "criticz.pth.tar"

transforms = A.Compose([
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
