import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from datetime import datetime

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

""" All the flags used for control and to choose before training/testing the model

    MODEL TRAINING:
            DEVICE                  -- Run on GPU and if not possible, to CPU
            BATCH_SIZE              -- Nº imgs of each stain to be processed at the same time, each epoch
            LEARNING_RATE           -- Optimizer lr to apply
            LAMBDA_IDENTITY         -- Weight factor to apply to  generator identity loss (between 0 and 1)
            LAMBDA_CYCLE            -- Weight factor to apply to  generator cycle loss (10 in cycleGAN paper)
            NUM_EPOCHS              -- Nº of times to pass the entire dataset through the network
            NUM_WORKERS             --
            
    GENERATOR/DISCRIMINATOR:
            D_FEATURES (list)       -- nº of channels/layer of the discriminator (from last to first layer)
            IN_CH                   -- nº of input channels to pass through the network (3 for RGB / 1 for black n white)
            N_RES_BLOCKS            -- nº of residual blocks of the generator architecture between downsampling and upsampling

    DATASET:
            TRAIN_DIR_IHC           -- Directory for IHC stain training data
            TEST_DIR_IHC            -- Directory for IHC stain test data
            TRAIN_DIR_HE            -- Directory for HE stain training data
            TEST_DIR_HE             -- Directory for HE stain test data
            SUBSET_PERCENTAGE       -- % of cropped dataset to use for train/val/test (total size of 20k imgs)
            SHUFFLE_DATA (bool)     -- Set to True/False to shuffle position index of data on the dataset
            transforms              -- Transformations applied to data before passing it through the model

    MODEL TRACKING:
            LOAD_MODEL (bool)       -- Set to True/False to load an already trained model, for further training or testing
            SAVE_MODEL (bool)       -- Set to True/False to save model after each epoch during training
            CHECKPOINT_GEN_HE       -- Checkpoint filename for HE Generator network with trained parameters
            CHECKPOINT_GEN_IHC      -- Checkpoint filename for IHC Generator network with trained parameters
            CHECKPOINT_DISC_HE      -- Checkpoint filename for HE Discriminator network with trained parameters
            CHECKPOINT_DISC_IHC     -- Checkpoint filename for IHC Discriminator network with trained parameters
            current_time            -- If LOAD_MODEL = True, manually define this name to match existing checkpoint filename

"""

TRAIN_DIR_IHC = parent_path / "BCI_dataset/IHC/train"
TEST_DIR_IHC = parent_path / "BCI_dataset/IHC/test"
TRAIN_DIR_HE = parent_path / "BCI_dataset/HE/train"
TEST_DIR_HE = parent_path / "BCI_dataset/HE/test"

ENDONUKE_DIR_IHC = parent_path / "BCI_dataset/IHC/test"
ENDONUKE_DIR_HE = parent_path / "BCI_dataset/HE/train"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10
NUM_EPOCHS = 150
NUM_WORKERS = 4
D_FEATURES = [64, 128, 256, 512]
IN_CH = 3
N_RES_BLOCKS = 6
SUBSET_PERCENTAGE = 100
SHUFFLE_DATA = False
EARLY_STOP = 15
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

LOAD_MODEL = False
SAVE_MODEL = True
SUFFIX1 = 200
SUFFIX2 = "20240628"

if not LOAD_MODEL:    # If LOAD_MODEL = True, must define manually current_time variable name to match an existing file with model learned parameters
    current_time = datetime.now().strftime("%Y%m%d") 
else:
    current_time = "20240628"

CHECKPOINT_GEN_HE = parent_path / f"training-models/genHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_GEN_IHC = parent_path / f"training-models/genIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_HE = parent_path / f"training-models/discHE_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"
CHECKPOINT_DISC_IHC = parent_path / f"training-models/discIHC_{NUM_EPOCHS}_epochs_{current_time}.pth.tar"

PRETRAINED_GEN_HE = parent_path / f"pretrained-models/genHE_{SUFFIX1}_epochs_{SUFFIX2}.pth.tar"
PRETRAINED_GEN_IHC = parent_path / f"pretrained-models/genIHC_{SUFFIX1}_epochs_{SUFFIX2}.pth.tar"
PRETRAINED_DISC_HE = parent_path / f"pretrained-models/discHE_{SUFFIX1}_epochs_{SUFFIX2}.pth.tar"
PRETRAINED_DISC_IHC = parent_path / f"pretrained-models/discIHC_{SUFFIX1}_epochs_{SUFFIX2}.pth.tar"


if __name__ == "__main__":
    pass
