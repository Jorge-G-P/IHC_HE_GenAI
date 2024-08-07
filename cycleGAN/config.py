import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from datetime import datetime

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
repo_path = dir_path.parent
parent_path = repo_path.parent

bci_dataset_ihc_train = repo_path / "Datasets/BCI/IHC/train"
bci_dataset_ihc_test =  repo_path / "Datasets/BCI/IHC/test"
bci_dataset_he_train = repo_path / "Datasets/BCI/HE/train"
bci_dataset_he_test =  repo_path / "Datasets/BCI/HE/test"
endonuke_dataset = repo_path / "Datasets/Endonuke/data/crop_images"
pannuke_dataset = repo_path / "Datasets/Pannuke"

""" All the flags used for control and to choose before training/testing the model

    DATASET:
            bci_dataset_ihc_train   -- training/validation data IHC [BCI Dataset]              (IMG RESOLUTION: 512x512)
            bci_dataset_ihc_test    -- test data IHC [BCI Dataset]                             (IMG RESOLUTION: 512x512)
            bci_dataset_he_train    -- training/validation data HE [BCI Dataset]               (IMG RESOLUTION: 512x512)  
            bci_dataset_he_test     -- test data HE [BCI Dataset]                              (IMG RESOLUTION: 512x512)
            endonuke_dataset        -- training/validation/test data IHC [Endonuke Dataset]    (IMG RESOLUTION: 256x256)
            pannuke_dataset         -- training/validation data HE [Pannuke Dataset]           (IMG RESOLUTION: 256x256)
            
            TRAIN_DIR_IHC           -- Directory for IHC stain training/validation data
            TEST_DIR_IHC            -- Directory for IHC stain test data
            TRAIN_DIR_HE            -- Directory for HE stain training/validation data
            TEST_DIR_HE             -- Directory for HE stain test data 
            IMG_ORIGINAL_SIZE       -- Resolution of input images
            PATCHES_SIZE            -- Applies N crops to input images based on the dimension of the input images and patch size pretended // self.num_patches_per_image = (self.img_size // self.patch_size) ** 2 )
            SUBSET_PERCENTAGE       -- % of dataset to use for train/val/test (total size of 20k imgs)
            SHUFFLE_DATASET (bool)     -- Set to True/False to shuffle position index of data on the dataset
            transforms              -- Transformations applied to train/val data before passing it through the model
            test_transforms         -- Transformations applied to test data before passing it through the model

            
    GENERATOR/DISCRIMINATOR:
            D_FEATURES (list)       -- nº of channels/layer of the discriminator
            IN_CH                   -- nº of input channels to pass through the network (3 for RGB / 1 for black n white)
            N_RES_BLOCKS            -- nº of residual blocks of the generator architecture between downsampling and upsampling

            
    MODEL TRAINING:
            DEVICE                  -- Run on GPU and if not possible, to CPU
            BATCH_SIZE              -- Nº imgs of each stain to be processed at the same time, each epoch
            LEARNING_RATE           -- Optimizer lr to apply
            LAMBDA_IDENTITY         -- Weight factor to apply to  generator identity loss (between 0 and 1)
            LAMBDA_CYCLE            -- Weight factor to apply to  generator cycle loss (10 in cycleGAN paper)
            NUM_EPOCHS              -- Nº of times to pass the entire dataset through the network
            NUM_WORKERS             -- Nº of subprocesses to use for data loading
            EARLY_STOP              -- Nº of epochs after which the model should stop training
            FID_FREQUENCY           -- Every N epochs it should calculate FID scores for generated images
            FID_BATCH_SIZE          -- Nº of images per batch to calculate FID scores
        

    MODEL TRACKING:
            LOAD_MODEL (bool)            -- Set to True/False to load an already trained model, for further training or testing
            SAVE_MODEL (bool)            -- Set to True/False to save model after each epoch during training
            SAVE_CHECKPOINT_GEN_HE       -- Checkpoint filename for HE Generator network to save every time it improves during training
            SAVE_CHECKPOINT_GEN_IHC      -- Checkpoint filename for IHC Generator network to save every time it improves during training
            SAVE_CHECKPOINT_DISC_HE      -- Checkpoint filename for HE Discriminator network to save every time it improves during training
            SAVE_CHECKPOINT_DISC_IHC     -- Checkpoint filename for IHC Discriminator network to save every time it improves during training
            LOAD_CHECKPOINT_GEN_HE       -- Checkpoint filename for HE Generator network to load for further training/inference
            LOAD_CHECKPOINT_GEN_IHC      -- Checkpoint filename for IHC Generator network to load for further training/inference
            LOAD_CHECKPOINT_DISC_HE      -- Checkpoint filename for HE Discriminator network to load for further training/inference
            LOAD_CHECKPOINT_DISC_IHC     -- Checkpoint filename for IHC Discriminator network to load for further training/inference
            PRETRAINED_GEN_HE            -- Checkpoint filename of pretrained HE Generator for inference/finetuning
            PRETRAINED_GEN_IHC           -- Checkpoint filename of pretrained IHC Generator for inference/finetuning
            PRETRAINED_DISC_HE           -- Checkpoint filename of pretrained HE Discriminator for inference/finetuning
            PRETRAINED_DISC_IHC          -- Checkpoint filename of pretrained IHC Discriminator for inference/finetuning

"""

TRAIN_DIR_IHC = bci_dataset_ihc_train
TRAIN_DIR_HE = bci_dataset_he_train
TEST_DIR_IHC = bci_dataset_ihc_test
TEST_DIR_HE = bci_dataset_he_test

IMG_ORIGINAL_SIZE = 1024
PATCHES_SIZE = 512
SUBSET_PERCENTAGE = 100
SHUFFLE_DATASET = False

IN_CH = 3
D_FEATURES = [64, 128, 256, 512]
N_RES_BLOCKS = 9

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 150
NUM_WORKERS = 4

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10

EARLY_STOP = 20
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

SAVE_SUFFIX1 = NUM_EPOCHS
SAVE_SUFFIX2 = "try_1"
SAVE_CHECKPOINT_GEN_HE = repo_path / f"cycleGAN/training-models/genHE_{SAVE_SUFFIX1}_epochs_{SAVE_SUFFIX2}.pth.tar"
SAVE_CHECKPOINT_GEN_IHC = repo_path / f"cycleGAN/training-models/genIHC_{SAVE_SUFFIX1}_epochs_{SAVE_SUFFIX2}.pth.tar"
SAVE_CHECKPOINT_DISC_HE = repo_path / f"cycleGAN/training-models/discHE_{SAVE_SUFFIX1}_epochs_{SAVE_SUFFIX2}.pth.tar"
SAVE_CHECKPOINT_DISC_IHC = repo_path / f"cycleGAN/training-models/discIHC_{SAVE_SUFFIX1}_epochs_{SAVE_SUFFIX2}.pth.tar"

LOAD_SUFFIX1 = "150"
LOAD_SUFFIX2 = "try_1"
LOAD_CHECKPOINT_GEN_HE = repo_path / f"cycleGAN/training-models/genHE_{LOAD_SUFFIX1}_epochs_{LOAD_SUFFIX2}.pth.tar"
LOAD_CHECKPOINT_GEN_IHC = repo_path / f"cycleGAN/training-models/genIHC_{LOAD_SUFFIX1}_epochs_{LOAD_SUFFIX2}.pth.tar"
LOAD_CHECKPOINT_DISC_HE = repo_path / f"cycleGAN/training-models/discHE_{LOAD_SUFFIX1}_epochs_{LOAD_SUFFIX2}.pth.tar"
LOAD_CHECKPOINT_DISC_IHC = repo_path / f"cycleGAN/training-models/discIHC_{LOAD_SUFFIX1}_epochs_{LOAD_SUFFIX2}.pth.tar"

PRETRAINED_SUFFIX1 = 200
PRETRAINED_SUFFIX2 = 20240702
PRETRAINED_GEN_HE = repo_path / f"pretrained-models/genHE_{PRETRAINED_SUFFIX1}_epochs_{PRETRAINED_SUFFIX2}.pth.tar"
PRETRAINED_GEN_IHC = repo_path / f"pretrained-models/genIHC_{PRETRAINED_SUFFIX1}_epochs_{PRETRAINED_SUFFIX2}.pth.tar"
PRETRAINED_DISC_HE = repo_path / f"pretrained-models/discHE_{PRETRAINED_SUFFIX1}_epochs_{PRETRAINED_SUFFIX2}.pth.tar"
PRETRAINED_DISC_IHC = repo_path / f"pretrained-models/discIHC_{PRETRAINED_SUFFIX1}_epochs_{PRETRAINED_SUFFIX2}.pth.tar"


if __name__ == "__main__":
    print(f"Repository path: {repo_path}")
    print(f"Repository parent path: {parent_path}")
    pass
        
