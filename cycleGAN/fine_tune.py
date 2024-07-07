import torch
import config
import torch.nn as nn
import torch.optim as optim
from HE_IHC_dataset import GanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from evaluate import evaluate_fid_scores
from utils import load_checkpoint, save_checkpoint, set_seed, custom_collate
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def load_pretrained_model():

    disc_HE = Discriminator(in_channels=config.IN_CH, features=config.D_FEATURES).to(config.DEVICE) 
    disc_IHC = Discriminator(in_channels=config.IN_CH, features=config.D_FEATURES).to(config.DEVICE)

    gen_HE = Generator(img_channels=3, num_residuals=config.N_RES_BLOCKS).to(config.DEVICE)
    gen_IHC = Generator(img_channels=3, num_residuals=config.N_RES_BLOCKS).to(config.DEVICE)

    optim_disc = optim.Adam(
        list(disc_HE.parameters()) + list(disc_IHC.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    optim_gen = optim.Adam(
        list(gen_HE.parameters()) + list(gen_IHC.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    train_epoch = load_checkpoint(config.CHECKPOINT_GEN_HE, gen_HE, optim_gen, config.LEARNING_RATE)[0]
    train_epoch = load_checkpoint(config.CHECKPOINT_GEN_IHC, gen_IHC, optim_gen, config.LEARNING_RATE)[0]
    train_epoch = load_checkpoint(config.CHECKPOINT_DISC_HE, disc_HE, optim_disc, config.LEARNING_RATE)[0]
    train_epoch = load_checkpoint(config.CHECKPOINT_DISC_IHC, disc_IHC, optim_disc, config.LEARNING_RATE)[0]
    print(f"\nModel loaded for test set was trained during {train_epoch-1} epochs\n")

    return disc_HE, disc_IHC, gen_HE, gen_IHC, optim_gen, optim_disc


def main():

    set_seed(42) # To ensure reproducibility

    disc_HE, disc_IHC, gen_HE, gen_IHC, optim_gen, optim_disc = load_pretrained_model()

    features_discHE = disc_HE.get_features()
    features_discIHC = disc_IHC.get_features()
    features_genHE = gen_HE.get_features()
    features_genIHC = gen_IHC.get_features()
    # print(f"\nGenerators Structure:\n {features_genHE}")
    # print(f"\nDiscriminators Structure:\n {features_discHE}\n\n")

    # Freeze pre-trained model layers
    for layer in features_genHE[:5]:        # Freeze layers 0 to 4
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_genHE[5:]:        # Train layers 5 to 5
        for param in layer.parameters():
            param.requires_grad = True

    for layer in features_genIHC[:5]:       # Freeze layers 0 to 4
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_genIHC[5:]:       # Train layers 5 to 5
        for param in layer.parameters():
            param.requires_grad = True

    for layer in features_discHE[:3]:       # Freeze layers 0 to 2
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_discHE[3:]:       # Train layers 3 to 3
        for param in layer.parameters():
            param.requires_grad = True

    for layer in features_discIHC[:3]:      # Freeze layers 0 to 2
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_discIHC[3:]:      # Train layers 3 to 3
        for param in layer.parameters():
            param.requires_grad = True
    
    gen_last_layer = gen_HE.clone_layer(gen_HE.last_layer)
    # print("\n", gen_last_layer, "\n")
    model_genHE = features_genHE
    model_genHE.append(gen_last_layer)
    model_genHE.to(config.DEVICE)
    # print("\n", model_genHE, "\n")

    # Print to confirm if weights of new layer created with clone_layer() are not the same as the layer from the loaded pretrained model
    for param1, param2 in zip(gen_HE.last_layer.parameters(), gen_last_layer.parameters()):
        if torch.equal(param1.data, param2.data):
            print("Weights are the same")
        else:
            print("Weights are different")


if __name__ == "__main__":
    main()