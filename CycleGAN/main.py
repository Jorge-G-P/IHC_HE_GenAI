# main.py
import torch
import config
import torch.nn as nn
import torch.optim as optim
from HE_IHC_dataset import GanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint, save_checkpoint
from torchvision.utils import save_image

def train_func(disc_HE, disc_IHC, gen_HE, gen_IHC, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, loader):
    
    #Loop generates a progress bar
    loop = tqdm(loader, leave=True)
    
    for idx, (IHC, HE) in enumerate(loop):
        IHC = IHC.to(config.DEVICE)
        HE = HE.to(config.DEVICE)

        # Train Discriminator: Generates fake images using the generators.
        with torch.cuda.amp.autocast():
            # Generate fake horse from zebra
            fake_HE = gen_HE(IHC)
            # Discriminator H real and fake losses
            D_HE_real = disc_HE(HE)
            D_HE_fake = disc_HE(fake_HE.detach())
            D_HE_real_loss = mse(D_HE_real, torch.ones_like(D_HE_real))
            D_HE_fake_loss = mse(D_HE_fake, torch.zeros_like(D_HE_fake))
            D_HE_loss = D_HE_real_loss + D_HE_fake_loss

            # Generate fake zebra from horse
            fake_IHC = gen_IHC(HE)
            # Discriminator Z real and fake losses
            D_IHC_real = disc_IHC(IHC)
            D_IHC_fake = disc_IHC(fake_IHC.detach())
            D_IHC_real_loss = mse(D_IHC_real, torch.ones_like(D_IHC_real))
            D_IHC_fake_loss = mse(D_IHC_fake, torch.zeros_like(D_IHC_fake))
            D_IHC_loss = D_IHC_real_loss + D_IHC_fake_loss

            D_loss = (D_HE_loss + D_IHC_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator H & Z
        with torch.cuda.amp.autocast():
            # Discriminator H and Z fake losses for generator
            D_HE_fake = disc_HE(fake_HE)
            D_IHC_fake = disc_IHC(fake_IHC)
            loss_G_HE = mse(D_HE_fake, torch.ones_like(D_HE_fake))
            loss_G_IHC = mse(D_IHC_fake, torch.ones_like(D_IHC_fake))
            # Cycle consistency loss
            cycle_IHC = gen_IHC(fake_HE)
            cycle_HE = gen_HE(fake_IHC)
            cycle_IHC_loss = L1(IHC, cycle_IHC)
            cycle_HE_loss = L1(HE, cycle_HE)

            # Total generator loss
            G_loss = (
                loss_G_IHC
                + loss_G_HE
                + cycle_IHC_loss * config.LAMBDA_CYCLE
                + cycle_HE_loss * config.LAMBDA_CYCLE
            )
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            #*0.5+0.5 normalices the image
            save_image(HE * 0.5 + 0.5, f"/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/BCI_dataset/Saved images/HE_{idx}.png")
            save_image(fake_HE * 0.5 + 0.5, f"/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/BCI_dataset/Saved images/fake_HE_{idx}.png")
            save_image(IHC * 0.5 + 0.5, f"/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/BCI_dataset/Saved images/IHC_{idx}.png")
            save_image(fake_IHC * 0.5 + 0.5, f"/Users/josep/Desktop/aidl-2024-spring-mlops/BCI/BCI_dataset/Saved images/fake_IHC_{idx}.png")

def main():
    # Initialize Discriminators from discriminator.py 
    disc_HE = Discriminator(in_channels=3).to(config.DEVICE) 
    #DEVICE is ("cuda" if torch.cuda.is_available() else "cpu") defined in config.py
    disc_IHC = Discriminator(in_channels=3).to(config.DEVICE)
    # Initialize Generators from generator.py
    gen_HE = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_IHC = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # Initialize Optimizers
    # Adam is an adaptive learning rate optimization algorithm
    # BETA1: exponential decay rate for the first moment estimates
    # BETA2: exponential decay rate for the second moment estimates
    opt_disc = optim.Adam(
        list(disc_HE.parameters()) + list(disc_IHC.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_HE.parameters()) + list(gen_IHC.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load checkpoints if necessary
    if config.LOAD_MODEL:
        # LOAD_MODEL is set to FALSE for now
        # load_checkpoint is in utils.py
        load_checkpoint(config.CHECKPOINT_GEN_HE, gen_HE, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_IHC, gen_IHC, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_HE, disc_HE, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_IHC, disc_IHC, opt_disc, config.LEARNING_RATE)

    # Initialize datasets and dataloaders
    # Here we use the dataset we created
    train_dataset = GanDataset(config.TRAIN_DIR_IHC, config.TRAIN_DIR_HE, transform=config.transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = GanDataset(config.VAL_DIR_IHC, config.VAL_DIR_HE, transform=config.transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=config.NUM_WORKERS)

    # Initialize gradient scalers for mixed precision training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        train_func(disc_HE, disc_IHC, gen_HE, gen_IHC, opt_disc, opt_gen, g_scaler, d_scaler, L1, mse, train_loader)

        # Save model checkpoints
        #if config.SAVE_MODEL:
        #    save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
        #    save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
        #    save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
        #    save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()
