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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train_func(D_HE, D_IHC, G_HE, G_IHC, optim_D, optim_G, G_scaler, D_scaler, cycle_loss, loss, loader, epoch, writer):
    
    loop = tqdm(loader, leave=True)         #Loop generates a progress bar while iterating over dataset
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        '''Set the generators and discriminators to training mode'''
        D_HE.train()
        D_IHC.train()
        G_HE.train()
        G_IHC.train()

        with torch.cuda.amp.autocast():     # For mixed precision training
            '''Train the Discriminator of HE images'''
            fake_HE = G_HE(ihc)
            D_real_HE = D_HE(he)
            D_fake_HE = D_HE(fake_HE.detach())

            label_real_HE = torch.ones_like(D_real_HE).to(config.DEVICE)
            label_fake_HE = torch.zeros_like(D_fake_HE).to(config.DEVICE)

            D_HE_real_loss = loss(D_real_HE, label_real_HE)
            D_HE_fake_loss = loss(D_fake_HE, label_fake_HE)
            D_HE_loss = D_HE_real_loss + D_HE_fake_loss

            '''Train the Discriminator of IHC images'''
            fake_IHC = G_IHC(he)
            D_real_IHC = D_IHC(ihc)
            D_fake_IHC = D_IHC(fake_IHC.detach())

            label_real_IHC = torch.ones_like(D_real_IHC).to(config.DEVICE)
            label_fake_IHC = torch.zeros_like(D_fake_IHC).to(config.DEVICE)

            D_IHC_real_loss = loss(D_real_IHC, label_real_IHC)
            D_IHC_fake_loss = loss(D_fake_IHC, label_fake_IHC)
            D_IHC_loss = D_IHC_real_loss + D_IHC_fake_loss

            D_loss = (D_HE_loss + D_IHC_loss) / 2   # Using simple averaging for the discriminator loss

            writer.add_scalar("HE(True) Discriminator Loss (mean)", D_HE_loss, epoch)
            writer.add_scalar("IHC(True) Discriminator Loss (mean)", D_IHC_loss, epoch)
            writer.add_scalar("Total Discriminator Loss (mean)", D_loss, epoch)

        optim_D.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(optim_D)
        D_scaler.update()

        with torch.cuda.amp.autocast():         # For mixed precision training
            '''Train the Generator of HE and IHC images'''
            D_fake_HE = D_HE(fake_HE)
            D_fake_IHC = D_IHC(fake_IHC)

            # Now the label for the generated images must be one (real) to fool the Discriminator
            label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    
            label_fake_IHC = torch.ones_like(D_fake_IHC).to(config.DEVICE)

            G_HE_loss = loss(D_fake_HE, label_fake_HE)
            G_IHC_loss = loss(D_fake_IHC, label_fake_IHC)

            # Cycle Consistency Loss
            cycle_IHC = G_IHC(fake_HE)
            cycle_HE = G_HE(fake_IHC)
            cycle_IHC_loss = cycle_loss(ihc, cycle_IHC)
            cycle_HE_loss = cycle_loss(he, cycle_HE)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_IHC = G_IHC(ihc)
            identity_HE = G_HE(he)
            identity_IHC_loss = cycle_loss(ihc, identity_IHC)
            identity_HE_loss = cycle_loss(he, identity_HE)

            # Total generator loss
            G_loss = (
                G_IHC_loss
                + G_HE_loss
                + cycle_IHC_loss * config.LAMBDA_CYCLE
                + cycle_HE_loss * config.LAMBDA_CYCLE
                + identity_HE_loss * config.LAMBDA_IDENTITY
                + identity_IHC_loss * config.LAMBDA_IDENTITY
                )

            writer.add_scalar("Cycle IHC Loss", cycle_IHC_loss, epoch)
            writer.add_scalar("Cycle HE Loss", cycle_HE_loss, epoch)
            writer.add_scalar("Identity IHC Loss", identity_IHC_loss, epoch)
            writer.add_scalar("Identity HE Loss", identity_HE_loss, epoch)
            writer.add_scalar("IHC(False) Fooling Discriminator Loss", G_IHC_loss, epoch)
            writer.add_scalar("HE(False) Fooling Discriminator Loss", G_HE_loss, epoch)
            writer.add_scalar("Total Generator Loss", G_loss, epoch)

        optim_G.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(optim_G)
        G_scaler.update()

        print(f"\nTRAIN EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx}/{len(loader)},"
                    + f" G_loss: {G_loss}, D_loss: {D_loss}")

        if epoch % 5 == 0:
            if idx % 800 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")


def eval_single_epoch(D_HE, D_IHC, G_HE, G_IHC, cycle_loss, loss, loader, epoch, writer):

    loop = tqdm(loader, leave=True)
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        D_HE.eval()
        D_IHC.eval()
        G_HE.eval()
        G_IHC.eval()

        with torch.no_grad():
            # fake_IHC = generator_HE(ihc)
            # fake_HE = generator_IHC(he)

            with torch.cuda.amp.autocast():     # For mixed precision training
                '''Train the Discriminator of HE images'''
                fake_HE = G_HE(ihc)
                D_real_HE = D_HE(he)
                D_fake_HE = D_HE(fake_HE.detach())

                label_real_HE = torch.ones_like(D_real_HE).to(config.DEVICE)
                label_fake_HE = torch.zeros_like(D_fake_HE).to(config.DEVICE)

                D_HE_real_loss = loss(D_real_HE, label_real_HE)
                D_HE_fake_loss = loss(D_fake_HE, label_fake_HE)
                D_HE_loss = D_HE_real_loss + D_HE_fake_loss

                '''Train the Discriminator of IHC images'''
                fake_IHC = G_IHC(he)
                D_real_IHC = D_IHC(ihc)
                D_fake_IHC = D_IHC(fake_IHC.detach())

                label_real_IHC = torch.ones_like(D_real_IHC).to(config.DEVICE)
                label_fake_IHC = torch.zeros_like(D_fake_IHC).to(config.DEVICE)

                D_IHC_real_loss = loss(D_real_IHC, label_real_IHC)
                D_IHC_fake_loss = loss(D_fake_IHC, label_fake_IHC)
                D_IHC_loss = D_IHC_real_loss + D_IHC_fake_loss

                D_loss = (D_HE_loss + D_IHC_loss) / 2   # Using simple averaging for the discriminator loss

                writer.add_scalar("[VAL] - HE(True) Discriminator Loss (mean)", D_HE_loss, epoch)
                writer.add_scalar("[VAL] - IHC(True) Discriminator Loss (mean)", D_IHC_loss, epoch)
                writer.add_scalar("[VAL] - Total Discriminator Loss (mean)", D_loss, epoch)

            with torch.cuda.amp.autocast():         # For mixed precision training
                '''Train the Generator of HE and IHC images'''
                D_fake_HE = D_HE(fake_HE)
                D_fake_IHC = D_IHC(fake_IHC)

                # Now the label for the generated images must be one (real) to fool the Discriminator
                label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    
                label_fake_IHC = torch.ones_like(D_fake_IHC).to(config.DEVICE)

                G_HE_loss = loss(D_fake_HE, label_fake_HE)
                G_IHC_loss = loss(D_fake_IHC, label_fake_IHC)

                # Cycle Consistency Loss
                cycle_IHC = G_IHC(fake_HE)
                cycle_HE = G_HE(fake_IHC)
                cycle_IHC_loss = cycle_loss(ihc, cycle_IHC)
                cycle_HE_loss = cycle_loss(he, cycle_HE)

                # Identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_IHC = G_IHC(ihc)
                identity_HE = G_HE(he)
                identity_IHC_loss = cycle_loss(ihc, identity_IHC)
                identity_HE_loss = cycle_loss(he, identity_HE)

                # Total generator loss
                G_loss = (
                    G_IHC_loss
                    + G_HE_loss
                    + cycle_IHC_loss * config.LAMBDA_CYCLE
                    + cycle_HE_loss * config.LAMBDA_CYCLE
                    + identity_HE_loss * config.LAMBDA_IDENTITY
                    + identity_IHC_loss * config.LAMBDA_IDENTITY
                )

                writer.add_scalar("[VAL] - Cycle IHC Loss", cycle_IHC_loss, epoch)
                writer.add_scalar("[VAL] - Cycle HE Loss", cycle_HE_loss, epoch)
                writer.add_scalar("[VAL] - Identity IHC Loss", identity_IHC_loss, epoch)
                writer.add_scalar("[VAL] - Identity HE Loss", identity_HE_loss, epoch)
                writer.add_scalar("[VAL] - IHC(False) Fooling Discriminator Loss", G_IHC_loss, epoch)
                writer.add_scalar("[VAL] - HE(False) Fooling Discriminator Loss", G_HE_loss, epoch)
                writer.add_scalar("[VAL] - Total Generator Loss", G_loss, epoch)
        
        print(f"\nVALIDATION EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx}/{len(loader)},"
                    + f" G_loss: {G_loss}, D_loss: {D_loss}")
        
        if epoch % 5 == 0:
            if idx % 210 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")



def custom_collate(batch):
    # Create separate lists for each key in the batch
    A_patches = [item['A'] for item in batch]
    B_patches = [item['B'] for item in batch]

    # Convert the lists to tensors (if needed)
    # You can use transforms.ToTensor() to convert PIL images to tensors
    A_patches = torch.stack(A_patches)  # Assuming A_patches is a list of tensors
    B_patches = torch.stack(B_patches)  # Assuming B_patches is a list of tensors

    return {'A': A_patches, 'B': B_patches}  # Return the collated batch as a dictionary


def main():

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

    # Losses used during training
    cycle_loss = nn.L1Loss()
    discrim_loss = nn.MSELoss()

    start_epoch = 0
    log_dir = None

    # Load checkpoints if necessary
    if config.LOAD_MODEL:
        start_epoch, log_dir = load_checkpoint(config.CHECKPOINT_GEN_HE, gen_HE, optim_gen, config.LEARNING_RATE)
        start_epoch = max(start_epoch, load_checkpoint(config.CHECKPOINT_GEN_IHC, gen_IHC, optim_gen, config.LEARNING_RATE)[0])
        start_epoch = max(start_epoch, load_checkpoint(config.CHECKPOINT_DISC_HE, disc_HE, optim_disc, config.LEARNING_RATE)[0])
        start_epoch = max(start_epoch, load_checkpoint(config.CHECKPOINT_DISC_IHC, disc_IHC, optim_disc, config.LEARNING_RATE)[0])
    
    if log_dir is None:
        log_dir = f"logs/GAN_{config.NUM_EPOCHS}_epochs_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    writer = SummaryWriter(log_dir=log_dir)

    '''Initialize datasets and dataloaders:
        1) Create dataset for later split between training/validation
        2) Split between training and validation size
        3) Create train and validation sets and loaders
    '''
    my_dataset = GanDataset(config.TRAIN_DIR_IHC, config.TRAIN_DIR_HE, config.SUBSET_PERCENTAGE, patch_size=512, transform=config.transforms, shuffle=config.SHUFFLE_DATA)
    
    dataset_lenght = len(my_dataset)
    train_size = int(0.8 * dataset_lenght)
    val_size = dataset_lenght - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=custom_collate)

    # Initialize gradient scalers for mixed precision training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"TRAINING MODEL [Epoch {epoch}]:")
        train_func(disc_HE, disc_IHC, gen_HE, gen_IHC, optim_disc, optim_gen, g_scaler, d_scaler, cycle_loss, discrim_loss, train_loader, epoch, writer)
        eval_single_epoch(disc_HE, disc_IHC, gen_HE, gen_IHC, cycle_loss, discrim_loss, val_loader, epoch, writer)

        # Save model checkpoints
        if config.SAVE_MODEL:
           save_checkpoint(epoch, gen_HE, optim_gen, filename=config.CHECKPOINT_GEN_HE, log_dir=log_dir)
           save_checkpoint(epoch, gen_IHC, optim_gen, filename=config.CHECKPOINT_GEN_IHC, log_dir=log_dir)
           save_checkpoint(epoch, disc_HE, optim_disc, filename=config.CHECKPOINT_DISC_HE, log_dir=log_dir)
           save_checkpoint(epoch, disc_IHC, optim_disc, filename=config.CHECKPOINT_DISC_IHC, log_dir=log_dir)

    writer.close()

if __name__ == "__main__":
    main()


