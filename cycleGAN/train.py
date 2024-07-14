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
from utils import load_checkpoint, save_checkpoint, set_seed, custom_collate, create_directories
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def train_func(D_HE, D_IHC, G_HE, G_IHC, optim_D, optim_G, G_scaler, D_scaler, cycle_loss, disc_loss, ident_loss, loader, epoch, writer):
    
    '''Set the generators and discriminators to training mode'''
    D_HE.train()
    D_IHC.train()
    G_HE.train()
    G_IHC.train()

    loop = tqdm(loader, leave=True)     
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        with torch.cuda.amp.autocast():     # For mixed precision training
            '''Train the Discriminator of HE images'''
            fake_HE = G_HE(ihc)
            D_real_HE = D_HE(he)
            D_fake_HE = D_HE(fake_HE.detach())

            label_real_HE = torch.ones_like(D_real_HE).to(config.DEVICE)
            label_fake_HE = torch.zeros_like(D_fake_HE).to(config.DEVICE)

            D_HE_real_loss = disc_loss(D_real_HE, label_real_HE)
            D_HE_fake_loss = disc_loss(D_fake_HE, label_fake_HE)
            D_HE_loss = D_HE_real_loss + D_HE_fake_loss

            '''Train the Discriminator of IHC images'''
            fake_IHC = G_IHC(he)
            D_real_IHC = D_IHC(ihc)
            D_fake_IHC = D_IHC(fake_IHC.detach())

            label_real_IHC = torch.ones_like(D_real_IHC).to(config.DEVICE)
            label_fake_IHC = torch.zeros_like(D_fake_IHC).to(config.DEVICE)

            D_IHC_real_loss = disc_loss(D_real_IHC, label_real_IHC)
            D_IHC_fake_loss = disc_loss(D_fake_IHC, label_fake_IHC)
            D_IHC_loss = D_IHC_real_loss + D_IHC_fake_loss

            D_loss = (D_HE_loss + D_IHC_loss) / 2   # Using simple averaging for the discriminator loss

            writer.add_scalars("[TRAIN] - HE/IHC Discriminator Loss", {"HE": D_HE_loss, "IHC": D_IHC_loss}, epoch)
            writer.add_scalar("[TRAIN] - Total Discriminator Loss", D_loss, epoch)

        optim_D.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(optim_D)
        D_scaler.update()

        with torch.cuda.amp.autocast():   
            '''Train the Generator of HE and IHC images'''
            D_fake_HE = D_HE(fake_HE)
            D_fake_IHC = D_IHC(fake_IHC)

            # Now the label for the generated images must be one (real) to fool the Discriminator
            label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    
            label_fake_IHC = torch.ones_like(D_fake_IHC).to(config.DEVICE)

            D_gen_HE_loss = disc_loss(D_fake_HE, label_fake_HE)
            D_gen_IHC_loss = disc_loss(D_fake_IHC, label_fake_IHC)

            # Cycle Consistency Loss
            cycle_IHC = G_IHC(fake_HE)
            cycle_HE = G_HE(fake_IHC)
            cycle_IHC_loss = cycle_loss(ihc, cycle_IHC)
            cycle_HE_loss = cycle_loss(he, cycle_HE)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_IHC = G_IHC(ihc)
            identity_HE = G_HE(he)
            identity_IHC_loss = ident_loss(ihc, identity_IHC)
            identity_HE_loss = ident_loss(he, identity_HE)

            # Total generator loss
            G_loss = (
                D_gen_HE_loss
                + D_gen_IHC_loss
                + cycle_IHC_loss * config.LAMBDA_CYCLE
                + cycle_HE_loss * config.LAMBDA_CYCLE
                + identity_HE_loss * config.LAMBDA_IDENTITY
                + identity_IHC_loss * config.LAMBDA_IDENTITY
                )

            writer.add_scalars("[TRAIN] - Cycle Loss", {"HE": cycle_HE_loss, "IHC": cycle_IHC_loss}, epoch)
            writer.add_scalars("[TRAIN] - Identity Loss", {"HE": identity_HE_loss, "IHC" : identity_IHC_loss}, epoch)
            writer.add_scalars("[TRAIN] - Gen_Img Discriminator Loss", {"Fake_HE": D_gen_HE_loss, "Fake_IHC" : D_gen_IHC_loss}, epoch)
            writer.add_scalar("[TRAIN] - Total Generator Loss", G_loss, epoch)

        optim_G.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(optim_G)
        G_scaler.update()

        if epoch % 5 == 0 and idx % 1000 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
        
        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    print(f"\nTRAIN EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}, D_loss: {D_loss}\n")
    
    return G_loss, D_loss

def eval_single_epoch(D_HE, D_IHC, G_HE, G_IHC, cycle_loss, disc_loss, ident_loss, loader, epoch, writer):

    G_HE.eval()
    G_IHC.eval()

    loop = tqdm(loader, leave=True)
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast():   
                fake_HE = G_HE(ihc)
                fake_IHC = G_IHC(he)

                D_fake_HE = D_HE(fake_HE)
                D_fake_IHC = D_IHC(fake_IHC)

                # Now the label for the generated images must be one (real) to fool the Discriminator
                label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    
                label_fake_IHC = torch.ones_like(D_fake_IHC).to(config.DEVICE)

                D_gen_HE_loss = disc_loss(D_fake_HE, label_fake_HE)
                D_gen_IHC_loss = disc_loss(D_fake_IHC, label_fake_IHC)

                # Cycle Consistency Loss
                cycle_IHC = G_IHC(fake_HE)
                cycle_HE = G_HE(fake_IHC)
                cycle_IHC_loss = cycle_loss(ihc, cycle_IHC)
                cycle_HE_loss = cycle_loss(he, cycle_HE)

                # Identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_IHC = G_IHC(ihc)
                identity_HE = G_HE(he)
                identity_IHC_loss = ident_loss(ihc, identity_IHC)
                identity_HE_loss = ident_loss(he, identity_HE)

                # Total generator loss
                G_loss = (
                    D_gen_HE_loss
                    + D_gen_IHC_loss
                    + cycle_IHC_loss * config.LAMBDA_CYCLE
                    + cycle_HE_loss * config.LAMBDA_CYCLE
                    + identity_HE_loss * config.LAMBDA_IDENTITY
                    + identity_IHC_loss * config.LAMBDA_IDENTITY
                )

                writer.add_scalars("[VAL] - Cycle Loss", {"HE": cycle_HE_loss, "IHC": cycle_IHC_loss}, epoch)
                writer.add_scalars("[VAL] - Identity Loss", {"HE": identity_HE_loss, "IHC" : identity_IHC_loss}, epoch)
                writer.add_scalars("[VAL] - Gen_Img Discriminator Loss", {"Fake_HE": D_gen_HE_loss, "Fake_IHC" : D_gen_IHC_loss}, epoch)
                writer.add_scalar("[VAL] - Total Generator Loss", G_loss, epoch)
        
        if epoch % 5 == 0 and idx % 170 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.repo_path / f"cycleGAN/gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")

    print(f"\nVALIDATION EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}\n")

    return G_loss

def main():
    set_seed(42) # To ensure reproducibility

    create_directories() # Create paths for saving images during train, val and test

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
    identity_loss = nn.L1Loss()
    discrim_loss = nn.MSELoss()

    my_dataset = GanDataset(
        config.TRAIN_DIR_IHC, 
        config.TRAIN_DIR_HE, 
        config.SUBSET_PERCENTAGE,
        transform=config.transforms, 
        shuffle=config.SHUFFLE_DATASET
    )
    dataset_length = len(my_dataset)
    train_size = int(0.8 * len(my_dataset))
    val_size = dataset_length - train_size
    print(f"Train dataset size has {train_size} data points --> {train_size/dataset_length}% of training data")
    print(f"Val dataset size has {val_size} data points --> {val_size/dataset_length}% of training data")
    train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=custom_collate,
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        pin_memory=True, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=custom_collate,
        drop_last=True
    )

    # Initialize gradient scalers for mixed precision training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    log_dir = None
    val_loss = None

    # Load checkpoints if necessary
    if config.LOAD_MODEL:
        start_epoch, log_dir, val_loss = load_checkpoint(config.LOAD_CHECKPOINT_GEN_HE, gen_HE, optim_gen, config.LEARNING_RATE)
        start_epoch = max(start_epoch, load_checkpoint(config.LOAD_CHECKPOINT_GEN_IHC, gen_IHC, optim_gen, config.LEARNING_RATE)[0])
        start_epoch = max(start_epoch, load_checkpoint(config.LOAD_CHECKPOINT_DISC_HE, disc_HE, optim_disc, config.LEARNING_RATE)[0])
        start_epoch = max(start_epoch, load_checkpoint(config.LOAD_CHECKPOINT_DISC_IHC, disc_IHC, optim_disc, config.LEARNING_RATE)[0])
        print(f"Val_loss of loading is {val_loss} from epoch {start_epoch-1}")

    if log_dir is None:
        log_dir = config.repo_path / f"cycleGAN/logs/GAN_{config.SAVE_SUFFIX1}_epochs_{config.SAVE_SUFFIX2}"
    if val_loss is None:
        val_loss = float('inf')

    writer = SummaryWriter(log_dir=log_dir)

    # Early stopping parameters
    patience = config.EARLY_STOP
    best_val_loss = val_loss
    epochs_no_improve = 0
    best_epoch = 0

    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nTRAINING MODEL [Epoch {epoch}]:")
        gen_train_loss, disc_train_loss = train_func(
                                            disc_HE, disc_IHC, 
                                            gen_HE, gen_IHC, 
                                            optim_disc, optim_gen, 
                                            g_scaler, d_scaler, 
                                            cycle_loss, discrim_loss, identity_loss,
                                            train_loader, 
                                            epoch, 
                                            writer
                                        )

        if epoch % config.FID_FREQUENCY == 0:
            print(f"CALCULATING FID SCORES [Epoch {epoch}]:")
            fid_he, fid_ihc = evaluate_fid_scores(gen_HE, gen_IHC, val_loader, config.DEVICE, config.FID_BATCH_SIZE)
            print(f"\nFID Scores - HE: {fid_he}, IHC: {fid_ihc}")
            writer.add_scalars("FID Scores", {"HE": fid_he, "IHC": fid_ihc}, epoch)

        print(f"\nVALIDATING MODEL [Epoch {epoch}]:")
        gen_val_loss = eval_single_epoch(
                            disc_HE, disc_IHC, 
                            gen_HE, gen_IHC, 
                            cycle_loss, discrim_loss, identity_loss,
                            val_loader, 
                            epoch, 
                            writer
                        )

        writer.add_scalars("Generators Losses", {"train": gen_train_loss, "val": gen_val_loss}, epoch)

        # Check for improvement
        if gen_val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = gen_val_loss
            epochs_no_improve = 0
            # Save the best model
            if config.SAVE_MODEL:
                save_checkpoint(epoch, gen_HE, optim_gen, filename=config.SAVE_CHECKPOINT_GEN_HE, log_dir=log_dir, loss=best_val_loss)
                save_checkpoint(epoch, gen_IHC, optim_gen, filename=config.SAVE_CHECKPOINT_GEN_IHC, log_dir=log_dir, loss=best_val_loss)
                save_checkpoint(epoch, disc_HE, optim_disc, filename=config.SAVE_CHECKPOINT_DISC_HE, log_dir=log_dir, loss=best_val_loss)
                save_checkpoint(epoch, disc_IHC, optim_disc, filename=config.SAVE_CHECKPOINT_DISC_IHC, log_dir=log_dir, loss=best_val_loss)
        else:
            epochs_no_improve += 1
            print(f"Best epoch so far was {best_epoch}\n")
            print(f"Nº of epochs without improvement: {epochs_no_improve}\n")

        # Check for early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            print(f"Last model saved was on epoch {best_epoch} with loss: {best_val_loss}")
            break

    writer.close()

if __name__ == "__main__":
    main()
