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

    gen_HE = Generator(img_channels=3, num_residuals=config.N_RES_BLOCKS).to(config.DEVICE)
    gen_IHC = Generator(img_channels=3, num_residuals=config.N_RES_BLOCKS).to(config.DEVICE)

    optim_disc = optim.Adam(
        list(disc_HE.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    optim_gen = optim.Adam(
        list(gen_HE.parameters()) + list(gen_IHC.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    train_epoch = load_checkpoint(config.PRETRAINED_GEN_HE, gen_HE, optim_gen, config.LEARNING_RATE)[0]
    train_epoch = load_checkpoint(config.PRETRAINED_GEN_HE, gen_IHC, optim_gen, config.LEARNING_RATE)[0]
    train_epoch = load_checkpoint(config.PRETRAINED_DISC_HE, disc_HE, optim_disc, config.LEARNING_RATE)[0]
    print(f"\nModels parameters updated with pretrained network\n")

    return disc_HE, gen_HE, gen_IHC, optim_gen, optim_disc

def new_gan_model(disc_HE, gen_HE, gen_IHC, gen_frozen_layers, disc_frozen_layers):

    features_discHE = disc_HE.get_features()
    features_genHE = gen_HE.get_features()
    features_genIHC = gen_IHC
    # print(f"\nGenerators Structure:\n {features_genHE}")
    # print(f"\nDiscriminator IHC Structure:\n {features_genIHC}\n\n")

    # Freeze pre-trained model layers
    for layer in features_genHE[:gen_frozen_layers]:        # Freeze layers 0 to 4
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_genHE[gen_frozen_layers:]:        # Train layers 5 to 5
        for param in layer.parameters():
            param.requires_grad = True

    for layer in features_genIHC:                        # Freeze layers 0 to 4
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_discHE[:disc_frozen_layers]:       # Freeze layers 0 to 2
        for param in layer.parameters():
            param.requires_grad = False

    for layer in features_discHE[disc_frozen_layers:]:       # Train layers 3 to 3
        for param in layer.parameters():
            param.requires_grad = True

    
    gen_last_layer = gen_HE.clone_layer(gen_HE.last_layer)
    disc_last_layer = disc_HE.clone_layer(disc_HE.model[-1])
    # print("\n", disc_last_layer, "\n")

    model_genHE = features_genHE
    model_discHE = features_discHE
    model_genIHC = features_genIHC

    model_genHE.append(gen_last_layer)
    model_discHE.append(disc_last_layer)

    model_genHE.to(config.DEVICE)
    model_discHE.to(config.DEVICE)
    model_genIHC.to(config.DEVICE)
    # print("\n", model_discHE, "\n")

    # # Print to confirm if weights of new layer created with clone_layer() are not the same as the layer from the loaded pretrained model
    # for param1, param2 in zip(gen_HE.last_layer.parameters(), gen_last_layer.parameters()):
    #     if torch.equal(param1.data, param2.data):
    #         print("Weights are the same")
    #     else:
    #         print("Weights are different")

    return model_discHE, model_genHE, model_genIHC

def train_func(D_HE, G_HE, G_IHC, optim_D, optim_G, G_scaler, D_scaler, cycle_loss, disc_loss, ident_loss, loader, epoch, writer):
    
    '''Set the generators and discriminators to training mode'''
    D_HE.train()
    G_HE.train()
    # G_IHC.train()

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

            D_loss = D_HE_loss  

            writer.add_scalars("[TRAIN] - HE Discriminator Loss", D_HE_loss, epoch)

        optim_D.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(optim_D)
        D_scaler.update()

        with torch.cuda.amp.autocast():  
            '''Train the Generator of HE images'''
            fake_IHC = G_IHC(he)
            D_fake_HE = D_HE(fake_HE)

            # Now the label for the generated images must be one (real) to fool the Discriminator
            label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    

            D_gen_HE_loss = disc_loss(D_fake_HE, label_fake_HE)

            # Cycle Consistency Loss
            cycle_HE = G_HE(fake_IHC)
            cycle_HE_loss = cycle_loss(he, cycle_HE)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_HE = G_HE(he)
            identity_HE_loss = ident_loss(he, identity_HE)

            # Total generator loss
            G_loss = (
                D_gen_HE_loss
                + cycle_HE_loss * config.LAMBDA_CYCLE
                + identity_HE_loss * config.LAMBDA_IDENTITY
                )

            writer.add_scalar("[TRAIN] - HE Cycle Loss", cycle_HE_loss, epoch)
            writer.add_scalar("[TRAIN] - HE Identity Loss", identity_HE_loss, epoch)
            writer.add_scalar("[TRAIN] - Gen_HE Discriminator Loss", D_gen_HE_loss, epoch)
            writer.add_scalar("[TRAIN] - HE Generator Loss", G_loss, epoch)

        optim_G.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(optim_G)
        G_scaler.update()

        if epoch % 5 == 0 and idx % 1000 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
        
        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    print(f"\nTRAIN EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}, D_loss: {D_loss}\n")
    
    return G_loss, D_loss

def eval_single_epoch(D_HE, G_HE, G_IHC, cycle_loss, disc_loss, ident_loss, loader, epoch, writer):

    G_HE.eval()

    loop = tqdm(loader, leave=True)
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        with torch.no_grad():
            with torch.cuda.amp.autocast():   
                fake_HE = G_HE(ihc)
                fake_IHC = G_IHC(he)
                D_fake_HE = D_HE(fake_HE)

                label_fake_HE = torch.ones_like(D_fake_HE).to(config.DEVICE)    
                D_gen_HE_loss = disc_loss(D_fake_HE, label_fake_HE)

                # Cycle Consistency Loss
                cycle_HE = G_HE(fake_IHC)
                cycle_HE_loss = cycle_loss(he, cycle_HE)

                # Identity loss (remove these for efficiency if you set lambda_identity=0)
                identity_HE = G_HE(he)
                identity_HE_loss = ident_loss(he, identity_HE)

                # Total generator loss
                G_loss = (
                    D_gen_HE_loss
                    + cycle_HE_loss * config.LAMBDA_CYCLE
                    + identity_HE_loss * config.LAMBDA_IDENTITY
                )

                writer.add_scalar("[VAL] - HE Cycle Loss", cycle_HE_loss, epoch)
                writer.add_scalar("[VAL] - HE Identity Loss", identity_HE_loss, epoch)
                writer.add_scalar("[VAL] - Gen_HE Discriminator Loss", D_gen_HE_loss, epoch)
                writer.add_scalar("[VAL] - HE Generator Loss", G_loss, epoch)
        
        if epoch % 5 == 0 and idx % 170 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
        
        loop.set_postfix(G_loss=G_loss.item())

    print(f"\nVALIDATION EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}\n")

    return G_loss


def main():

    set_seed(42) # To ensure reproducibility

    disc_HE, gen_HE, gen_IHC, optim_gen, optim_disc = load_pretrained_model()

    finetuned_discHE, finetuned_genHE, gen_IHC = new_gan_model(disc_HE, gen_HE, gen_IHC, gen_frozen_layers=5, disc_frozen_layers=3)

     # Losses used during training
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    discrim_loss = nn.MSELoss()

    endonuke_dataset = GanDataset(
        config.ENDONUKE_DIR_IHC,
        config.ENDONUKE_DIR_HE,
        config.SUBSET_PERCENTAGE, 
        patch_size=512, 
        transform=config.transforms, 
        shuffle=config.SHUFFLE_DATA
    )
    dataset_length = len(endonuke_dataset)
    train_size = int(0.8 * len(endonuke_dataset))
    val_size = dataset_length - train_size
    print(f"Train dataset size has {train_size} data points --> {train_size/dataset_length}% of training data")
    print(f"Val dataset size has {val_size} data points --> {val_size/dataset_length}% of training data")
    train_dataset, val_dataset = torch.utils.data.random_split(endonuke_dataset, [train_size, val_size])

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
        start_epoch, log_dir, val_loss = load_checkpoint(config.CHECKPOINT_GEN_HE, finetuned_genHE, optim_gen, config.LEARNING_RATE)
        start_epoch = max(start_epoch, load_checkpoint(config.CHECKPOINT_DISC_HE, finetuned_discHE, optim_disc, config.LEARNING_RATE)[0])
        print(f"Val_loss of loading is {val_loss} from epoch {start_epoch-1}")

    if log_dir is None:
        log_dir = f"logs/GAN_FT_{config.NUM_EPOCHS}_epochs_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
        print(f"TRAINING MODEL [Epoch {epoch}]:")
        gen_train_loss, disc_train_loss = train_func(
                                            finetuned_discHE,
                                            finetuned_genHE, gen_IHC, 
                                            optim_disc, optim_gen, 
                                            g_scaler, d_scaler, 
                                            cycle_loss, discrim_loss, identity_loss,
                                            train_loader, 
                                            epoch, 
                                            writer
                                        )

        if epoch % config.FID_FREQUENCY == 0:
            print(f"CALCULATING FID SCORES [Epoch {epoch}]:")
            fid_he, fid_ihc = evaluate_fid_scores(finetuned_genHE, gen_IHC, val_loader, config.DEVICE, config.FID_BATCH_SIZE)
            print(f"\nFID Scores - HE: {fid_he}, IHC: {fid_ihc}")
            writer.add_scalars("FID Scores", {"HE": fid_he, "IHC": fid_ihc}, epoch)

        print(f"\nVALIDATING MODEL [Epoch {epoch}]:")
        gen_val_loss = eval_single_epoch(
                            finetuned_discHE,
                            finetuned_genHE, gen_IHC, 
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
                save_checkpoint(epoch, gen_HE, optim_gen, filename=config.CHECKPOINT_GEN_HE, log_dir=log_dir, loss=best_val_loss)
                save_checkpoint(epoch, disc_HE, optim_disc, filename=config.CHECKPOINT_DISC_HE, log_dir=log_dir, loss=best_val_loss)
        else:
            epochs_no_improve += 1
            print(f"Best epoch so far was {epoch}\n")

        # Check for early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            print(f"Last model saved was on epoch {best_epoch} with loss: {best_val_loss}")
            break

    writer.close()


if __name__ == "__main__":
    main()