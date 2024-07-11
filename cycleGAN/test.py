import torch
import config
import torch.nn as nn
import torch.optim as optim
from HE_IHC_dataset import GanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from evaluate import evaluate_fid_scores_2
from utils import load_checkpoint, set_seed, custom_collate
from torchvision.utils import save_image

set_seed(42) # To ensure reproducibility

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

cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()
discrim_loss = nn.MSELoss()

test_dataset = GanDataset(
    config.TEST_DIR_IHC, 
    config.TEST_DIR_HE, 
    config.SUBSET_PERCENTAGE, 
    patch_size=512, 
    transform=config.test_transforms, 
    shuffle=False
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=config.BATCH_SIZE, 
    pin_memory=True, 
    shuffle=False, 
    num_workers=config.NUM_WORKERS, 
    collate_fn=custom_collate,
    drop_last=False
)

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

train_epoch = load_checkpoint(config.CHECKPOINT_GEN_HE, gen_HE, optim_gen, config.LEARNING_RATE)[0]
train_epoch = load_checkpoint(config.CHECKPOINT_GEN_IHC, gen_IHC, optim_gen, config.LEARNING_RATE)[0]
train_epoch = load_checkpoint(config.CHECKPOINT_DISC_HE, disc_HE, optim_disc, config.LEARNING_RATE)[0]
train_epoch = load_checkpoint(config.CHECKPOINT_DISC_IHC, disc_IHC, optim_disc, config.LEARNING_RATE)[0]
print(f"\nModel loaded for test set was trained during {train_epoch-1} epochs")

def test_performance(D_HE, D_IHC, G_HE, G_IHC, cycle_loss, disc_loss, ident_loss, loader):
    G_HE.eval()
    G_IHC.eval()
    D_HE.eval()
    D_IHC.eval()

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

            if idx % 20 == 0:
                for i in range(len(ihc)): 
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/test/batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/test/batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/test/batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/test/batch[{idx}]_IHC[{i}]_fake.png")
    
    print(f"\nCALCULATING FID SCORES:")
    fid_he, fid_ihc = evaluate_fid_scores_2(gen_HE, gen_IHC, test_loader, config.DEVICE, config.FID_BATCH_SIZE)

    return fid_he, fid_ihc, G_loss

if __name__=="__main__":
    
    print(f"TESTING MODEL:")
    fid_he, fid_ihc, gen_loss = test_performance(disc_HE, disc_IHC, gen_HE, gen_IHC, cycle_loss, discrim_loss, identity_loss, test_loader)
    print(f"\nTEST EPOCH:\nFID SCORE [HE]: {fid_he}\nFID SCORE [IHC]: {fid_ihc}\nGenerator Loss: {gen_loss}\n")