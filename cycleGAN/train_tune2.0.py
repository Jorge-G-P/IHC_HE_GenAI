import torch
import config
import torch.nn as nn
import torch.optim as optim
from HE_IHC_dataset import GanDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from evaluate import evaluate_fid_scores, evaluate_fid_scores_2
from utils import load_checkpoint, save_checkpoint, set_seed, custom_collate
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def convert_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors(v) for v in obj]
    else:
        return obj

def train_func(D_HE, D_IHC, G_HE, G_IHC, optim_D, optim_G, G_scaler, D_scaler, cycle_loss, disc_loss, ident_loss, lambda_identity, lambda_cycle, loader, epoch, writer):
    
    # device = config.DEVICE
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # print(f"TRAIN FUNC - Using device: {device}")


    '''Set the generators and discriminators to training mode'''
    D_HE.train()
    D_IHC.train()
    G_HE.train()
    G_IHC.train()

    loop = tqdm(loader, leave=True)         #Loop generates a progress bar while iterating over dataset
    for idx, sample in enumerate(loop):
        ihc = sample['A'].to(config.DEVICE)
        he = sample['B'].to(config.DEVICE)

        # print(ihc.device)  # Should print 'cuda:0'
        # print(he.device)   # Should print 'cuda:0'

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

        with torch.cuda.amp.autocast():         # For mixed precision training
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
                + cycle_IHC_loss * lambda_cycle
                + cycle_HE_loss * lambda_cycle
                + identity_HE_loss * lambda_identity
                + identity_IHC_loss * lambda_identity
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
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/train/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/train/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")
        
        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

    print(f"\nTRAIN EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}, D_loss: {D_loss}\n")
    
    return G_loss, D_loss

def eval_single_epoch(D_HE, D_IHC, G_HE, G_IHC, cycle_loss, disc_loss, ident_loss, lambda_identity, lambda_cycle, loader, epoch, writer):

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
                    + cycle_IHC_loss * lambda_cycle
                    + cycle_HE_loss * lambda_cycle
                    + identity_HE_loss * lambda_identity
                    + identity_IHC_loss * lambda_identity
                )

                writer.add_scalars("[VAL] - Cycle Loss", {"HE": cycle_HE_loss, "IHC": cycle_IHC_loss}, epoch)
                writer.add_scalars("[VAL] - Identity Loss", {"HE": identity_HE_loss, "IHC" : identity_IHC_loss}, epoch)
                writer.add_scalars("[VAL] - Gen_Img Discriminator Loss", {"Fake_HE": D_gen_HE_loss, "Fake_IHC" : D_gen_IHC_loss}, epoch)
                writer.add_scalar("[VAL] - Total Generator Loss", G_loss, epoch)
        
        if epoch % 5 == 0 and idx % 210 == 0:
                for i in range(len(ihc)):   # (*0.5 + 0.5) before saving img to be on range [0, 1]
                    save_image(he[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}].png")
                    save_image(fake_HE[i]*0.5 + 0.5, config.parent_path / f"gan-img/HE/val/epoch[{epoch}]_batch[{idx}]_HE[{i}]_fake.png")
                    save_image(ihc[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}].png")
                    save_image(fake_IHC[i]*0.5 + 0.5, config.parent_path / f"gan-img/IHC/val/epoch[{epoch}]_batch[{idx}]_IHC[{i}]_fake.png")

    print(f"\nVALIDATION EPOCH: {epoch}/{config.NUM_EPOCHS}, batch: {idx+1}/{len(loader)}," + f" G_loss: {G_loss}\n")

    return G_loss

def main(h_params):
    set_seed(42) # To ensure reproducibility

    device = config.DEVICE
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"MAIN - Using device: {device}")

    num_residuals = h_params.get("num_residuals")
    lr_discriminator = h_params.get("lr_discriminator")
    lr_generator = h_params.get("lr_generator")
    batch_size = h_params.get("batch_size")
    # lambda_identity = h_params.get("lambda_identity")
    # lambda_cycle = h_params.get("lambda_cycle")
    # beta1 = h_params.get("beta1")
    # beta2 = h_params.get("beta2")
    log_dir = h_params.get("log_dir")

    lambda_identity = config.LAMBDA_IDENTITY
    lambda_cycle = config.LAMBDA_CYCLE

    disc_HE = Discriminator(in_channels=config.IN_CH, features=config.D_FEATURES).to(config.DEVICE) 
    disc_IHC = Discriminator(in_channels=config.IN_CH, features=config.D_FEATURES).to(config.DEVICE)

    gen_HE = Generator(img_channels=3, num_residuals=num_residuals).to(config.DEVICE)
    gen_IHC = Generator(img_channels=3, num_residuals=num_residuals).to(config.DEVICE)

    optim_disc = optim.Adam(
        list(disc_HE.parameters()) + list(disc_IHC.parameters()),
        lr=lr_discriminator,
        betas=(0.9, 0.999)
    )
    optim_gen = optim.Adam(
        list(gen_HE.parameters()) + list(gen_IHC.parameters()),
        lr=lr_generator,
        betas=(0.9, 0.999)
    )

    # Losses used during training
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    discrim_loss = nn.MSELoss()

    '''Initialize datasets and dataloaders:
        1) Create dataset for later split between training/validation
        2) Split between training and validation size
        3) Create train and validation sets and loaders
    '''
    train_dataset = GanDataset(
        config.TRAIN_DIR_IHC, 
        config.TRAIN_DIR_HE, 
        config.SUBSET_PERCENTAGE, 
        patch_size=512, 
        transform=config.transforms, 
        shuffle=False
    )
    val_dataset = GanDataset(
        config.TEST_DIR_IHC, 
        config.TEST_DIR_HE, 
        config.SUBSET_PERCENTAGE, 
        patch_size=512, 
        transform=config.transforms, 
        shuffle=False,
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=custom_collate,
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        collate_fn=custom_collate,
        drop_last=True
    )

    # Initialize gradient scalers for mixed precision training
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=log_dir)

    fid_he = None
    fid_ihc = None
    step = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            print(f"[DEBUGGING] - Checkpoint path is {checkpoint_dir}")
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), map_location=config.DEVICE)

        disc_HE.load_state_dict(checkpoint_dict["disc_HE_model"])
        disc_IHC.load_state_dict(checkpoint_dict["disc_IHC_model"])
        gen_HE.load_state_dict(checkpoint_dict["gen_HE_model"])
        gen_IHC.load_state_dict(checkpoint_dict["gen_IHC_model"])
        optim_disc.load_state_dict(checkpoint_dict["optim_disc"])
        optim_gen.load_state_dict(checkpoint_dict["optim_gen"])

        last_step = checkpoint_dict["step"]
        step = last_step + 1

        if "lr_discriminator" in h_params:
            for param_group in optim_disc.param_groups:
                param_group["lr"] = lr_discriminator
        if "lr_generator" in h_params:
            for param_group in optim_gen.param_groups:
                param_group["lr"] = lr_generator

    while True:
        print(f"TRAINING MODEL [Epoch {step}]:")
        gen_train_loss, disc_train_loss = train_func(
                                            disc_HE, disc_IHC, 
                                            gen_HE, gen_IHC, 
                                            optim_disc, optim_gen, 
                                            g_scaler, d_scaler, 
                                            cycle_loss, discrim_loss, identity_loss,
                                            lambda_identity, lambda_cycle,
                                            train_loader, 
                                            step, 
                                            writer
                                        )
        
        if step % config.FID_FREQUENCY == 0:
            print(f"CALCULATING FID SCORES [Epoch {step}]:")
            fid_he, fid_ihc = evaluate_fid_scores(gen_HE, gen_IHC, val_loader, config.DEVICE, config.FID_BATCH_SIZE)
            print(f"FID Scores - HE: {fid_he}, IHC: {fid_ihc}")
            writer.add_scalars("FID Scores", {"HE": fid_he, "IHC": fid_ihc}, step)

        print(f"VALIDATING MODEL [Epoch {step}]:")
        gen_val_loss = eval_single_epoch(
                            disc_HE, disc_IHC, 
                            gen_HE, gen_IHC, 
                            cycle_loss, discrim_loss, identity_loss,
                            lambda_identity, lambda_cycle,
                            val_loader, 
                            step, 
                            writer
                        )

        writer.add_scalars("Generators Losses", {"train": gen_train_loss, "val": gen_val_loss}, step)
        
        metrics = {
            "gen_val_loss": float(gen_val_loss),
            "train_loss_disc": float(disc_train_loss),
            "fid_he": float(fid_he) if fid_he is not None else None,
            "fid_ihc": float(fid_ihc) if fid_ihc is not None else None
        }

        converted_metrics = convert_tensors(metrics)

        if step % h_params["checkpoint_interval"] == 0:
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.save(
                    {
                        "disc_HE_model": disc_HE.state_dict(),
                        "gen_HE_model": gen_HE.state_dict(),
                        "disc_IHC_model": disc_IHC.state_dict(),
                        "gen_IHC_model": gen_IHC.state_dict(),
                        "step": step,
                    },
                    os.path.join(tmpdir, "checkpoint.pt"),   
                )
                train.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            train.report(converted_metrics)
        
        step += 1

    writer.close()

import matplotlib.pyplot as plt
import os
from pathlib import Path
import tempfile
from filelock import FileLock
import ray
from ray import train, tune
from ray.train import Checkpoint, FailureConfig, RunConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner


if __name__ == "__main__":

    # print(Path(os.path.dirname(os.path.realpath(__file__))).parent)

    perturbation_interval = 4

    hyperparameters = {
        "num_residuals": tune.choice([6, 9]),
        "lr_discriminator": tune.choice([1e-4, 1e-5]),
        "lr_generator": tune.choice([1e-4, 1e-5]),
        "batch_size": tune.choice([2, 4]),
        # "lambda_identity": tune.choice([0, 0.5, 1]),
        # "lambda_cycle": tune.choice([8, 10, 12]),
        # "beta1": tune.choice([0.5, 0.9]),
        # "beta2": tune.choice([0.999, 0.99]),
        "checkpoint_interval": perturbation_interval,
        "log_dir" : Path(os.path.dirname(os.path.realpath(__file__))).parent / f"logs/GAN_PBT_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    }

    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturbation_interval,
        metric="fid_he",
        mode="min",
        hyperparam_mutations={
            "num_residuals": tune.choice([6, 9]),
            "lr_discriminator": tune.uniform(1e-5, 1e-4),
            "lr_generator": tune.uniform(1e-5, 1e-4),
            "batch_size": tune.choice([2, 4]),
            # "lambda_identity": tune.choice([0, 0.5, 1]),
            # "lambda_cycle": tune.choice([8, 10, 12]),
            # "beta1": tune.choice([0.5, 0.9]),
            # "beta2": tune.choice([0.999, 0.99]),
        },
    )

    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    stopper = False # For testing purposes: set this to False to run the full experiment
    trainable_with_resources = tune.with_resources(main, {"cpu": 4, "gpu": 1})
    tuner = tune.Tuner(
        trainable_with_resources,
        run_config=train.RunConfig(
            name="pbt_cycleGAN",
            stop={"training_iteration": 10 if stopper else config.NUM_EPOCHS},
        ),
        tune_config=tune.TuneConfig(
            # metric = "fid_ihc",
            # mode="min",
            num_samples=2 if stopper else 4,
            scheduler = pbt_scheduler,
        ),
        param_space=hyperparameters,
    )

    results_grid = tuner.fit()
    result_dfs = [result.metrics_dataframe for result in results_grid]
    best_result = results_grid.get_best_result(metric="fid_he", mode="min")
    # best_result_he = results_grid.get_best_result(metric="fid_he", mode="min")
    # best_result_ihc = results_grid.get_best_result(metric="fid_ihc", mode="min")

    print("Best hyperparameters found were: ", tuner.best_config)



