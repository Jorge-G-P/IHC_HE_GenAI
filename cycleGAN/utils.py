import torch
import cycleGAN.config as config
import os
import random
import numpy as np

def save_checkpoint(epoch, model, optimizer, filename, log_dir=None, loss=None):
    print(f"=> Saving checkpoint for Epoch: {epoch}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "log_dir": log_dir,
        "loss": loss,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    try:
        print(f"=> Loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return checkpoint["epoch"] + 1 , checkpoint.get("log_dir", None), checkpoint.get("loss", None)
    except Exception as e:
        print(f"=> Failed to load checkpoint {checkpoint_file}: {str(e)}")
        raise


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def custom_collate(batch):
    # Initialize dictionaries to store batches for each key
    batch_dict = {key: [] for key in batch[0]}
    
    # Append each item to the corresponding list in the batch_dict
    for item in batch:
        for key in item:
            batch_dict[key].append(item[key])
    
    # Convert lists to tensors where applicable (A and B are tensors, others can be left as lists)
    batch_dict['A'] = torch.stack(batch_dict['A'])
    batch_dict['B'] = torch.stack(batch_dict['B'])
    
    return batch_dict

def create_directories():
    os.makedirs(config.parent_path / "gan-img/realHE_fakeIHC_gt", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/realHE_fakeIHC_cycle", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/realIHC_fakeHE_gt", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/realIHC_fakeHE_cycle", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/HE/train/", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/HE/val/", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/HE/test/", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/IHC/train/", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/IHC/val/", exist_ok=True)
    os.makedirs(config.parent_path / "gan-img/IHC/test/", exist_ok=True)
    os.makedirs(config.parent_path / "logs/", exist_ok=True)
