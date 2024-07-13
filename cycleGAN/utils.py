import torch
import config
import os
import random
import numpy as np

def save_checkpoint(epoch, model, optimizer, filename, log_dir=None, loss=None, fid_he=None, fid_ihc=None):
    print(f"=> Saving checkpoint for Epoch: {epoch}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "log_dir": log_dir,
        "loss": loss,
        "fid_he": fid_he,
        "fid_ihc": fid_ihc
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
        return checkpoint["epoch"] + 1 , checkpoint.get("log_dir", None), checkpoint.get("loss", None), checkpoint.get("fid_he", None), checkpoint.get("fid_ihc", None)
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
