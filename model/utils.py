import torch
import config
import os
import random
import numpy as np

def save_checkpoint(epoch, model, optimizer, filename, log_dir=None):
    print(f"=> Saving checkpoint for Epoch: {epoch}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "log_dir": log_dir,
    }
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    try:
        print(f"=> Loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return checkpoint["epoch"] + 1 , checkpoint.get("log_dir", None)    # Return next epoch to start from
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

