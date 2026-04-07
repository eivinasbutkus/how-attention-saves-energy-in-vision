
import torch
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from .config_utils import get_config_entry
from .paths import ROOT_DIR



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch_generator = torch.Generator().manual_seed(seed)
    return torch_generator


def get_random_state(g):
    random_state = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'torch_cuda_all': [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate(),
        'dataloader_generator': g.get_state()  # Save DataLoader generator state
    }
    return random_state


def restore_random_state(random_state, generator):
    torch.set_rng_state(random_state['torch'])

    if torch.cuda.is_available() and random_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(random_state['torch_cuda'])
    
    if torch.cuda.is_available() and random_state['torch_cuda_all'] is not None:
        for i, state_i in enumerate(random_state['torch_cuda_all']):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(state_i, i)
    
    np.random.set_state(random_state['numpy'])
    random.setstate(random_state['random'])

    generator.set_state(random_state['dataloader_generator'])


def checkpoint_elements_to_str(cfg: DictConfig):
    s = ""
    for key_string in cfg.checkpoint_elements:
        s += "-" + key_string + "=" + str(get_config_entry(cfg, key_string))
    return s

def get_checkpoint_dir(cfg: DictConfig):
    model = cfg.model.name
    model += checkpoint_elements_to_str(cfg)

    final_dir = "instance" + str(cfg.train.instance)
    checkpoint_dir = ROOT_DIR / 'checkpoints' / cfg.dataset.name / cfg.experiment.name / model / final_dir

    return checkpoint_dir


def get_checkpoint_path_from_dir(checkpoints_dir: Path, best: bool = False, printing: bool =True):
    glob_str = 'checkpoint_*.pth' if not best else 'best_checkpoint_*.pth'

    checkpoints = list(checkpoints_dir.glob(glob_str))
    if len(checkpoints) == 0:
        return None
    elif len(checkpoints) > 1 and printing:
        print("WARNING: More than one checkpoint found. Taking the last one.")

    checkpoints.sort()
    checkpoint  = checkpoints[-1] # there should be only one best, but if not, then take the last
    if printing:
        print("checkpoint: ", checkpoint, "\n")
    return checkpoint

"""
returns latest checkpoint path (or best according to validation loss)
"""
def get_checkpoint_path(cfg: DictConfig, best: bool = False):
    checkpoint_dir = get_checkpoint_dir(cfg)
    print("checkpoint dir: ", checkpoint_dir)
    path = get_checkpoint_path_from_dir(checkpoint_dir, best=best)
    return path


def save_checkpoint(cfg, epoch, model, optimizer, lr_scheduler, torch_generator, checkpoint_dir):
    checkpoint = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'epoch': epoch,

        'model_state_dict': model.state_dict(), # model weights
        
        # training components
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        
        'random_state': get_random_state(torch_generator),
    }

    torch.save(checkpoint, checkpoint_dir / "checkpoint_{:03d}.pth".format(epoch))

def load_cnn_weights(model, checkpoint_path):
    """
    Load weights specifically for the CNN component of the model.

    Args:
        model: The model containing self.cnn
        checkpoint_path: Path to the checkpoint file
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Handle different checkpoint formats
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint

        # Filter for only CNN-related weights
        cnn_state_dict = {}
        for key, value in state_dict.items():
            # If key starts with 'cnn.' directly
            if key.startswith('cnn.'):
                new_key = key[4:]  # remove 'cnn.' prefix
                cnn_state_dict[new_key] = value

        # Load the weights into the CNN component
        model.cnn.load_state_dict(cnn_state_dict, strict=False)
        print("Successfully loaded CNN weights")

        # Print what was loaded
        print(f"Loaded {len(cnn_state_dict)} CNN layers:")
        for key in cnn_state_dict.keys():
            print(f"  - {key}")



    except Exception as e:
        print(f"Error loading CNN weights: {str(e)}")
        print("\nDebug information:")
        print(f"Available keys in checkpoint: {state_dict.keys()}")
        print(f"Model CNN state dict keys: {model.cnn.state_dict().keys()}")
        raise


def prep_checkpoints(cfg, model, optimizer, lr_scheduler, generator):
    if cfg.model.cnn.pre_training.train_weights:
        checkpoint_dir = ROOT_DIR / "checkpoints" / cfg.model.cnn.pre_training.checkpoint_dir
    else:
        checkpoint_dir = get_checkpoint_dir(cfg)

    start_epoch = 1

    # latest_checkpoint = get_checkpoint_path(cfg)
    checkpoint_path = get_checkpoint_path_from_dir(checkpoint_dir, best=False)

    if checkpoint_path is not None:
        print("Loading checkpoint: ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # restoring model state, training components, and random state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        restore_random_state(checkpoint['random_state'], generator)

        start_epoch = checkpoint['epoch'] + 1

    else: # this is a new training run and we need to load the pretrained cnn weights
        if cfg.model.cnn.pre_training.load_weights:
            cnn_checkpoint_dir = ROOT_DIR / "checkpoints" / cfg.model.cnn.pre_training.checkpoint_dir

            print("loading cnn checkpoint... from ", cnn_checkpoint_dir)
            cnn_checkpoint_path = get_checkpoint_path_from_dir(cnn_checkpoint_dir, best=False)
            if cnn_checkpoint_path is None:
                raise FileNotFoundError(f"No pretrained checkpoint found in {cnn_checkpoint_dir}, specify a valid checkpoint directory using cfg.model.cnn.pre_training.checkpoint_dir")
            load_cnn_weights(model, cnn_checkpoint_path)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, start_epoch