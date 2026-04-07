

import torch
from typing import Tuple, Dict
from torch import nn

import what_where as ww




def apply_noise(cfg, x, noise_anneal):
    noise_std = cfg.model.activity_noise * noise_anneal
    noise = torch.randn_like(x) * noise_std
    x = x + noise

    return x

def scale_gain(cfg, gain):
    return gain * (cfg.model.gain.max - cfg.model.gain.min) + cfg.model.gain.min

def sample_gain_suppression(cfg, batch_size, n_passes, stimulus_onsets, device, amount=None):
    bounds = (cfg.experiment.gain_suppression.min, cfg.experiment.gain_suppression.max)
    if amount is None:
        amounts = torch.rand(batch_size, device=device) * (bounds[1] - bounds[0]) + bounds[0]
    else:
        amounts = torch.full((batch_size,), amount, device=device)

    indices = torch.rand(batch_size, device=device) > 0.5
    
    gain_suppression = {}
    gain_suppression["conv1"] = torch.zeros(batch_size, n_passes, device=device)
    stimulus_onsets = stimulus_onsets.to(device)
    
    # Apply suppression to selected batches at their specific stimulus onset times
    selected_batches = torch.where(indices)[0]
    gain_suppression["conv1"][selected_batches, stimulus_onsets[selected_batches]] = amounts[selected_batches]
    
    return gain_suppression

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_parameters_by_layer(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Count parameters for each layer in the model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts per layer
    """
    from collections import defaultdict
    params_dict = defaultdict(lambda: {"total": 0, "trainable": 0})

    for name, param in model.named_parameters():
        # Get the module name by splitting at the last dot
        module_name = name.rsplit(".", 1)[0] if "." in name else name
        params_dict[module_name]["total"] += param.numel()
        if param.requires_grad:
            params_dict[module_name]["trainable"] += param.numel()

    return dict(params_dict)



def get_readout(cfg):
    readouts = {
        "tiny_imagenet": ww.model.TinyImageNetReadout,
        "vcs": ww.model.VCSReadout,
        "contrast_detection": ww.model.ContrastDetectionReadout,
        "orientation_change_detection": ww.model.OrientationChangeDetectionReadout,
    }
    if cfg.dataset.name not in readouts:
        raise ValueError(f"Readout for dataset {cfg.dataset.name} not implemented.")

    return readouts[cfg.dataset.name](cfg)
