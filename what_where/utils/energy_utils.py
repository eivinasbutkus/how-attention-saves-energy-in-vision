
import torch
torch._dynamo.config.cache_size_limit = 32 # to allow many different sizes

import numpy as np


@torch.compile()
def compute_synaptic_transmission_linear(input_activations, linear_layer, sample_ratio=0.1):
    """
    computes energy use for synaptic transmission in a linear layer
    """

    spikes = input_activations # renaming (batch, n_spikes)
    weights = linear_layer.weight # extracting the weights from the linear layer (n_synapses, n_spikes)

    # sample a fraction of the spikes to reduce the computational cost
    n_total_spikes = spikes.shape[1]
    n_samples = max(int(n_total_spikes * sample_ratio), 1) # at least one sample (e.g. in the MLP with input_size=1)
    perm = torch.randperm(n_total_spikes, device=spikes.device)
    indices = perm[:n_samples]

    n_samples_weights = max(int(weights.shape[0] * sample_ratio), 1) # at least one sample
    perm_weights = torch.randperm(weights.shape[0], device=weights.device)
    indices_weights = perm_weights[:n_samples_weights]

    spikes_sampled = spikes[:, indices]
    weights_sampled = weights[:, indices][indices_weights]

    # postsynaptic potentials (absolute activity along each synapse)
    psps_sampled = spikes_sampled.unsqueeze(1) * weights_sampled.unsqueeze(0)
    psps_sampled.abs_() # in-place absolute value

    # summing across weights and spikes (leaving just the batch dimension)
    synaptic_transmission = torch.sum(psps_sampled, dim=(1,2)) * (n_total_spikes/n_samples) * (weights.shape[0]/n_samples_weights)

    return synaptic_transmission


@torch.compile()
def compute_synaptic_transmission_conv(input_activations, conv_layer, sample_ratio=0.1):
    """
    computes energy use for synaptic transmission in a convolutional layer
    """
    spikes = input_activations  # (B, C_in, H, W)
    conv_weights = conv_layer.weight  # (C_out, C_in, K_h, K_w)
    C_out, C_in, K_h, K_w = conv_weights.shape
    
    # Sample channels to reduce computational cost
    sampled_channels_in = max(int(C_in * sample_ratio), 1)
    perm_in = torch.randperm(C_in, device=spikes.device)
    sampled_in_channels = perm_in[:sampled_channels_in]

    sampled_channels_out = max(int(C_out * sample_ratio), 1)
    perm_out = torch.randperm(C_out, device=spikes.device)
    sampled_out_channels = perm_out[:sampled_channels_out]

    # Calculate PSPs with correct einsum pattern
    psps_sampled = torch.abs(torch.einsum('bcxy,dcij->bdcxyij', 
                                          spikes[:, sampled_in_channels], 
                                          conv_weights[sampled_out_channels][:, sampled_in_channels]))
    
    # Sum over all dimensions except batch
    synaptic_transmission_sampled = torch.sum(psps_sampled, dim=(1,2,3,4,5,6))
    
    # Scale for output channel sampling
    synaptic_transmission = synaptic_transmission_sampled * (C_out / sampled_channels_out) * (C_in / sampled_channels_in)
    
    return synaptic_transmission



def get_energy_use(cfg, out, t, device):
    """
    sums energy from the out dictionary to be used for backpropagation
    """

    # inferring batch size
    first_key = next(iter(out[t]["activations"]))
    batch_size = out[t]["activations"][first_key].shape[0]

    energy_use = {
        "ap" : torch.zeros(batch_size).to(device), # action potentials
        "st" : torch.zeros(batch_size).to(device), # synaptic transmission
        "norm" : torch.zeros(batch_size).to(device), # normalization
        "gain" : torch.zeros(batch_size).to(device), # gain
    }

    # action potentials
    for (layer_name, activations) in out[t]["activations"].items():
        dim = tuple(range(1, activations.dim())) # keep the batch dimension (works on tensors from different layers)
        energy_use["ap"] += activations.sum(dim=dim) # simply sum the activations

    # synaptic transmission
    for (layer_name, synaptic_transmission) in out[t]["synaptic_transmission"].items():
        energy_use["st"] += synaptic_transmission # should already be summed

    # normalization
    for (layer_name, normalization_energy) in out[t]["normalization"].items():
        energy_use["norm"] += normalization_energy # should already be summed


    for gain_type in ["what", "where", "when"]:
        if dict(cfg.model.gain)[gain_type]["active"]:
            if gain_type == "what":
                what_gain = out[t]["gain"][gain_type]
                for (layer_name, gain_pattern) in what_gain.items():
                    energy_use["gain"] += torch.mean(torch.abs(gain_pattern - 0.5), dim=1)/len(what_gain) # summing across channels

            elif gain_type == "where":
                gain_pattern = out[t]["gain"][gain_type]
                energy_use["gain"] += torch.mean(torch.abs(gain_pattern - 0.5), dim=(1,2)) # summing across spatial dimensions

            elif gain_type == "when":
                gain_pattern = out[t]["gain"][gain_type].squeeze(-1) # (batch_size)
                energy_use["gain"] += torch.abs(gain_pattern - 0.5)

    # scales the energy
    energy_use["ap"] *= cfg.train.energy.ap_scale # using 1.0, acts as a reference for the scale of other energy terms
    energy_use["st"] *= cfg.train.energy.st_scale # st : ap ratio is supposed to be ~3:1 according to updated energy budgets paper
    energy_use["norm"] *= cfg.train.energy.norm_scale # normalization energy scale is somewhat arbitrary, but conservatively scaled to be in the ballpark of the other energy terms
    energy_use["gain"] *= cfg.train.energy.gain_scale # gain

    return energy_use

    
def get_energy_anneal(cfg, epoch, train=True):
    if train and cfg.train.energy_anneal.enabled:
        warmup_epochs = cfg.train.energy_anneal.warmup_epochs
        anneal_epochs = cfg.train.energy_anneal.anneal_epochs

        if epoch < warmup_epochs: # warmup period
            return 0.0
        elif epoch > warmup_epochs + anneal_epochs: # post annealing period
            return 1.0
        else: # annealing period
            return (epoch - warmup_epochs) / anneal_epochs
    else:
        return 1.0
    


def sample_log_energy_cost(cfg, batch_size):
    min_cost = torch.tensor(cfg.train.energy.cost.min)
    max_cost = torch.tensor(cfg.train.energy.cost.max)
    log_energy_cost = torch.rand(batch_size, 1) * (max_cost - min_cost) + min_cost
    return log_energy_cost


