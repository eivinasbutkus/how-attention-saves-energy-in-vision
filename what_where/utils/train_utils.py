

import torch
from torch import nn


from .energy_utils import get_energy_use


def get_task_loss_multiplier(cfg, t):
    # computing the loss multiplier based on the number of passes

    n_passes = cfg.model.n_passes

    if n_passes == 1: # only one pass
        return 1.0 # just put the full weight
    elif t == n_passes - 1: # last pass
        return cfg.train.loss.last_pass
    else: # all other passes
        return (1 - cfg.train.loss.last_pass) / (n_passes - 1)



def get_losses(cfg, out, labels, device, energy_anneal, log_energy_cost):

    n_passes = cfg.model.n_passes
    loss_task = {
        "what" : 0.0,
        "where" : torch.tensor(0.0).to(device)
    }

    loss_energy = {"ap" : 0.0, "st" : 0.0, "norm" : 0.0, "gain" : 0.0}

    for t in range(n_passes):
        # task loss
        task_loss_multiplier = get_task_loss_multiplier(cfg, t) # in pretrain and vcs, we encourage the last pass to be more accurate

        # what (used in all tasks)
        pred_what = out[t]["prediction"]["what"]

        what = labels["what"].to(device)
        if cfg.dataset.name == "orientation_change_detection":
            what_t = what[:,t] # target changes based on the time step
        else:
            what_t = what

        loss_task["what"] += task_loss_multiplier * cfg.train.loss.weights.what * nn.NLLLoss()(pred_what, what_t)

        # where (used in vcs)
        if "where" in labels:
            pred_where = out[t]["prediction"]["where"]
            where = labels["where"].flatten(start_dim=1).to(device)
            loss_task["where"] += task_loss_multiplier * cfg.train.loss.weights.where * nn.KLDivLoss(reduction="batchmean")(pred_where, where)

        energy_use = get_energy_use(cfg, out, t, device)
        for key in energy_use:
            # (annealing, multiplying with the flexible cost, taking the mean, accounting for number of passes)
            loss_energy[key] += 1 / n_passes * (energy_use[key] * energy_anneal * torch.exp(log_energy_cost).squeeze()).mean()

    return loss_task, loss_energy


def get_accuracy(cfg, out, labels, device):
    n_passes = cfg.model.n_passes
    what = labels["what"].to(device)

    if cfg.dataset.name == "orientation_change_detection":
        # prediction needs to agree on each time step to be considered correct
        pred_what = torch.stack([torch.argmax(out[t]["prediction"]["what"], dim=1) for t in range(n_passes)], dim=1) # (batch_size, n_passes)
        accuracy = torch.sum(torch.all(pred_what == what, dim=1)).item() / what.size(0)
    else:
        # only considering the last pass to calculate accuracy for all other tasks
        accuracy = torch.sum(torch.argmax(out[n_passes-1]["prediction"]["what"], dim=1) == what).item() / what.size(0)
    return accuracy


# noise anneal
def get_noise_anneal(cfg, epoch, train=True):
    if train and cfg.train.noise_anneal.enabled:
        warmup_epochs = cfg.train.noise_anneal.warmup_epochs
        anneal_epochs = cfg.train.noise_anneal.anneal_epochs

        if epoch < warmup_epochs: # warmup period
            return 0.0
        elif epoch > warmup_epochs + anneal_epochs: # post annealing period
            return 1.0
        else: # annealing period
            return (epoch - warmup_epochs) / anneal_epochs

    else:
        return 1.0
    


def get_lr_scheduler(cfg, optimizer):
    if cfg.train.lr_scheduler.enabled:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_scheduler.step_size, gamma=cfg.train.lr_scheduler.gamma)
    else:
        return None
    


def update_orientation_change_detection_epoch(cfg, dataloaders, model, epoch):
    if epoch >= cfg.dataset.attend_valid_prob_final_epoch:
        # updating the epoch for the attend_valid_prob scheduler
        dataloaders["train"].dataset.attend_valid_prob = cfg.dataset.attend_valid_prob_final
        dataloaders["valid"].dataset.attend_valid_prob = cfg.dataset.attend_valid_prob_final
