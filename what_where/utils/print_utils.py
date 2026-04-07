
from omegaconf import DictConfig, OmegaConf

import what_where as ww

def pretty_print_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


def print_start_epoch_info(cfg, epoch, optimizer):
    print("\nEpoch: ", epoch)
    print("learning rate: ", optimizer.param_groups[0]['lr'])
    print("noise level: ", cfg.model.activity_noise * ww.utils.get_noise_anneal(cfg, epoch, True))
    print("energy anneal: ", ww.utils.get_energy_anneal(cfg, epoch, train=True))




