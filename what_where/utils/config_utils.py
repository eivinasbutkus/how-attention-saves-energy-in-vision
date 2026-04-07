
from omegaconf import OmegaConf
from hydra import initialize, compose


def get_config_entry(cfg, key_string):
    keys = key_string.split('.')
    value = cfg
    for key in keys:
        value = getattr(value, key, None)
        if value is None:
            return ''
    return value




def load_config(config_name, cfg_dict: dict = None):
    if cfg_dict is None:
        # initialize takes only relative paths
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name=config_name)
    else:
        cfg = OmegaConf.create(cfg_dict)
    return cfg


