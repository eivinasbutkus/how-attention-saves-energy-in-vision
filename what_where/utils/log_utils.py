

from collections import OrderedDict


def init_stats_epoch(cfg):
    stats_epoch = {
        "loss": 0.0,
        "loss_what": 0.0,
        "loss_where": 0.0,

        "accuracy" : 0.0, # always what component

        "loss_energy_ap": 0.0,
        "loss_energy_st": 0.0,
        "loss_energy_norm": 0.0,
        "loss_energy_gain": 0.0,

        "st:ap": 0.0,
    }

    return stats_epoch


def update_stats_epoch(cfg, stats_epoch, loss, loss_task, loss_energy, accuracy, out, n_passes, batch_idx):
    stats_epoch["loss"] += loss.item()
    stats_epoch["loss_what"] += loss_task["what"].item()
    stats_epoch["loss_where"] += loss_task["where"].item()

    stats_epoch["accuracy"] += accuracy * 100 # in percentage
    stats_epoch["loss_energy_ap"] += loss_energy["ap"].item()
    stats_epoch["loss_energy_st"] += loss_energy["st"].item()
    stats_epoch["loss_energy_norm"] += loss_energy["norm"].item()
    stats_epoch["loss_energy_gain"] += loss_energy["gain"].item()
    stats_epoch["st:ap"] += (loss_energy["st"] / loss_energy["ap"]).item()


    return stats_epoch

def log_epoch_stats(cfg, split, stats_epoch, n, epoch, writer):
    # average the stats over the epoch
    for key in stats_epoch:
        stats_epoch[key] /= n

    for key in stats_epoch:
        writer.add_scalar(f'{split}/{key}', stats_epoch[key], epoch)

def get_progress_bar_postfix(stats_epoch, batch_idx):
    postfix = OrderedDict()

    postfix["l"] = f"{stats_epoch['loss'] / (batch_idx + 1):.3f}"
    postfix["l_what"] = f"{stats_epoch['loss_what'] / (batch_idx + 1):.3f}"
    postfix["l_where"] = f"{stats_epoch['loss_where'] / (batch_idx + 1):.3f}"
    postfix["acc"] = f"{stats_epoch['accuracy'] / (batch_idx + 1):.2f}%"
    postfix["e_ap"] = f"{stats_epoch['loss_energy_ap'] / (batch_idx + 1):.3f}"
    postfix["e_st"] = f"{stats_epoch['loss_energy_st'] / (batch_idx + 1):.3f}"
    postfix["e_norm"] = f"{stats_epoch['loss_energy_norm'] / (batch_idx + 1):.3f}"
    postfix["e_gain"] = f"{stats_epoch['loss_energy_gain'] / (batch_idx + 1):.3f}"
    postfix["st:ap"] = f"{stats_epoch['st:ap'] / (batch_idx + 1):.3f}"

    return postfix
