import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import torch.optim as optim

import what_where as ww


def train_epoch(cfg, epoch, model, dataloader, device, optimizer, split, writer):
    n_passes = cfg.model.n_passes
    stats_epoch = ww.utils.init_stats_epoch(cfg)

    if split == "train":
        model.train()
    else:
        model.eval()

    noise_anneal = ww.utils.get_noise_anneal(cfg, epoch, split=="train")
    print(noise_anneal)
    energy_anneal = ww.utils.get_energy_anneal(cfg, epoch, split=="train")

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} ({split})", unit="batch")
    for batch_idx, data in enumerate(progress_bar):

        # imgs, labels (dict), meta_info = data
        imgs, labels, _ = data
        imgs = imgs.to(device)

        # sampling an energy cost (in fixed experiments, the range only includes one value)
        log_energy_cost = ww.utils.sample_log_energy_cost(cfg, imgs.size(0)).to(device)

        # forward pass through the model
        if split == "train":
            optimizer.zero_grad()
        out = model(imgs, log_energy_cost, n_passes, noise_anneal)

        loss_task, loss_energy = ww.utils.get_losses(cfg, out, labels, device, energy_anneal, log_energy_cost)
        loss = loss_task["what"] + loss_task["where"] + loss_energy["ap"] + loss_energy["st"] + loss_energy["norm"] + loss_energy["gain"]

        if split == "train":
            loss.backward()
            optimizer.step()

        accuracy = ww.utils.get_accuracy(cfg, out, labels, device)

        stats_epoch = ww.utils.update_stats_epoch(cfg, stats_epoch, loss, loss_task, loss_energy, accuracy, out, n_passes, batch_idx)

        # setting tqdm progress bar postfix
        progress_bar.set_postfix(ww.utils.get_progress_bar_postfix(stats_epoch, batch_idx))

    ww.utils.log_epoch_stats(cfg, split, stats_epoch, len(dataloader), epoch, writer)


@hydra.main(version_base=None, config_name='config_vcs_flexible', config_path=str(ww.utils.CONFIG_DIR))
def train(cfg: DictConfig):
    ww.utils.pretty_print_cfg(cfg)
    torch_generator = ww.utils.set_random_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ww.model.Model(cfg).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), # including only non-frozen params in the optimizer
                            **dict(cfg.train.adam))
    lr_scheduler = ww.utils.get_lr_scheduler(cfg, optimizer)

    checkpoint_dir, start_epoch = ww.utils.prep_checkpoints(cfg, model, optimizer, lr_scheduler, generator=torch_generator)
    print('start_epoch: ', start_epoch)

    dataloaders = ww.utils.get_dataloaders(cfg, generator=torch_generator)

    writer = SummaryWriter(checkpoint_dir / 'tensorboard')

    # Training loop
    for epoch in range(start_epoch, cfg.train.epochs+1):  # loop over the dataset multiple times
        ww.utils.print_start_epoch_info(cfg, epoch, optimizer)
        # ww.utils.log_normalization(cfg, model, epoch, writer)

        if cfg.dataset.name == "orientation_change_detection":
            ww.utils.update_orientation_change_detection_epoch(cfg, dataloaders, model, epoch)


        if (checkpoint_dir / "checkpoint_{:03d}.pth".format(epoch)).exists():
            print("Checkpoint already exists for epoch {}. Skipping.".format(epoch))
            continue

        for split in ["train", "valid"]:
            if split == "valid" and cfg.train.skip_validation:
                print("Skipping validation as per configuration.")
                continue

            train_epoch(cfg, epoch, model, dataloaders[split], device, optimizer, split, writer)


        if lr_scheduler is not None:
            lr_scheduler.step()

        # saving the checkpoint
        if epoch % cfg.train.checkpoint_save_interval == 0 or epoch == 1 or epoch == cfg.train.epochs:
            ww.utils.save_checkpoint(cfg, epoch, model, optimizer, lr_scheduler, torch_generator, checkpoint_dir)

    writer.close()


if __name__ == "__main__":
    train()
