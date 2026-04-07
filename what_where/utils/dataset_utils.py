

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import what_where as ww


def get_datasets(cfg):
    train_transform, valid_transform = get_transforms(cfg)

    if cfg.dataset.name == "tiny_imagenet":
        train_dataset = ww.datasets.TinyImageNet(ww.utils.ROOT_DIR / 'data', split='train', download=True, transform=train_transform, pre_load=False)
        valid_dataset = ww.datasets.TinyImageNet(ww.utils.ROOT_DIR / 'data', split='val', download=False, transform=valid_transform, pre_load=False)
    else:
        datasets = {
            "vcs": ww.datasets.VCSDataset,
            "contrast_detection": ww.datasets.ContrastDetectionDataset,
            "orientation_change_detection": ww.datasets.OrientationChangeDetectionDataset,
        }
        if cfg.dataset.name in datasets:
            train_dataset = datasets[cfg.dataset.name](cfg, train=True, transform=train_transform)
            valid_dataset = datasets[cfg.dataset.name](cfg, train=False, transform=valid_transform)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")
        
    datasets = {
        "train": train_dataset,
        "valid": valid_dataset
    }

    return datasets

def get_dataloaders(cfg, generator=None):
    datasets = get_datasets(cfg)
    dataloaders = {}
    dataloaders['train'] = DataLoader(datasets['train'], generator=generator, **dict(cfg.train.dataloader))
    dataloaders['valid'] = DataLoader(datasets['valid'], generator=generator, **dict(cfg.train.dataloader))
    return dataloaders




def get_transforms(cfg):
    if cfg.dataset.name == "tiny_imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Grayscale(), # transform to grayscale
            transforms.Normalize(mean=[0.48], std=[0.23]),
            transforms.RandomErasing(value=0.0, p=0.5)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(), # transform to grayscale
            transforms.Normalize(mean=[0.48], std=[0.23]),
        ])

    else:
        if cfg.dataset.name == "vcs":
            normalize = transforms.Normalize(mean=[0.1819]*3, std=[0.1756]*3)
        elif cfg.dataset.name == "contrast_detection" or cfg.dataset.name == "orientation_change_detection":
            normalize = transforms.Normalize(mean=[0.5]*3, std=[0.01]*3)
        else:
            raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

        train_transform = transforms.Compose([
            lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x,
            normalize,
            transforms.Grayscale(),
        ])
        valid_transform = train_transform # use the same transform for validation

    return train_transform, valid_transform
