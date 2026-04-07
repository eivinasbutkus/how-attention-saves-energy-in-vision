
import torch
from torch.utils.data import Subset

import cv2
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

from what_where import CONFIG_DIR, ROOT_DIR
from what_where.utils import get_dataset

import tqdm
import numpy as np


@hydra.main(version_base=None, config_name='config', config_path=str(CONFIG_DIR))
def generate_behavioral_datasets(cfg: DictConfig):
    """
    Behavioral datasets are used to test humans and models.
    """
    torch.manual_seed(42)

    # we want meta info and mask images
    cfg.dataset.meta_info = True
    cfg.dataset.mask_img = True

    # loading the test set of the dataset
    dataset = get_dataset(cfg, train=False)

    # shuffle dataset
    dataset = Subset(dataset, torch.randperm(len(dataset)))

    n = cfg.dataset_generation.n_behavioral_images

    # take subset of the first n images
    subset = Subset(dataset, range(0, n))

    # save the experiment dataset (pytorch)
    dataset_path = ROOT_DIR / "data" / "behavioral_datasets" / cfg.dataset_generation.behavioral_dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, dataset_path / f"behavioral_dataset.yaml")

    # save images separately to load in unity (png)
    images_path = dataset_path / "images"
    mask_images_path = dataset_path / "mask_images"
    meta_info_path = dataset_path / "meta_info"

    images_path.mkdir(parents=True, exist_ok=True)
    mask_images_path.mkdir(parents=True, exist_ok=True)
    meta_info_path.mkdir(parents=True, exist_ok=True)

    # save the experiment dataset (csv)
    whats = []
    where_x = []
    where_y = []
    noise = []
    n_distractors = []

    # for i, (image, (what, where), meta_info) in enumerate(subset):
    for i in tqdm.tqdm(range(len(subset))):
        image, (what, where), meta_info = subset[i]

        # saving the image
        image = image.squeeze().numpy()
        image = (image * 255).astype('uint8')
        cv2.imwrite(str(images_path / f"image_{i}.png"), image)

        mask_image = meta_info["mask_img"]["img"].squeeze()
        mask_image = (mask_image * 255).astype('uint8')
        cv2.imwrite(str(mask_images_path / f"mask_image_{i}.png"), mask_image)

        # saving the trial data (what, where, noise, n_distractors)
        whats.append(meta_info["search_img"]["what"])
        where_x.append(meta_info["search_img"]["target_x"])
        where_y.append(meta_info["search_img"]["target_y"])
        noise.append(meta_info["search_img"]["noise"])
        n_distractors.append(meta_info["search_img"]["n_distractors"])

        # write meta_info to a file for this particular image
        np.save(meta_info_path /  f"meta_info_{i}.npy", meta_info)


    csv_path = dataset_path / f"behavioral_dataset.csv"
    df = pd.DataFrame({"trial" : range(0, n),
                       "what": whats, "where_x": where_x, "where_y": where_y, "noise": noise, "n_distractors": n_distractors})
    df.to_csv(csv_path, index=False)



if __name__ == '__main__':
    generate_behavioral_datasets()
