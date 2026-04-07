import torch
from torchvision.datasets import MNIST
from torchvision import transforms

from abc import ABC, abstractmethod
import h5py

from what_where.utils import ROOT_DIR


def create_where_heatmap(y, x, width, height, sigma):
    """Create a heatmap with a 2D Gaussian centered at (x, y)."""
    xx, yy = torch.meshgrid(torch.arange(width),
                            torch.arange(height), indexing="ij")
    heatmap = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap / heatmap.sum()


class MNISTDistractorsDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, cfg, train=True,
                 transform=None,
                 target_transform=None, distractor_transform=None):

        # load the mnist dataset
        root = ROOT_DIR / "data"
        self.mnist = MNIST(root, train=train, download=True)
        self.train = train

        self.n_classes = 10

        # save the config
        self.cfg = cfg.dataset

        self.transform = transform # applied to the image

        self.xy_min = self.cfg.target.size / 2
        self.xy_max = self.cfg.large_img_size - self.cfg.target.size / 2


        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = transforms.Compose(
                    [
                        transforms.Resize((self.cfg.target.size, self.cfg.target.size)),
                        transforms.ToTensor(),
                    ]
                )

        if distractor_transform is not None: 
            self.distractor_transform = distractor_transform
        else:
            self.distractor_transform = transforms.Compose(
                    [
                        transforms.Resize((self.cfg.distractors.size, self.cfg.distractors.size)),
                        transforms.ToTensor(),
                    ]
                )
        
        # disk
        if self.cfg.disk:
            self.disk_path = ROOT_DIR / "data" / "datasets" / self.cfg.name / "disk"

            # load h5py file
            self.h5py_file = h5py.File(self.disk_path / "dataset.h5", "r")


    def __len__(self):
        if self.train:
            return self.cfg.train_n
        else:
            return self.cfg.valid_n

    @abstractmethod
    def sample_distractor(self):
        """
        Sample n_distractors images from the mnist dataset.
        Should return a distractor_img and distractor metainfo (e.g. class in mnist,
        or latent code that was used to generate the distractor).
        """
        pass


    def _min_distance_satified(self, xy, locations):
        for location in locations:
            if torch.norm(xy - location) < self.cfg.min_distance:
                return False
        return True

    def sample_locations(self, n_locations):
        locations = []
        steps = 0

        # target can be centered for representational space understanding (to reduce the spatial variability)
        if self.cfg.target.centered:
            locations.append(torch.tensor([self.cfg.large_img_size / 2, self.cfg.large_img_size / 2]))
            n_locations -= 1

        for i in range(n_locations):
            xy = torch.rand(2) * (self.xy_max - self.xy_min) + self.xy_min
            while not self._min_distance_satified(xy, locations):
                xy = torch.rand(2) * (self.xy_max - self.xy_min) + self.xy_min
                steps += 1
            locations.append(xy)
        return locations


    def place_small_img(self, xy, image, large_image, where_heatmap=False):
        # sampling the point where the mnist image will be placed
        x, y = xy.unbind()

        # place the image in the larger image
        j = x.round().int() - image.size(2) // 2
        i = y.round().int() - image.size(1) // 2
        large_image[:, i:i+image.size(1), j:j+image.size(2)] += image

        if where_heatmap:
            heatmap_size = self.cfg.where.size
            sigma = self.cfg.where.sigma
            scale = self.cfg.large_img_size / self.cfg.where.size
            where_heatmap = create_where_heatmap(x/scale, y/scale, heatmap_size, heatmap_size, sigma)
            return where_heatmap
        else:
            return None


    def generate_image(self, idx, mask=False, mu=None, noise=None):
        """
        Generates a single image with target (with index idx in the mnist dataset) and distractors.
        If mask=True, the image is generated without the target (in this case the mu and noise parameters should be specified).
        """

        mnist_idx = idx % len(self.mnist) # looping over the mnist dataset
        target_img, what = self.mnist[mnist_idx] # load the target
        target_img = self.target_transform(target_img) # transform the target
        what = torch.tensor(what, dtype=torch.long)

        # Create the larger image
        if mu is None:
            mu = torch.sum(target_img) / target_img.numel() # mean pixel value
        large_img = torch.zeros((1, self.cfg.large_img_size, self.cfg.large_img_size))

        # creating a list of images (targets and distractors)
        images = []
        if not mask: # adding the target if not a mask image
            images.append(target_img)

        # sample number of distractors
        n_min = self.cfg.distractors.n_min if not mask else self.cfg.mask_distractors.n_min
        n_max = self.cfg.distractors.n_max if not mask else self.cfg.mask_distractors.n_max
        n_distractors = torch.randint(n_min, n_max + 1, (1,)).item()

        distractors_meta_info = [] 
        for i in range(n_distractors):
            distractor_img, distractor_meta_info = self.sample_distractor() # sample distractor images
            distractors_meta_info.append(distractor_meta_info)
            images.append(distractor_img)


        # sample locations
        locations = self.sample_locations(n_distractors + 1)
            
        # place the target and generate where heatmap/distribution
        where = self.place_small_img(locations[0], images[0], large_img, where_heatmap=True)

        # place the distractor images
        for i, (xy, image) in enumerate(zip(locations[1:], images[1:])):
            # place the distractor images
            self.place_small_img(xy, image, large_img, where_heatmap=False)

        # add noise to the whole image
        if noise is None:
            noise = self.cfg.noise.min + torch.rand(1) * (self.cfg.noise.max - self.cfg.noise.min)
        large_img[large_img < mu] = mu
        large_img += torch.randn_like(large_img) * noise
        large_img = torch.clamp(large_img, 0, 1)

        # change x with y
        target_x = locations[0][0] / self.cfg.large_img_size - 0.5
        target_y = 1 - locations[0][1] / self.cfg.large_img_size - 0.5

        out = {
            "large_img": large_img,
            "what": what,
            "where": where,
            "mu": mu,
            "noise": noise,
            "target_x": target_x,
            "target_y": target_y,
            "locations": locations,
            "distactors_meta_info": distractors_meta_info,
            "n_distractors": n_distractors,
        }

        return out


    def load_from_disk(self, idx):
        # load from h5py file
        search_img = torch.tensor(self.h5py_file["search_img"][idx])
        what = torch.tensor(self.h5py_file["what"][idx])
        where = torch.tensor(self.h5py_file["where"][idx])
        labels = {"what": what, "where": where}

        meta_info = {}

        if self.transform:
            search_img = self.transform(search_img)
        
        return search_img, labels, meta_info

    def __getitem__(self, idx):

        if self.cfg.disk:
            return self.load_from_disk(idx)
        
        search_out = self.generate_image(idx, mask=False)
        search_img = search_out["large_img"]
        if self.transform:
            search_img = self.transform(search_img)

        what = search_out["what"]
        where = search_out["where"]

        if self.cfg.mask_img:
            # generating mask image with the same mean value and noise
            mask_out = self.generate_image(idx, mask=True, mu=search_out["mu"], noise=search_out["noise"])
            mask_img = mask_out["large_img"]
            if self.transform:
                mask_img = self.transform(mask_img)
        else:
            mask_img = None


        if self.cfg.meta_info:
            meta_info = {
                "search_img" : {
                    "img" : search_img.cpu().numpy(),
                    "what": what.cpu().numpy(),
                    "where": where.cpu().numpy(),

                    "noise": search_out["noise"].item(),
                    "target_x": search_out["target_x"].item(),
                    "target_y": search_out["target_y"].item(),

                    "locations": torch.concat(search_out["locations"]).cpu().numpy(),
                    "n_distractors": search_out["n_distractors"],
                    "distactors_meta_info": search_out["distactors_meta_info"],
                }
            }

            if self.cfg.mask_img:
                meta_info["mask_img"] = {
                    "img" : mask_out["large_img"].cpu().numpy(),
                    "locations": torch.concat(mask_out["locations"]).cpu().numpy(),
                    "n_distractors": mask_out["n_distractors"],
                    "distactors_meta_info": mask_out["distactors_meta_info"],
                }
        else:
            meta_info = {}

        labels = {"what": what, "where": where} 
        return search_img, labels, meta_info

