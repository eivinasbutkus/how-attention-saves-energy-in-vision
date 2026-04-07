
import torch
from torchvision import transforms
from torchvision.datasets import EMNIST

from .mnist_distractors_dataset import MNISTDistractorsDataset
from what_where.utils import ROOT_DIR

from PIL import Image

class VCSDataset(MNISTDistractorsDataset):

    def __init__(self, cfg, train=True, transform=None, target_transform=None, distractor_transform=None):
        super().__init__(cfg, train=train, transform=transform, target_transform=target_transform, distractor_transform=distractor_transform)

        # loading extended MNIST
        root = ROOT_DIR / "data"
        self.emnist = EMNIST(root, train=train, download=True, split="letters")

        # exclude letters that look like digits
        self.excluded_letters = ["i", "l", "o", "q", "s", "b", "z", "g"]

    def sample_distractor(self):

        # sampling a non excluded letter
        sampled = False
        while not sampled:
            i = torch.randint(0, len(self.emnist), (1,)).item()
            distractor_letter = chr(self.emnist[i][1] + 96)
            if distractor_letter not in self.excluded_letters:
                sampled = True
                distractor_img = self.emnist[i][0]

        # performing transformation
        distractor_img = distractor_img.rotate(270).transpose(Image.FLIP_LEFT_RIGHT) # EMNIST is rotated and flipped for some reason
        distractor_img = self.distractor_transform(distractor_img)

        distractor_meta_info = {"what" : self.emnist[i][1],
                                "what_letter" : distractor_letter}

        return distractor_img, distractor_meta_info