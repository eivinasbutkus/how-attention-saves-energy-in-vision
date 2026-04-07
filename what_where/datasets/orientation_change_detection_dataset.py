import torch
import math
import numpy as np

from what_where.datasets.gratings_utils import oriented_sine_grating, gaussian_aperture, apply_random_translation


"""
Dataset replicating classical attention effects:
- increased firing rates
- decreased fano-factor & noise correlationa

The task is to detect a change in the orientation of the grating.
Input: image with two gratings with a change on 2-4th frame
Output: 3-way softmax ("saccade left, center, right")
The model is supposed to output center, unless there is a change on that frame.

Based on:
Cohen, M. R., & Maunsell, J. H. (2009). Attention improves performance primarily
by reducing interneuronal correlations. Nature neuroscience, 12(12), 1594-1600.
"""


class OrientationChangeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train=True, transform=None):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.n = cfg.dataset.n

        self.size = (cfg.dataset.large_img_size, cfg.dataset.large_img_size)
        self.cycles = cfg.dataset.cycles
        self.radius = cfg.dataset.radius
        self.frequency = self.cycles / (self.radius * 2)

        self.empty_frame = torch.zeros((1, *self.size)) + 0.5 # gray empty image

        self.center = (self.size[0] // 2, self.size[1] // 2)
        self.center_left = cfg.dataset.center_left
        self.center_right = cfg.dataset.center_right

        # to keep track of the schedule for the attend valid cue probability
        self.attend_valid_prob = cfg.dataset.attend_valid_prob_init
        self.day_orientations = None

    def add_attention_cue(self, frame, attend_left):
        color = 1.0 if attend_left else 0.0  # white or black dot
        x0, x1 = self.center[0] - 1, self.center[0] + 2
        y0, y1 = self.center[1] - 1, self.center[1] + 2
        frame[0, x0:x1, y0:y1] = color
        return frame

    def set_day_orientations(self, day_orientations):
        """
        This is to replicate the Cohen & Maunsell study design.
        - fixing the orientations of the gabors (pre-change) for the experimental "day"
        - orientations_days has shape (n_days, 2) for left-right gabor
        """
        self.day_orientations = torch.tensor(day_orientations, dtype=torch.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        if self.day_orientations is None:
            # sample orientation
            day_idx = -1 # indicating that we don't have orientation_days
            orientations = torch.rand(2) * math.pi
        else:
            day_idx = torch.randint(0, len(self.day_orientations), (1,)).item()
            orientations = self.day_orientations[day_idx]

        # independently sample presence of grating on both sides and contrasts
        attend_left = torch.rand(1).item() > 0.5  # attend left or right
        attend_idx = 0 if attend_left else 1

        # attention valid
        attend_valid = torch.rand(1).item() < self.attend_valid_prob

        # change frame and side
        change_t = torch.randint(1, 4, (1,)).item()  # either 1,2,3
        change_side = attend_idx if attend_valid else 1 - attend_idx

        presences = torch.ones(2) # both sides are always present
        contrasts = torch.ones(2) # 100% contrast for the orientation change detection task

        # input frames
        gratings_frame = self.get_gratings_frame(contrasts, presences, orientations)

        cue_frame = gratings_frame.clone()
        cue_frame = self.add_attention_cue(cue_frame, attend_left)

        min_change, max_change = self.cfg.dataset.orientation_change_range
        change_degrees = np.random.uniform(min_change, max_change) * np.random.choice([-1, 1])
        change_radians = math.radians(change_degrees)

        orientations_changed = orientations.clone()
        orientations_changed[change_side] += change_radians
        change_frame = self.get_gratings_frame(contrasts, presences, orientations_changed)

        frames = torch.stack([
            cue_frame,
            *[change_frame if t >= change_t else gratings_frame for t in range(1, 4)]
        ])


        # target output (n_frames, 3)
        target = torch.ones(frames.shape[0]).long() # 1 means center
        target[change_t] = 0 if change_side == 0 else 2 # 0 means left, 2 means right

        meta_data = {
            'attend_left': attend_left,
            'attend_valid': attend_valid,
            'orientation_left': orientations[0],
            'orientation_right': orientations[1],
            'contrast_left': contrasts[0],
            'contrast_right': contrasts[1],
            'present_left': presences[0],
            'present_right': presences[1],
            'change_t': change_t,
            'change': change_degrees,
            'change_side': change_side,
            'day': day_idx
        }

        labels = {
            "what": target, # saccade target (3-way classification: left, center, right)
        }
        return frames, labels, meta_data


    def get_gratings_frame(self, contrasts, presences, orientations):
        frame = torch.zeros((1, *self.size), dtype=torch.float32)

        for i in range(len(presences)):
            if presences[i]:
                grating = oriented_sine_grating(size=self.size, frequency=self.frequency, orientation=orientations[i], contrast=contrasts[i])
                center = self.center_left if i == 0 else self.center_right
                mask = gaussian_aperture(size=self.size, radius=self.radius, center=center)
                frame[0] += grating * mask

        frame += 0.5 # setting the mean luminance to 0.5 (gray)
        return frame
