import torch
import math


from what_where.datasets.gratings_utils import oriented_sine_grating, circular_aperture_mask



"""
Gratings dataset to reproduce the experiment in:
https://www.science.org/doi/10.1126/science.ade1855

Four frames in each trial:
1. Cue: white or black dot in the middle indicating where to attend.
2. Stimulus
3. Stimulus
4. Clear display
"""


class ContrastDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, train=True, transform=None):
        super().__init__()
        self.cfg = cfg
        self.train = train
        self.n = cfg.dataset.n

        self.size = (cfg.dataset.large_img_size, cfg.dataset.large_img_size)
        self.cycles = cfg.dataset.cycles
        self.radius = cfg.dataset.radius
        self.frequency = self.cycles / (self.radius * 2)
        self.retinal_noise = cfg.dataset.retinal_noise

        self.empty_frame = torch.zeros((1, *self.size)) + 0.5 # gray empty image

        self.center = (self.size[0] // 2, self.size[1] // 2)
        self.center_left = cfg.dataset.center_left
        self.center_right = cfg.dataset.center_right
        self.center_random_shift = cfg.dataset.center_random_shift

    def add_attention_cue(self, frame, attend_left):
        color = 1.0 if attend_left else 0.0  # white or black dot
        x0, x1 = self.center[0] - 1, self.center[0] + 2
        y0, y1 = self.center[1] - 1, self.center[1] + 2
        frame[0, x0:x1, y0:y1] = color
        return frame

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # sample orientation
        orientations = torch.rand(2) * math.pi

        # independently sample presence of grating on both sides and contrasts
        attend_left = torch.rand(1).item() > 0.5  # attend left or right
        attend_idx = 0 if attend_left else 1
        presences = torch.rand(2) > 0.5 # true or false for each side
        contrasts = torch.rand(2) * self.cfg.dataset.max_contrast  # between 0.0 and 0.5
        for (i, presence) in enumerate(presences):
            if not presence:
                contrasts[i] = 0.0

        # input frames
        cue_frame = self.empty_frame.clone()
        cue_frame = self.add_attention_cue(cue_frame, attend_left)

        gratings_frame = self.get_gratings_frame(contrasts, presences, orientations)

        # sample whether the stimulus frame is on the second or third pass
        stimulus_onset = torch.randint(1, 3, (1,)).item()  # either 1 or 2

        frames = torch.stack([
                cue_frame,
            gratings_frame if stimulus_onset == 1 else self.empty_frame,  # stimulus frame (on the second pass)
            gratings_frame if stimulus_onset == 2 else self.empty_frame,  # stimulus frame (on the third pass)
            self.empty_frame  # clear display frame
        ])

        # retinal noise
        frames += torch.randn_like(frames) * self.retinal_noise

        # target output
        present = presences[attend_idx].long()  # 0 or 1, indicating presence of grating on the attended side
        if present:
            orientation = orientations[attend_idx]  # orientation of the grating on the attended side
        else:
            orientation = torch.tensor(math.pi/2)

        meta_data = {
            'attend_left': attend_left,
            'stimulus_onset': stimulus_onset,
            'orientation_left': orientations[0],
            'orientation_right': orientations[1],
            'contrast_left': contrasts[0],
            'contrast_right': contrasts[1],
            'present_left': presences[0],
            'present_right': presences[1],
        }

        labels = {"what": present} 
        return frames, labels, meta_data


    def get_gratings_frame(self, contrasts, presences, orientations):
        frame = torch.zeros((1, *self.size), dtype=torch.float32)

        for i in range(len(presences)):
            if presences[i]:
                grating = oriented_sine_grating(size=self.size, frequency=self.frequency, orientation=orientations[i], contrast=contrasts[i])
                center = self.center_left if i == 0 else self.center_right
                # add some noise
                shift = self.center_random_shift
                center[0] += torch.randint(-shift, shift+1, (1,)).item()  # small random shift in x
                center[1] += torch.randint(-shift, shift+1, (1,)).item() # small random shift in y
                # random normal
                center = (center[0] + torch.randn(1).item() * shift, center[1] + torch.randn(1).item() * shift)
                mask = circular_aperture_mask(size=self.size, radius=self.radius, center=center)
                frame[0] += grating * mask

        frame += 0.5 # setting the mean luminance to 0.5 (gray)
        return frame
