
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia



class DivisiveNormalizationConv(nn.Module):
    def __init__(self, cfg, layer_name, pool_size=5):
        super().__init__()
        self.cfg = cfg
        self.layer_name = layer_name
        self.pool_size = pool_size
        
        # fixed parameters
        self.sigma = cfg.model.cnn.normalization.sigma # used in the denominator
        self.alpha = cfg.model.cnn.normalization.alpha
        self.sigma_spatial = cfg.model.cnn.normalization.sigma_spatial

        self.register_buffer('running_scale', torch.ones(1))
        self.momentum = 0.1

    
    def forward(self, x, out, t):
        channel_mean = x.mean(dim=1, keepdim=True)
        pooled = kornia.filters.gaussian_blur2d(
            channel_mean,
            kernel_size=(self.pool_size, self.pool_size),
            sigma=(self.sigma_spatial, self.sigma_spatial),
            border_type='reflect'
        )

        if self.training:
            current_scale = channel_mean.mean() + 1e-05
            with torch.no_grad():
                self.running_scale = (1 - self.momentum) * self.running_scale + \
                                     self.momentum * current_scale

        denominator = self.sigma + self.alpha * pooled / self.running_scale
        normalized_x = x / denominator

        # Store for energy computation
        out[t]["normalization"][self.layer_name] = denominator.sum(dim=(1,2,3))  # batch-level energy
        
        return normalized_x


class DivisiveNormalizationRNN(nn.Module):

    def __init__(self, cfg, layer_name):
        super().__init__()
        self.cfg = cfg
        self.layer_name = layer_name
        
        self.sigma = cfg.model.rnn.normalization.sigma
        self.alpha = cfg.model.rnn.normalization.alpha

    def forward(self, x, out, t):
        # For RNN: simple mean pooling across the hidden dimension
        pooled = x.mean(dim=1, keepdim=True)
        denominator = self.sigma + self.alpha * pooled

        # Normalization
        normalized_x = x / denominator

        out[t]["normalization"][self.layer_name] = self.alpha * pooled.sum(dim=1)  # batch-level energy

        return normalized_x