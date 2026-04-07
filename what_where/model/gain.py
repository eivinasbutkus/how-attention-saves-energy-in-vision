

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from what_where.model.mlp import MLP

class Gain(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden_size = cfg.model.rnn.hidden_size

        self.conv_channels = cfg.model.cnn.conv_channels

        # GAIN
        if cfg.model.gain.what.active:
            self.what_gain = MLP(cfg, "what_gain_mlp", hidden_size, cfg.model.gain.mlp_hidden, sum(self.conv_channels))

        if cfg.model.gain.where.active:
            where_gain_output = cfg.model.gain.where.size**2
            self.where_gain = MLP(cfg, "where_gain_mlp", hidden_size, cfg.model.gain.mlp_hidden, where_gain_output)

        if cfg.model.gain.when.active:
            self.when_gain = MLP(cfg, "when_gain_mlp", hidden_size, cfg.model.gain.mlp_hidden, 1)


    def get_what_gain(self, hidden, out, t, noise_anneal, fixed_gain):
        if "what" in fixed_gain:
            return fixed_gain["what"]

        if not self.cfg.model.gain.what.active:
            return{**{layer_name: None for layer_name in ["conv{}".format(i+1) for i in range(len(self.conv_channels))]}}

        gain_input = hidden[:,-1,:] # taking the last layer
        what_gain_full = self.what_gain(gain_input, out, t, noise_anneal) # predicting the full gain using MLP for all layers

        what_gain_dict = {}
        # carving out the full gain by conv layer and applying sigmoid
        start_channel = 0
        for (i, n_channels) in enumerate(self.conv_channels):
            layer_name = "conv{}".format(i+1)
            end_channel = start_channel + n_channels
            what_gain = what_gain_full[:, start_channel:end_channel]
            start_channel = end_channel

            what_gain = F.sigmoid(self.cfg.model.gain.sensitivity * what_gain) # applying sensitivity

            what_gain_dict[layer_name] = what_gain

        return what_gain_dict


    def get_where_gain(self, hidden, out, t, noise_anneal, fixed_gain):
        if not self.cfg.model.gain.where.active:
            return None

        if "where" in fixed_gain:
            return fixed_gain["where"]

        # batch_size = hidden.size(1)
        batch_size = hidden.size(0)
        n = self.cfg.model.gain.where.size
        shape = (batch_size, n, n)

        gain_input = hidden[:,-1,:] # taking the last layer
        gain = self.where_gain(gain_input, out, t, noise_anneal)

        # spatial smoothing to prevent checkerboard
        if self.cfg.model.gain.where.smoothing.enabled:
            kernel_size = self.cfg.model.gain.where.smoothing.kernel_size
            sigma = self.cfg.model.gain.where.smoothing.sigma
            gain_smoothed = kornia.filters.gaussian_blur2d(
                gain.reshape(shape).unsqueeze(1),
                kernel_size=(kernel_size, kernel_size),
                sigma=(sigma, sigma),
                border_type=self.cfg.model.gain.where.smoothing.border_type
            ).squeeze(1)
        else:
            gain_smoothed = gain.reshape(shape)

        where_gain = F.sigmoid(self.cfg.model.gain.sensitivity * gain_smoothed)

        return where_gain


    def get_when_gain(self, hidden, out, t, noise_anneal, fixed_gain):
        if not self.cfg.model.gain.when.active:
            return None

        if "when" in fixed_gain:
            return fixed_gain["when"]

        gain_input = hidden[:,-1,:] # taking the last layer
        gain = self.when_gain(gain_input, out, t, noise_anneal)

        when_gain = F.sigmoid(self.cfg.model.gain.sensitivity * gain)

        return when_gain


    def forward(self, hidden, out, t, noise_anneal, fixed_gain):
                
        # compute gain (or retrieve from input)
        gain = {}

        gain["what"] = self.get_what_gain(hidden, out, t, noise_anneal, fixed_gain)
        gain["where"] = self.get_where_gain(hidden, out, t, noise_anneal, fixed_gain)

        if self.cfg.model.gain.when.active and t > 0 and self.cfg.model.gain.when.constant:
            # global gain model is implemented by using a constant "when" gain across time (initialized at first time step)
            gain["when"] = out[0]["gain"]["when"]
        else:
            # normal when gain
            gain["when"] = self.get_when_gain(hidden, out, t, noise_anneal, fixed_gain)

        return gain
