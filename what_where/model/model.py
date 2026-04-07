import torch
from torch import nn
from torch.nn import functional as F
import hydra

import what_where as ww



class Model(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg

        # CNN
        self.cnn = ww.model.CNN(cfg)

        # RNN
        self.rnn = ww.model.RNN(cfg)

        # Gain
        self.gain = ww.model.Gain(cfg)

        # READOUT
        self.readout = ww.utils.get_readout(cfg)

        if cfg.model.cnn.pre_training.load_weights:
            self.cnn.conv_layers.requires_grad_(False) # freezing the CNN when just loading the weights for transfer


    def _init_out(self, n_passes):
        out = {}
        for t in range(n_passes):
            out[t] = {
                "prediction" : {}, # model output
                "gain" : {}, # gain patterns for what, where, when

                # elements for energy computation
                "activations": {},
                "synaptic_transmission": {},
                "normalization": {},
                "normalization_pooled_mean": {}, # for logging

                # saving this for time based divisive normalization
                "suppressive_drive": {},
            }
        return out


    def forward_t(self, x, hidden_rnn, out, t, noise_anneal, gain_suppression=None, fixed_gain=[], readout_in=None):
        # gain
        fixed_gain_t = fixed_gain[t] if fixed_gain else {} # can pass fixed gain patterns for each pass

        gain_dict = self.gain(hidden_rnn, out, t, noise_anneal, fixed_gain_t)
        out[t]["gain"] = gain_dict

        # cnn
        x, combined_gains = self.cnn(x, out, t, gain_dict, gain_suppression, noise_anneal)
        out[t]["gain"]["combined"] = combined_gains # saving the combined gains for visualization

        # rnn
        x = x.view(x.size(0), -1) # flattening
        x, hidden_rnn = self.rnn(x, hidden_rnn, out, t, noise_anneal)

        # readout
        self.readout(x, out, t, readout_in) # what and where prediction

        return hidden_rnn


    def forward(self, x, log_energy_cost, n_passes,
                noise_anneal,
                gain_suppression=None, # dict(layer_name: (batch_size, n_passes) tensor with gain suppression amount for each pass)
                # out is a dict with keys "prediction", "gain", "activations", "synaptic_transmission", "normalization"
                fixed_gain=[]): # a list of gain dictionaries
        
        assert x.dim() in [4,5], "Input must be a 4D tensor (batch_size, channels, height, width) or a 5D tensor (batch_size, n_frames, channels, height, width)"
        assert n_passes > 0, "Number of passes must be greater than 0"
        if x.dim() == 5:
            assert x.size(1) == n_passes, "If input is a 5D tensor, the second dimension must match n_passes"
        assert len(fixed_gain) == 0 or len(fixed_gain) == n_passes, "Fixed gain must be empty or have the same length as n_passes"
        assert log_energy_cost.dim() == 2, "log_energy_cost must be a 2D tensor (batch_size, 1)"
        assert x.size(0) == log_energy_cost.size(0), "Batch size of input and log_energy_cost must match"
                
        out = self._init_out(n_passes)

        # initializing the hidden state using log energy cost for the inference
        hidden_rnn = self.rnn.init_hidden(log_energy_cost, out, noise_anneal).to(x.device)

        for t in range(n_passes):
            x_t = x if x.dim() == 4 else x[:, t]  # if input is a 5D tensor, take the t-th frame
            hidden_rnn = self.forward_t(x_t, hidden_rnn, out, t, noise_anneal, gain_suppression, fixed_gain, readout_in=None)

        return out


