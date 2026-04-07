
import torch
import torch.nn as nn

import what_where as ww



class RNNLayer(nn.Module):
    def __init__(self, cfg, layer_name, input_size, hidden_size):
        super().__init__()

        self.cfg = cfg
        self.layer_name = layer_name
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)

        # normalization
        nn.init.kaiming_normal_(self.input_to_hidden.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.hidden_to_hidden.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.input_to_hidden.bias, 0)
        nn.init.constant_(self.hidden_to_hidden.bias, 0)

        if cfg.model.rnn.normalization.active:
            self.divisive_normalization = ww.model.DivisiveNormalizationRNN(cfg, layer_name)


    def forward(self, x, hidden, out, t, noise_anneal):

        # computing synaptic transmission use
        out[t]["synaptic_transmission"][f"{self.layer_name}_ih"] = ww.utils.compute_synaptic_transmission_linear(x, self.input_to_hidden, sample_ratio=self.cfg.train.energy.st_sample_ratio)
        out[t]["synaptic_transmission"][f"{self.layer_name}_hh"] = ww.utils.compute_synaptic_transmission_linear(hidden, self.hidden_to_hidden, sample_ratio=self.cfg.train.energy.st_sample_ratio)

        hidden_pre = self.input_to_hidden(x) + self.hidden_to_hidden(hidden) # computing preactivations

        # noise
        hidden_pre = ww.utils.apply_noise(self.cfg, hidden_pre, noise_anneal)

        # activations
        hidden = torch.relu(hidden_pre)

        if self.cfg.model.rnn.normalization.active:
            hidden = self.divisive_normalization(hidden, out, t)

        out[t]['activations'][self.layer_name] = hidden # saving the activations for energy loss

        return hidden



class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        rnn_layers = []
        cumulative_stride = 1
        for stride in cfg.model.cnn.strides:
            cumulative_stride *= stride
        w = cfg.dataset.large_img_size // cumulative_stride

        input_size = int(w * w * self.cfg.model.cnn.conv_channels[-1])
        hidden_size = cfg.model.rnn.hidden_size

        for i in range(cfg.model.rnn.n_layers):
            name = f"rnn{i+1}"
            rnn_layers.append(RNNLayer(cfg, name, input_size, hidden_size))
            input_size = hidden_size

        self.rnn_layers = nn.ModuleList(rnn_layers)

        # flexible initialization of hidden state based on energy cost for a particular trial
        self.h0 = ww.model.MLP(cfg, "rnn_init_mlp",
                               input_size=1,
                               hidden_size=cfg.model.rnn.init_mlp_hidden_size,
                               output_size=hidden_size*cfg.model.rnn.n_layers)
        nn.init.normal_(self.h0.fc2.weight, mean=0, std=0.01)


    # this initialize the hidden state using the energy cost for this particular trial (implements flexible energy cost relative to accuracy)
    def init_hidden(self, log_energy_cost, out, noise_anneal):
        t = 0 # initializing at time step 0
        hidden_pre = self.h0(log_energy_cost, out, t, noise_anneal)
        hidden = torch.relu(hidden_pre)
        hidden = hidden.reshape(-1, self.cfg.model.rnn.n_layers, self.cfg.model.rnn.hidden_size) # (batch_size, n_layers, hidden_size)
        out[t]["activations"]["rnn_init_ap"] = hidden
        return hidden


    def forward(self, x, hidden, out, t, noise_anneal):
        new_hidden = torch.zeros_like(hidden).to(x.device)

        for (i, layer) in enumerate(self.rnn_layers):
            x = layer(x, hidden[:,i], out, t, noise_anneal)
            new_hidden[:,i] = x

        return x, new_hidden

