
import torch
import torch.nn as nn
import what_where as ww


"""
MLP with activation, synaptic transmission, and noise
"""
class MLP(nn.Module):
    def __init__(self, cfg, name, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.cfg = cfg
        self.name = name

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.sample_ratio = cfg.train.energy.st_sample_ratio

        # He initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)



    def forward(self, x_in, out, t, noise_anneal):
        # synaptic transmission fc1
        out[t]["synaptic_transmission"][f"{self.name}_fc1"] = ww.utils.compute_synaptic_transmission_linear(x_in, self.fc1, sample_ratio=self.sample_ratio)
        x = self.fc1(x_in)

        # noise
        x = ww.utils.apply_noise(self.cfg, x, noise_anneal)

        # activation
        x = torch.relu(x)
        out[t]['activations'][self.name] = x

        # synaptic transmission fc2
        out[t]["synaptic_transmission"][f"{self.name}_fc2"] = ww.utils.compute_synaptic_transmission_linear(x, self.fc2, sample_ratio=self.sample_ratio)
        x = self.fc2(x)

        return x

