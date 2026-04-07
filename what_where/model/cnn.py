

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import what_where as ww



class CNNLayer(nn.Module):
    def __init__(self, cfg, layer_name, layer_index, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.cfg = cfg
        self.layer_name = layer_name

        cumulative_stride = 1
        for i in range(layer_index + 1):
            cumulative_stride *= cfg.model.cnn.strides[i]

        h = w = cfg.dataset.large_img_size // cumulative_stride
        self.layer_shape = (out_channels, h, w) # activation shape (for hidden state)

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

        # if there are multiple gain types, we divide by the max (to normalize the combined gain)
        n_gain_types = cfg.model.gain.what.active + cfg.model.gain.where.active + cfg.model.gain.when.active
        if n_gain_types == 1 or n_gain_types == 0:
            self.gain_function = lambda x: x # identity
        elif n_gain_types == 2:
            self.gain_function = lambda x: torch.sqrt(x)
        else:
            raise NotImplementedError("More than two gain types not implemented")

        if cfg.model.cnn.normalization.active:
            pool_size = cfg.model.cnn.normalization.pool_sizes[layer_index]
            self.divisive_normalization = ww.model.DivisiveNormalizationConv(cfg, layer_name+"_norm", pool_size=pool_size)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')




    def _apply_gain_suppression(self, gain, suppression):
        if suppression is not None:
            # Generate suppression factors between 0 and 1
            suppression_factors = torch.exp(-torch.abs(torch.randn_like(gain)) * suppression.unsqueeze(1).unsqueeze(2).unsqueeze(3))
            
            # Scale towards 1.0 (neutral attention)
            gain = 1.0 + (gain - 1.0) * suppression_factors

        return gain

    def _combine_gains(self, x, gain_dict, suppression, layer_name, baseline_gain_param):
        gain = torch.ones_like(x)

        if gain_dict["what"][layer_name] is not None:
            what_gain = gain_dict["what"][layer_name]
            what_gain = ww.utils.scale_gain(self.cfg, what_gain)
            what_gain_resized = what_gain.unsqueeze(2).unsqueeze(3)
            gain *= what_gain_resized

        if gain_dict["where"] is not None:
            where_gain = gain_dict["where"]
            where_gain = F.interpolate(where_gain.unsqueeze(1), size=x.shape[-1], mode="bilinear")
            where_gain = ww.utils.scale_gain(self.cfg, where_gain)
            gain *= where_gain

        if gain_dict["when"] is not None:
            when_gain = gain_dict["when"]
            when_gain = ww.utils.scale_gain(self.cfg, when_gain)
            gain *= when_gain.unsqueeze(2).unsqueeze(3)

        if self.cfg.model.name == "baseline":
            baseline_gain = ww.utils.scale_gain(self.cfg, torch.sigmoid(baseline_gain_param))
            gain *= baseline_gain

        gain = self.gain_function(gain) # applying the gain function (e.g. square root for two gain types)


        if suppression is not None:
            gain = self._apply_gain_suppression(gain, suppression)



        if "combined" not in gain_dict:
            gain_dict["combined"] = {}
        gain_dict["combined"][layer_name] = gain

        return gain


    def forward(self, x, out, t, gain_dict, gain_suppression_layer, noise_anneal, baseline_gain_param):

        # computing synaptic transmission
        out[t]["synaptic_transmission"][f"{self.layer_name}"] = ww.utils.compute_synaptic_transmission_conv(x, self.conv_layer, sample_ratio=self.cfg.train.energy.st_sample_ratio)

        # feedforward convolution
        x = self.conv_layer(x) # clean signal

        # gain
        suppression = gain_suppression_layer[:, t] if gain_suppression_layer is not None else None # layer and pass selected (only used in the debes & dragoi replication)
        combined_gain = self._combine_gains(x, gain_dict, suppression, self.layer_name, baseline_gain_param)

        x = x * combined_gain

        # noise
        # cell voltage fluctuation (noise) is independent of gain rate
        # therefore we apply it after the gain modulation
        # https://www.cell.com/neuron/pdf/S0896-6273(02)00820-6.pdf
        x = ww.utils.apply_noise(self.cfg, x, noise_anneal) 

        # activations
        x = F.relu(x)

        # divise normalization
        if self.cfg.model.cnn.normalization.active:
            x = self.divisive_normalization(x, out, t)

        out[t]['activations'][self.layer_name] = x # saving the activations for energy loss

        return x, combined_gain



class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.conv_channels = self.cfg.model.cnn.conv_channels
        self.kernel_sizes = self.cfg.model.cnn.kernel_sizes
        self.strides = self.cfg.model.cnn.strides

        # constructing the convolutional layers
        conv_layers = []
        all_channels = [1] + self.conv_channels # including the image
        for i in range(1, len(all_channels)):
            kernel_size = self.kernel_sizes[i-1]
            stride = self.strides[i-1]

            # conv_layers.append(CNNLayer(cfg, f"conv{i}", i-1, all_channels[i-1], all_channels[i], kernel_size, stride=2))
            conv_layers.append(CNNLayer(cfg, f"conv{i}", i-1, all_channels[i-1], all_channels[i], kernel_size, stride))

        self.conv_layers = nn.ModuleList(conv_layers)

        # for baseline model we instantiate global param for gain, so that
        # baseline model can increase SNR across all examples non-adaptively
        if cfg.model.name == "baseline":
            self.baseline_gain_param = nn.Parameter(torch.zeros(1), requires_grad=True) # initialized at 0 -> sigmoid(0) = 0.5
        else:
            self.baseline_gain_param = None


    def get_layer_names(self):
        return [layer.layer_name for layer in self.conv_layers]


    def forward(self, x, out, t, gain_dict, gain_suppression, noise_anneal):
        combined_gains = {}

        for layer in self.conv_layers:
            gain_suppression_layer = gain_suppression[layer.layer_name] if (gain_suppression and layer.layer_name in gain_suppression) else None

            x, combined_gain = layer(x, out, t, gain_dict, gain_suppression_layer, noise_anneal, self.baseline_gain_param)
            combined_gains[layer.layer_name] = combined_gain

        return x, combined_gains

