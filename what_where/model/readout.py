

import torch.nn as nn
import torch.nn.functional as F

class TinyImageNetReadout(nn.Module):
    def __init__(self, cfg):
        super(TinyImageNetReadout, self).__init__()
        self.cfg = cfg
        self.rnn_hidden_size = self.cfg.model.rnn.hidden_size

        self.n_classes = 200 # what output

        # what output only (classification)
        self.what = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(self, hidden, out, t, readout_in=None):
        out[t]["prediction"]["what"] = F.log_softmax(self.what(hidden), dim=1) # classification output
        return out



class VCSReadout(nn.Module):
    def __init__(self, cfg):
        super(VCSReadout, self).__init__()
        self.cfg = cfg
        self.rnn_hidden_size = self.cfg.model.rnn.hidden_size

        self.n_classes = cfg.dataset.what.size # what output
        self.n_locations = cfg.dataset.where.size**2 # where output

        # what and where outputs
        self.what = nn.Linear(self.rnn_hidden_size, self.n_classes)
        self.where = nn.Linear(self.rnn_hidden_size, self.n_locations)

    def forward(self, rnn_out, out, t, readout_in=None):
        out[t]["prediction"]["what"] = F.log_softmax(self.what(rnn_out), dim=1)
        out[t]["prediction"]["where"] = F.log_softmax(self.where(rnn_out), dim=1)
        return out


class ContrastDetectionReadout(nn.Module):
    def __init__(self, cfg):
        super(ContrastDetectionReadout, self).__init__()
        self.cfg = cfg
        self.rnn_hidden_size = self.cfg.model.rnn.hidden_size

        # binary output
        self.present = nn.Linear(self.rnn_hidden_size, 2)

    def forward(self, rnn_out, out, t, readout_in=None):
        out[t]["prediction"]["what"] = F.log_softmax(self.present(rnn_out), dim=1)
        return out


class OrientationChangeDetectionReadout(nn.Module):
    def __init__(self, cfg):
        super(OrientationChangeDetectionReadout, self).__init__()
        self.cfg = cfg
        self.rnn_hidden_size = self.cfg.model.rnn.hidden_size

        # softmax output (saccade left, right, or keep at center)
        # the model is tasked to saccade to the side where the orientation change happened, otherwise keep at center
        self.saccade = nn.Linear(self.rnn_hidden_size, 3)


    def forward(self, rnn_out, out, t, readout_in=None):
        out[t]["prediction"]["what"] = F.log_softmax(self.saccade(rnn_out), dim=1)
        return out



class ImagenetReadout(nn.Module):
    def __init__(self, cfg):
        super(ImagenetReadout, self).__init__()
        self.cfg = cfg
        self.rnn_hidden_size = self.cfg.model.rnn.hidden_size

        self.n_classes = 1000 # what output

        # what output only (classification)
        self.what = nn.Linear(self.rnn_hidden_size, self.n_classes)

    def forward(self, rnn_out, out, t, readout_in=None):
        out[t]["prediction"]["what"] = F.log_softmax(self.what(rnn_out), dim=1) # classification output
        return out
    
