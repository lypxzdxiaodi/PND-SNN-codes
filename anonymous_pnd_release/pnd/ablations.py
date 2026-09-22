"""Additional ablations kept outside the core compatibility file."""

import torch
from torch import nn

from .raw_snn_s4 import psnns4


class SigmoidPND(psnns4):
    """Replace the exponential CDF with a learned sigmoid probability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability_bias = nn.Parameter(torch.zeros(self.h))

    def firing_probability(self, membrane):
        return torch.sigmoid(
            self.aver.unsqueeze(-1) * membrane
            + self.probability_bias.unsqueeze(-1)
        )


