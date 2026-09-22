"""Small residual sequence classifier shared by standalone experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .raw_snn_s4 import (
    Snn_s4,
    Snn_s4_linear,
    bidir_psnns4,
    bsnns4,
    bsnns4_linear,
    psnns4,
)
from .ablations import SigmoidPND


LAYER_TYPES = {
    "pnd": psnns4,
    "hard_gumbel": Snn_s4,
    "bernoulli_st": bsnns4,
    "linear_gumbel": Snn_s4_linear,
    "linear_bernoulli_st": bsnns4_linear,
    "bidirectional_pnd": bidir_psnns4,
    "sigmoid_pnd": SigmoidPND,
}


@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    d_model: int = 256
    d_state: int = 64
    n_layers: int = 6
    dropout: float = 0.0
    tau: float = 1.0
    layer: str = "pnd"
    input_type: str = "continuous"
    prenorm: bool = True
    pool: str = "mean"
    only_spike: bool = False
    fusion_scale: float = 2**-0.5


class ResidualBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        layer_type = LAYER_TYPES[config.layer]
        extra = {"fusion_scale": config.fusion_scale} if layer_type is bidir_psnns4 else {}
        self.norm = nn.LayerNorm(config.d_model)
        self.layer = layer_type(
            config.d_model,
            d_state=config.d_state,
            tau=config.tau,
            dropout=config.dropout,
            transposed=True,
            **extra,
        )
        self.prenorm = config.prenorm
        self.only_spike = config.only_spike

    def forward(self, x):
        residual = x
        if self.prenorm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if self.only_spike:
            probability = self.layer.firing_probability(self.layer.membrane(x))
            sampled = self.layer._hard_gumbel(probability, hard=True).detach()
            x = sampled - probability.detach() + probability
        else:
            x, _ = self.layer(x)
        x = residual + x
        if not self.prenorm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x


class PNDSequenceClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.input_type == "tokens":
            self.encoder = nn.Embedding(config.input_dim, config.d_model, padding_idx=0)
        else:
            self.encoder = nn.Linear(config.input_dim, config.d_model)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Linear(config.d_model, config.num_classes)

    def encode(self, x):
        return self.encoder(x).transpose(1, 2)

    def representations(self, x):
        x = self.encode(x)
        states = []
        for layer in self.layers:
            x = layer(x)
            states.append(x.transpose(1, 2))
        return states

    def forward(self, x, lengths=None, return_sequence=False):
        sequence = self.representations(x)[-1]
        if return_sequence:
            return sequence
        if self.config.pool == "last":
            if lengths is None:
                pooled = sequence[:, -1]
            else:
                index = (lengths - 1).clamp_min(0)
                pooled = sequence[torch.arange(sequence.size(0), device=x.device), index]
        else:
            if lengths is None:
                pooled = sequence.mean(dim=1)
            else:
                mask = torch.arange(sequence.size(1), device=x.device)[None] < lengths[:, None]
                pooled = (sequence * mask.unsqueeze(-1)).sum(1) / lengths.clamp_min(1).unsqueeze(-1)
        return self.decoder(pooled)

    def export_config(self):
        return asdict(self.config)


class PNDPairClassifier(PNDSequenceClassifier):
    """AAN-style paired sequence classifier with shared token embedding."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.pair_projection = nn.Linear(2 * config.d_model, config.d_model)

    def forward(self, pair, lengths=None, return_sequence=False):
        first, second = pair
        first = self.encoder(first)
        second = self.encoder(second)
        x = self.pair_projection(torch.cat([first, second], dim=-1)).transpose(1, 2)
        states = []
        for layer in self.layers:
            x = layer(x)
            states.append(x.transpose(1, 2))
        sequence = states[-1]
        if return_sequence:
            return sequence
        return self.decoder(sequence.mean(dim=1))
