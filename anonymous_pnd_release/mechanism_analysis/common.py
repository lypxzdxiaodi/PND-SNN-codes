from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pnd.model import ModelConfig, PNDSequenceClassifier


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PNDSequenceClassifier(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def prepare_output(path):
    root = Path(path)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    return root


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def mode_kernels(kernel, length):
    dt = torch.exp(kernel.log_dt)
    poles = -torch.exp(kernel.log_A_real) + 1j * kernel.A_imag
    coefficients = torch.view_as_complex(kernel.C)
    discrete = poles * dt.unsqueeze(-1)
    coefficients = coefficients * torch.expm1(discrete) / poles
    time = torch.arange(length, device=poles.device)
    return 2 * (coefficients.unsqueeze(-1) * torch.exp(discrete.unsqueeze(-1) * time)).real


class SpikeAccumulator:
    def __init__(self, model):
        self.values = [[] for _ in model.layers]
        self.handles = []
        for index, block in enumerate(model.layers):
            self.handles.append(block.layer.dropout.register_forward_pre_hook(self._hook(index)))

    def _hook(self, index):
        def collect(_module, inputs):
            self.values[index].append(inputs[0].detach().float().cpu())
        return collect

    def close(self):
        for handle in self.handles:
            handle.remove()

    def rates(self):
        return [float(torch.cat([x.flatten() for x in values]).mean()) for values in self.values]
