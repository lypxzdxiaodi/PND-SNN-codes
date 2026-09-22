"""Reproducible standalone training utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def optimizer_for(model, lr: float, weight_decay: float):
    regular, special = [], {}
    for parameter in model.parameters():
        options = getattr(parameter, "_optim", None)
        if options is None:
            regular.append(parameter)
        else:
            key = tuple(sorted(options.items()))
            special.setdefault(key, []).append(parameter)
    groups = [{"params": regular, "lr": lr, "weight_decay": weight_decay}]
    for key, parameters in special.items():
        options = dict(key)
        groups.append(
            {
                "params": parameters,
                "lr": options.get("lr", lr),
                "weight_decay": options.get("weight_decay", weight_decay),
            }
        )
    return torch.optim.AdamW(groups)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    predictions, targets = [], []
    for batch in loader:
        x, y, metadata = unpack_batch(batch, device)
        logits = model(x, lengths=metadata.get("lengths"))
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        prediction = logits.argmax(dim=-1)
        correct += (prediction == y).sum().item()
        total += y.numel()
        predictions.append(prediction.cpu())
        targets.append(y.cpu())
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
    }


def unpack_batch(batch, device):
    if len(batch) == 2:
        x, y = batch
        metadata = {}
    else:
        x, y, metadata = batch
    if isinstance(x, (tuple, list)):
        x = tuple(item.to(device) for item in x)
    else:
        x = x.to(device)
    metadata = {key: value.to(device) for key, value in metadata.items()}
    return x, y.to(device), metadata


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_validation, extra=None):
    payload = {
        "model": model.state_dict(),
        "model_config": model.export_config(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_accuracy": best_validation,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
