"""RQ3: time-resolved linear probes on frozen PND representations."""

import argparse
import copy
import numpy as np
import torch
from torch import nn

from experiments.datasets import make_loaders, scifar_bundle
from .common import load_checkpoint, prepare_output, save_json


@torch.no_grad()
def collect(model, loader, device, positions):
    features = [[] for _ in positions]
    labels = []
    for x, y in loader:
        sequence = model(x.to(device), return_sequence=True).cpu()
        for index, position in enumerate(positions):
            features[index].append(sequence[:, position])
        labels.append(y)
    return [torch.cat(values) for values in features], torch.cat(labels)


def fit_probe(train_x, train_y, validation_x, validation_y, classes, epochs):
    probe = nn.Linear(train_x.size(-1), classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    best, best_state = -1.0, None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            accuracy = (probe(validation_x).argmax(-1) == validation_y).float().mean().item()
        if accuracy > best:
            best, best_state = accuracy, copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    return probe, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    output = prepare_output(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, test = make_loaders(scifar_bundle(args.data_dir), args.batch_size, 0, 42)
    model, payload = load_checkpoint(args.checkpoint, device)
    length = 1024
    positions = [min(length - 1, max(0, round(fraction * length) - 1)) for fraction in args.fractions]
    train_x, train_y = collect(model, train, device, positions)
    validation_x, validation_y = collect(model, validation, device, positions)
    test_x, test_y = collect(model, test, device, positions)
    rows = []
    for fraction, tx, vx, sx in zip(args.fractions, train_x, validation_x, test_x):
        probe, validation_accuracy = fit_probe(tx, train_y, vx, validation_y, 10, args.epochs)
        with torch.no_grad():
            test_accuracy = (probe(sx).argmax(-1) == test_y).float().mean().item()
        rows.append({"fraction": fraction, "validation_accuracy": validation_accuracy, "test_accuracy": test_accuracy})
    np.savez_compressed(output / "raw_data.npz", fractions=args.fractions, validation=[x["validation_accuracy"] for x in rows], test=[x["test_accuracy"] for x in rows])
    save_json(output / "config.json", vars(args))
    save_json(output / "metrics.json", {"checkpoint_epoch": payload.get("epoch"), "probes": rows, "scope": "task evidence at selected sequence positions; backbone remains frozen"})


if __name__ == "__main__":
    main()
