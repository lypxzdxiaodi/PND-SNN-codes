"""Compare exponential-CDF and sigmoid firing-probability checkpoints."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.datasets import make_loaders, scifar_bundle
from .common import SpikeAccumulator, load_checkpoint, prepare_output, save_json, summary


@torch.no_grad()
def collect(model, loader, device, max_batches):
    probabilities, confidence = [], []
    accumulator = SpikeAccumulator(model)
    correct = total = 0
    try:
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            layer = model.layers[0].layer
            encoded = model.encode(x)
            probabilities.append(layer.firing_probability(layer.membrane(encoded)).cpu())
            logits = model(x)
            score, prediction = logits.softmax(-1).max(-1)
            confidence.append(score.cpu())
            correct += (prediction == y).sum().item()
            total += y.numel()
    finally:
        accumulator.close()
    return {
        "probabilities": torch.cat([x.flatten() for x in probabilities]).numpy(),
        "confidence": torch.cat(confidence).numpy(),
        "accuracy": correct / total,
        "spike_rates": accumulator.rates(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exponential-checkpoint", required=True)
    parser.add_argument("--sigmoid-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()
    output = prepare_output(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = make_loaders(scifar_bundle(args.data_dir), args.batch_size, 0, 42)
    exponential, _ = load_checkpoint(args.exponential_checkpoint, device)
    sigmoid, _ = load_checkpoint(args.sigmoid_checkpoint, device)
    values = {
        "exponential": collect(exponential, loader, device, args.max_batches),
        "sigmoid": collect(sigmoid, loader, device, args.max_batches),
    }
    figure, axis = plt.subplots(figsize=(6, 4))
    for name, color in (("exponential", "tab:blue"), ("sigmoid", "tab:orange")):
        axis.hist(values[name]["probabilities"], bins=50, density=True, alpha=0.55, label=name, color=color)
    axis.set_xlabel("firing probability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figures" / "probability_distribution.png", dpi=200)
    plt.close(figure)
    np.savez_compressed(
        output / "raw_data.npz",
        exponential_probability=values["exponential"]["probabilities"],
        sigmoid_probability=values["sigmoid"]["probabilities"],
        exponential_confidence=values["exponential"]["confidence"],
        sigmoid_confidence=values["sigmoid"]["confidence"],
    )
    save_json(output / "config.json", vars(args))
    save_json(output / "metrics.json", {
        name: {
            "subset_accuracy": item["accuracy"],
            "spike_rates": item["spike_rates"],
            "probability": summary(item["probabilities"]),
        }
        for name, item in values.items()
    })


if __name__ == "__main__":
    main()
