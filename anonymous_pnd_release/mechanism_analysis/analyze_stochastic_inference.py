"""RQ5: prediction variance induced only by hard stochastic firing."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from experiments.datasets import make_loaders, scifar_bundle
from pnd.training import evaluate, set_seed
from .common import load_checkpoint, prepare_output, save_json, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    output = prepare_output(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = make_loaders(scifar_bundle(args.data_dir), args.batch_size, 0, args.seed)
    model, payload = load_checkpoint(args.checkpoint, device)
    accuracies, predictions = [], []
    for repeat in range(args.repeats):
        set_seed(args.seed + repeat)
        result = evaluate(model, loader, device)
        accuracies.append(result["accuracy"])
        predictions.append(result["predictions"].numpy())
    predictions = np.stack(predictions)
    agreement = []
    for column in predictions.T:
        _, counts = np.unique(column, return_counts=True)
        agreement.append(counts.max() / args.repeats)
    agreement = np.asarray(agreement)
    np.savez_compressed(output / "raw_data.npz", accuracies=accuracies, predictions=predictions, agreement=agreement)
    save_json(output / "config.json", vars(args))
    save_json(
        output / "metrics.json",
        {
            "checkpoint_epoch": payload.get("epoch"),
            "accuracy": summary(accuracies),
            "mean_majority_agreement": float(agreement.mean()),
            "full_agreement_fraction": float((agreement == 1.0).mean()),
            "at_least_90_percent_agreement_fraction": float((agreement >= 0.9).mean()),
            "unstable_fraction": float((agreement < 0.6).mean()),
            "note": "checkpoint and data order are fixed; only inference RNG changes",
        },
    )


if __name__ == "__main__":
    main()
