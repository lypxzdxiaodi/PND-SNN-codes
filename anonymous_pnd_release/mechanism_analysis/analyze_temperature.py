"""RQ4: hard-sampling sensitivity to the configured Gumbel temperature."""

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
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    output = prepare_output(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = make_loaders(scifar_bundle(args.data_dir), args.batch_size, 0, args.seed)
    model, payload = load_checkpoint(args.checkpoint, device)
    results = {}
    raw = {}
    for temperature in args.temperatures:
        for block in model.layers:
            block.layer.tau = temperature
        values = []
        for repeat in range(args.repeats):
            set_seed(args.seed + repeat)
            values.append(evaluate(model, loader, device)["accuracy"])
        results[str(temperature)] = summary(values)
        raw[f"tau_{temperature}"] = np.asarray(values)
    np.savez_compressed(output / "raw_data.npz", **raw)
    save_json(output / "config.json", vars(args))
    save_json(
        output / "metrics.json",
        {
            "checkpoint_epoch": payload.get("epoch"),
            "temperatures": results,
            "caution": "for hard categorical sampling, a positive common temperature does not change the forward argmax; differences under common RNG should be numerical only",
        },
    )


if __name__ == "__main__":
    main()
