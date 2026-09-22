"""RQ1/RQ2: temporal modes, complexity, redundancy, and effective rank."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import load_checkpoint, mode_kernels, prepare_output, save_json, summary


def safe_correlation(matrix):
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norm, 1e-12)
    return normalized @ normalized.T


def effective_rank(matrix):
    singular = np.linalg.svd(matrix.T, compute_uv=False)
    probability = singular / max(singular.sum(), 1e-12)
    rank = np.exp(-(probability * np.log(probability + 1e-12)).sum())
    return singular, probability, float(rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = prepare_output(args.output)
    model, payload = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    all_modes, records = [], []
    for layer_index, block in enumerate(model.layers):
        modes = mode_kernels(block.layer.kernel, args.length).detach().cpu().numpy()
        compressed = modes.sum(axis=1)
        amplitude = np.linalg.norm(compressed, axis=1)
        high = np.argsort(amplitude)[-args.channels // 2 :]
        remaining = np.setdiff1d(np.arange(len(amplitude)), high)
        random = rng.choice(remaining, size=args.channels - len(high), replace=False)
        selected = np.concatenate([high, random])
        for channel in selected:
            basis = modes[channel]
            correlation = safe_correlation(basis)
            off_diagonal = np.abs(correlation[~np.eye(len(correlation), dtype=bool)])
            singular, spectrum, rank = effective_rank(basis)
            kernel = compressed[channel]
            slope = np.diff(np.log(np.abs(kernel) + 1e-8))
            records.append(
                {
                    "layer": layer_index,
                    "channel": int(channel),
                    "selection": "high_amplitude" if channel in high else "random",
                    "log_magnitude_slope_variance": float(np.var(slope)),
                    "mean_absolute_off_diagonal_correlation": float(off_diagonal.mean()),
                    "effective_rank": rank,
                    "normalized_effective_rank": rank / basis.shape[0],
                }
            )
            all_modes.append((layer_index, int(channel), basis, kernel, correlation, spectrum))
    figure, axes = plt.subplots(len(all_modes), 2, figsize=(12, 2.5 * len(all_modes)))
    axes = np.atleast_2d(axes)
    for row, (layer, channel, basis, kernel, correlation, _) in enumerate(all_modes):
        axes[row, 0].plot(basis.T, alpha=0.45, linewidth=0.8)
        axes[row, 0].plot(kernel, color="black", linewidth=1.5, label="compressed")
        axes[row, 0].set_title(f"layer {layer}, channel {channel}")
        axes[row, 1].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    figure.tight_layout()
    figure.savefig(output / "figures" / "temporal_modes_and_correlation.png", dpi=200)
    plt.close(figure)
    np.savez_compressed(
        output / "raw_data.npz",
        modes=np.array([item[2] for item in all_modes]),
        compressed=np.array([item[3] for item in all_modes]),
        correlations=np.array([item[4] for item in all_modes]),
        spectra=np.array([item[5] for item in all_modes]),
    )
    metrics = {
        "checkpoint_epoch": payload.get("epoch"),
        "channels": records,
        "response_complexity": summary([row["log_magnitude_slope_variance"] for row in records]),
        "mode_redundancy": summary([row["mean_absolute_off_diagonal_correlation"] for row in records]),
        "effective_rank": summary([row["effective_rank"] for row in records]),
        "interpretation": {
            "complexity": "departure from a single constant geometric response; larger is not universally better",
            "correlation": "temporal-basis redundancy, not activation-state redundancy",
            "rank": "number of empirically occupied temporal response directions",
        },
    }
    save_json(output / "config.json", vars(args))
    save_json(output / "metrics.json", metrics)


if __name__ == "__main__":
    main()
