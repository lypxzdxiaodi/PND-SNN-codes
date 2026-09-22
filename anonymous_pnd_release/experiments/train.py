"""Standalone training for sCIFAR and LRA tasks.

Example:
    python -m experiments.train --task scifar --data-dir ./data/cifar --output runs/scifar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pnd.model import ModelConfig, PNDPairClassifier, PNDSequenceClassifier
from pnd.training import evaluate, optimizer_for, save_checkpoint, save_json, set_seed, unpack_batch
from .datasets import (
    aan_bundle,
    imdb_bundle,
    listops_bundle,
    make_loaders,
    pathfinder_bundle,
    scifar_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["scifar", "listops", "imdb", "aan", "pathfinder"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", choices=["pnd", "hard_gumbel", "bernoulli_st", "linear_gumbel", "linear_bernoulli_st", "bidirectional_pnd", "sigmoid_pnd"], default="pnd")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only-spike", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--center", action="store_true")
    return parser.parse_args()


def build_data(args):
    if args.task == "scifar":
        return scifar_bundle(args.data_dir, args.seed, grayscale=True, download=args.download)
    if args.task == "listops":
        return listops_bundle(args.data_dir)
    if args.task == "imdb":
        return imdb_bundle(args.data_dir, seed=args.seed)
    if args.task == "aan":
        return aan_bundle(args.data_dir)
    return pathfinder_bundle(args.data_dir, seed=args.seed, center=args.center)


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_data(args)
    train_loader, validation_loader, test_loader = make_loaders(
        bundle, args.batch_size, args.workers, args.seed
    )
    config = ModelConfig(
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        dropout=args.dropout,
        tau=args.tau,
        layer=args.layer,
        input_type=bundle.input_type,
        only_spike=args.only_spike,
    )
    model_type = PNDPairClassifier if bundle.pair else PNDSequenceClassifier
    model = model_type(config).to(device)
    optimizer = optimizer_for(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    save_json(output / "config.json", vars(args) | {"model": model.export_config()})
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = total = correct = 0
        for batch in train_loader:
            x, y, metadata = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths=metadata.get("lengths"))
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_accuracy": correct / total,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
        }
        history.append(row)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if validation["accuracy"] > best:
            best = validation["accuracy"]
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch, best)
        print(row, flush=True)
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    test = evaluate(model, test_loader, device)
    save_json(
        output / "metrics.json",
        {
            "best_epoch": payload["epoch"],
            "best_validation_accuracy": payload["best_validation_accuracy"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
        },
    )


if __name__ == "__main__":
    main()
