"""Standalone MNIST, neuromorphic-event, and Speech Commands experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms

from pnd.model import ModelConfig, PNDSequenceClassifier, ResidualBlock
from pnd.training import evaluate, optimizer_for, save_checkpoint, save_json, set_seed, unpack_batch


class EventPatchClassifier(nn.Module):
    def __init__(self, config, patch_size):
        super().__init__()
        self.config = config
        self.patch_size = patch_size
        self.patch = nn.Conv2d(2, config.d_model, patch_size, stride=patch_size)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Linear(config.d_model, config.num_classes)

    def export_config(self):
        return vars(self.config) | {"frontend": "learned_event_patch", "patch_size": self.patch_size}

    def forward(self, x, lengths=None):
        batch, time, polarity, height, width = x.shape
        x = self.patch(x.reshape(batch * time, polarity, height, width))
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(batch, time * x.size(1), self.config.d_model).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x.mean(dim=-1))


class SpeechClassifier(nn.Module):
    def __init__(self, config, kernel_size=32, stride=500):
        super().__init__()
        self.config = config
        self.kernel_size = kernel_size
        self.stride = stride
        self.frontend = nn.Conv1d(1, config.d_model, kernel_size, stride=stride)
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.decoder = nn.Linear(config.d_model, config.num_classes)

    def export_config(self):
        return vars(self.config) | {"frontend": "learned_waveform_convolution", "kernel_size": self.kernel_size, "stride": self.stride}

    def forward(self, waveform, lengths=None):
        x = self.frontend(waveform.unsqueeze(1))
        for layer in self.layers:
            x = layer(x)
        return self.decoder(x.mean(dim=-1))


class SpeechFolder(Dataset):
    def __init__(self, root, subset, sample_rate=16000):
        import torchaudio

        self.torchaudio = torchaudio
        self.root = Path(root)
        self.sample_rate = sample_rate
        validation = set((self.root / "validation_list.txt").read_text().splitlines())
        testing = set((self.root / "testing_list.txt").read_text().splitlines())
        labels = sorted(path.name for path in self.root.iterdir() if path.is_dir() and not path.name.startswith("_"))
        self.label_map = {label: index for index, label in enumerate(labels)}
        self.files = []
        for label in labels:
            for path in sorted((self.root / label).glob("*.wav")):
                key = f"{label}/{path.name}"
                selected = (
                    (subset == "validation" and key in validation)
                    or (subset == "testing" and key in testing)
                    or (subset == "training" and key not in validation and key not in testing)
                )
                if selected:
                    self.files.append((path, self.label_map[label]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, label = self.files[index]
        waveform, rate = self.torchaudio.load(path)
        waveform = waveform.mean(0)
        if rate != self.sample_rate:
            waveform = self.torchaudio.functional.resample(waveform, rate, self.sample_rate)
        waveform = F.pad(waveform[: self.sample_rate], (0, max(0, self.sample_rate - waveform.numel())))
        return waveform, label


def loaders(args):
    if args.task == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten().unsqueeze(-1))])
        full = torchvision.datasets.MNIST(args.data_dir, train=True, transform=transform, download=args.download)
        test = torchvision.datasets.MNIST(args.data_dir, train=False, transform=transform, download=args.download)
        train, validation = random_split(full, [55000, 5000], generator=torch.Generator().manual_seed(args.seed))
        model = PNDSequenceClassifier(ModelConfig(1, 10, args.d_model, args.d_state, args.n_layers, args.dropout, args.tau, args.layer))
    elif args.task in {"nmnist", "dvs128"}:
        if args.task == "nmnist":
            from spikingjelly.datasets.n_mnist import NMNIST
            dataset_type, classes, size, patch, steps = NMNIST, 10, 34, 2, args.frames
            full = dataset_type(args.data_dir, train=True, data_type="frame", frames_number=steps, split_by="number")
            test = dataset_type(args.data_dir, train=False, data_type="frame", frames_number=steps, split_by="number")
        else:
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
            dataset_type, classes, size, patch, steps = DVS128Gesture, 11, 128, 16, args.frames
            full = dataset_type(args.data_dir, train=True, data_type="frame", frames_number=steps, split_by="number")
            test = dataset_type(args.data_dir, train=False, data_type="frame", frames_number=steps, split_by="number")
        validation_size = len(full) // 10
        train, validation = random_split(full, [len(full) - validation_size, validation_size], generator=torch.Generator().manual_seed(args.seed))
        config = ModelConfig(args.d_model, classes, args.d_model, args.d_state, args.n_layers, args.dropout, args.tau, args.layer)
        model = EventPatchClassifier(config, patch)
    else:
        train = SpeechFolder(args.data_dir, "training")
        validation = SpeechFolder(args.data_dir, "validation")
        test = SpeechFolder(args.data_dir, "testing")
        config = ModelConfig(args.d_model, len(train.label_map), args.d_model, args.d_state, args.n_layers, args.dropout, args.tau, args.layer)
        model = SpeechClassifier(config)
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    return model, DataLoader(train, shuffle=True, **common), DataLoader(validation, shuffle=False, **common), DataLoader(test, shuffle=False, **common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mnist", "nmnist", "dvs128", "speechcommands"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", default="pnd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_loader, validation_loader, test_loader = loaders(args)
    model = model.to(device)
    optimizer = optimizer_for(model, args.lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    save_json(output / "config.json", vars(args) | {"model": model.export_config()})
    best = -1.0
    for epoch in range(args.epochs):
        model.train(); total = correct = 0; loss_sum = 0.0
        for batch in train_loader:
            x, y, metadata = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, lengths=metadata.get("lengths"))
            loss = F.cross_entropy(logits, y); loss.backward(); optimizer.step()
            total += y.numel(); correct += (logits.argmax(-1) == y).sum().item(); loss_sum += loss.item() * y.numel()
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total, "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"]}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
        if validation["accuracy"] > best:
            best = validation["accuracy"]
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, epoch, best)
        print(row, flush=True)
    payload = torch.load(output / "best.pt", map_location=device, weights_only=False); model.load_state_dict(payload["model"])
    test = evaluate(model, test_loader, device)
    save_json(output / "metrics.json", {"best_epoch": payload["epoch"], "best_validation_accuracy": payload["best_validation_accuracy"], "test_loss": test["loss"], "test_accuracy": test["accuracy"]})


if __name__ == "__main__":
    main()
