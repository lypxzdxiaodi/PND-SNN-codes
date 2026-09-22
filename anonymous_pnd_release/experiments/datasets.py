"""Minimal standalone loaders for the datasets used in the paper."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
from PIL import Image

from pnd.training import seed_worker


@dataclass
class DataBundle:
    train: Dataset
    validation: Dataset
    test: Dataset
    input_dim: int
    num_classes: int
    input_type: str
    collate_fn: object = None
    pair: bool = False


class Vocabulary:
    def __init__(self, sequences, min_frequency=1):
        counts = {}
        for sequence in sequences:
            for token in sequence:
                counts[token] = counts.get(token, 0) + 1
        self.tokens = ["<pad>", "<unk>"] + sorted(
            token for token, count in counts.items() if count >= min_frequency
        )
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens, limit):
        unknown = self.indices["<unk>"]
        return torch.tensor(
            [self.indices.get(token, unknown) for token in tokens[:limit]],
            dtype=torch.long,
        )


class EncodedTextDataset(Dataset):
    def __init__(self, rows, vocabulary, limit, pair=False):
        self.rows = rows
        self.vocabulary = vocabulary
        self.limit = limit
        self.pair = pair

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.pair:
            first, second, label = row
            return (
                self.vocabulary.encode(first, self.limit),
                self.vocabulary.encode(second, self.limit),
            ), label
        tokens, label = row
        return self.vocabulary.encode(tokens, self.limit), label


def token_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences, torch.tensor(labels), {"lengths": lengths}


def pair_collate(batch):
    pairs, labels = zip(*batch)
    first, second = zip(*pairs)
    first = pad_sequence(first, batch_first=True, padding_value=0)
    second = pad_sequence(second, batch_first=True, padding_value=0)
    length = max(first.size(1), second.size(1))
    first = torch.nn.functional.pad(first, (0, length - first.size(1)))
    second = torch.nn.functional.pad(second, (0, length - second.size(1)))
    return (first, second), torch.tensor(labels), {}


def _read_listops(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            tokens = row["Source"].translate(
                {ord("]"): ord("X"), ord("("): None, ord(")"): None}
            ).split()
            rows.append((tokens, int(row["Target"])))
    return rows


def listops_bundle(data_dir, length=2048):
    root = Path(data_dir)
    train = _read_listops(root / "basic_train.tsv")
    validation = _read_listops(root / "basic_val.tsv")
    test = _read_listops(root / "basic_test.tsv")
    vocabulary = Vocabulary(tokens for tokens, _ in train)
    wrap = lambda rows: EncodedTextDataset(rows, vocabulary, length)
    return DataBundle(
        wrap(train), wrap(validation), wrap(test), len(vocabulary), 10, "tokens", token_collate
    )


def _read_aan(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for label, _, _, first, second in reader:
            rows.append((list(first), list(second), int(label)))
    return rows


def aan_bundle(data_dir, length=4000):
    root = Path(data_dir)
    train = _read_aan(root / "new_aan_pairs.train.tsv")
    validation = _read_aan(root / "new_aan_pairs.eval.tsv")
    test = _read_aan(root / "new_aan_pairs.test.tsv")
    vocabulary = Vocabulary(chain((a for a, _, _ in train), (b for _, b, _ in train)))
    wrap = lambda rows: EncodedTextDataset(rows, vocabulary, length, pair=True)
    return DataBundle(
        wrap(train), wrap(validation), wrap(test), len(vocabulary), 2, "tokens", pair_collate, True
    )


def imdb_bundle(data_dir, length=4096, seed=42):
    from datasets import load_dataset

    dataset = load_dataset("imdb", cache_dir=str(data_dir))
    train_rows = [(list(row["text"]), int(row["label"])) for row in dataset["train"]]
    test_rows = [(list(row["text"]), int(row["label"])) for row in dataset["test"]]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_rows), generator=generator).tolist()
    validation_size = len(train_rows) // 10
    validation = [train_rows[index] for index in order[:validation_size]]
    train = [train_rows[index] for index in order[validation_size:]]
    vocabulary = Vocabulary((tokens for tokens, _ in train), min_frequency=15)
    wrap = lambda rows: EncodedTextDataset(rows, vocabulary, length)
    return DataBundle(
        wrap(train), wrap(validation), wrap(test_rows), len(vocabulary), 2, "tokens", token_collate
    )


class PathfinderDataset(Dataset):
    def __init__(self, data_dir, center=False):
        self.root = Path(data_dir)
        self.center = center
        self.samples = []
        metadata_root = self.root / "curv_contour_length_14" / "metadata"
        for metadata_file in sorted(metadata_root.glob("*.npy"), key=lambda path: int(path.stem)):
            for line in metadata_file.read_text().splitlines():
                fields = line.split()
                image = Path("curv_contour_length_14") / fields[0] / fields[1]
                self.samples.append((image, int(fields[3])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with (self.root / path).open("rb") as handle:
            image = transforms.functional.to_tensor(Image.open(handle).convert("L"))
        if self.center:
            image = (image - 0.5) / 0.5
        return image.flatten().unsqueeze(-1), label


def pathfinder_bundle(data_dir, seed=42, center=False):
    dataset = PathfinderDataset(data_dir, center=center)
    validation = len(dataset) // 10
    test = len(dataset) // 10
    train = len(dataset) - validation - test
    splits = random_split(
        dataset, [train, validation, test], generator=torch.Generator().manual_seed(seed)
    )
    return DataBundle(*splits, input_dim=1, num_classes=2, input_type="continuous")


def scifar_bundle(data_dir, seed=42, grayscale=True, download=False):
    transform_list = [transforms.Grayscale()] if grayscale else []
    transform_list += [transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten(1).T)]
    transform = transforms.Compose(transform_list)
    full_train = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform, download=download)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, transform=transform, download=download)
    validation_size = len(full_train) // 10
    train, validation = random_split(
        full_train,
        [len(full_train) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return DataBundle(train, validation, test, 1 if grayscale else 3, 10, "continuous")


def make_loaders(bundle, batch_size, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=bundle.collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(bundle.train, shuffle=True, generator=generator, **common),
        DataLoader(bundle.validation, shuffle=False, **common),
        DataLoader(bundle.test, shuffle=False, **common),
    )
