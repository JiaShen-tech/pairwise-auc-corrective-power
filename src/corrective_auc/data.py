from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    """Dataset backed by a CSV manifest.

    Required columns are ``split`` and ``label`` plus either ``path`` (image)
    or ``text`` (NLP).  Multilabel manifests use the configured label columns;
    optional ``mask_<label>`` columns mark labels that should enter the loss.
    Paths are resolved under ``data_root``.
    """

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        modality: str,
        task_type: str,
        data_root: str | Path,
        transform: Any = None,
        label_columns: list[str] | None = None,
    ) -> None:
        frame = pd.read_csv(manifest)
        if "split" not in frame:
            raise ValueError("manifest must contain a split column")
        self.frame = frame.loc[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"manifest has no rows for split={split!r}")
        self.modality = modality
        self.task_type = task_type
        self.data_root = Path(data_root)
        self.transform = transform
        self.label_columns = list(label_columns or [])
        if task_type == "multilabel" and not self.label_columns:
            raise ValueError("multilabel experiments require label_columns")

    def __len__(self) -> int:
        return len(self.frame)

    def binary_labels(self) -> np.ndarray:
        if self.task_type != "binary":
            raise ValueError("binary_labels is only available for binary tasks")
        return self.frame["label"].to_numpy(dtype=np.int64)

    def class_labels(self) -> np.ndarray:
        if self.task_type == "multilabel":
            raise ValueError("class_labels is unavailable for multilabel tasks")
        return self.frame["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        if self.modality == "image":
            path = self.data_root / str(row["path"])
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            value: Any = self.transform(image) if self.transform is not None else image
            item = {"inputs": value}
        elif self.modality == "text":
            item = {"text": str(row["text"])}
            if "text_pair" in self.frame and not pd.isna(row["text_pair"]):
                item["text_pair"] = str(row["text_pair"])
        else:
            raise ValueError(f"unknown modality: {self.modality}")

        if self.task_type == "multilabel":
            labels = torch.tensor([float(row[name]) for name in self.label_columns], dtype=torch.float32)
            masks = []
            for name in self.label_columns:
                column = f"mask_{name}"
                masks.append(bool(row[column]) if column in self.frame else True)
            item["labels"] = labels
            item["mask"] = torch.tensor(masks, dtype=torch.bool)
        else:
            item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


class TextCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        first = [row["text"] for row in rows]
        has_pairs = any("text_pair" in row for row in rows)
        second = [row.get("text_pair", "") for row in rows] if has_pairs else None
        encoded = self.tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        encoded["labels"] = torch.stack([row["labels"] for row in rows])
        if "mask" in rows[0]:
            encoded["mask"] = torch.stack([row["mask"] for row in rows])
        return encoded


class StratifiedBatchSampler(Sampler[list[int]]):
    """Deterministic binary batches containing both classes."""

    def __init__(self, labels: np.ndarray, batch_size: int, positive_fraction: float, seed: int) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        self.positive = np.flatnonzero(labels == 1)
        self.negative = np.flatnonzero(labels == 0)
        if not self.positive.size or not self.negative.size:
            raise ValueError("binary training requires both classes")
        self.batch_size = int(batch_size)
        self.positive_per_batch = max(1, min(self.batch_size - 1, round(self.batch_size * positive_fraction)))
        self.negative_per_batch = self.batch_size - self.positive_per_batch
        self.seed = int(seed)
        self.epoch = 0
        self.num_batches = max(
            math.ceil(len(self.positive) / self.positive_per_batch),
            math.ceil(len(self.negative) / self.negative_per_batch),
        )

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw(pool: np.ndarray, amount: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(pool, size=amount, replace=amount > len(pool))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            batch = np.concatenate(
                [
                    self._draw(self.positive, self.positive_per_batch, rng),
                    self._draw(self.negative, self.negative_per_batch, rng),
                ]
            )
            rng.shuffle(batch)
            yield batch.tolist()


class CompleteClassBatchSampler(Sampler[list[int]]):
    """Draw the same number of examples from every class at every update."""

    def __init__(self, labels: np.ndarray, per_class: int, steps: int, seed: int) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        self.pools = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
        self.per_class = int(per_class)
        self.steps = int(steps)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batch = []
            for pool in self.pools.values():
                batch.extend(rng.choice(pool, self.per_class, replace=len(pool) < self.per_class))
            rng.shuffle(batch)
            yield [int(index) for index in batch]


def image_transforms(image_size: int, train: bool, style: str = "paper"):
    from torchvision import transforms

    if style == "simple":
        steps = [transforms.Resize((image_size, image_size))]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.ToTensor())
        return transforms.Compose(steps)
    resize = round(image_size / 0.875)
    if style == "medical":
        crop = transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size)
        steps = [transforms.Resize(resize), crop]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return transforms.Compose(steps)
    if style == "large":
        if train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    resize = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_loaders(
    config: dict[str, Any],
    manifest: str | Path,
    data_root: str | Path,
    seed: int,
    tokenizer: Any = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    modality = config["modality"]
    task_type = config["task_type"]
    label_columns = config.get("label_columns")
    image_size = int(config.get("image_size", 224))
    transform_style = str(config.get("transform_style", "paper"))
    train_transform = image_transforms(image_size, True, transform_style) if modality == "image" else None
    eval_transform = image_transforms(image_size, False, transform_style) if modality == "image" else None
    train_set = ManifestDataset(manifest, "train", modality, task_type, data_root, train_transform, label_columns)
    val_set = None if config.get("fixed_budget_test") else ManifestDataset(
        manifest, "val", modality, task_type, data_root, eval_transform, label_columns
    )
    test_set = ManifestDataset(manifest, "test", modality, task_type, data_root, eval_transform, label_columns)
    collate = TextCollator(tokenizer, int(config.get("max_length", 128))) if modality == "text" else None
    common = {
        "num_workers": int(config.get("num_workers", 4)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate,
    }
    batch_size = int(config.get("batch_size", config["eval_batch_size"]))
    if task_type == "binary":
        sampler = StratifiedBatchSampler(
            train_set.binary_labels(), batch_size, float(config.get("positive_fraction", 0.25)), seed
        )
        train_loader = DataLoader(train_set, batch_sampler=sampler, **common)
    elif task_type == "multiclass" and config.get("steps_per_epoch"):
        sampler = CompleteClassBatchSampler(
            train_set.class_labels(), int(config["per_class"]), int(config["steps_per_epoch"]), seed
        )
        train_loader = DataLoader(train_set, batch_sampler=sampler, **common)
    else:
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, **common)
    eval_batch_size = int(config.get("eval_batch_size", batch_size))
    val_loader = None if val_set is None else DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader

