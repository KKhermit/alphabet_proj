from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST

# 37 classes: 0-9 (indices 0-9), A-Z (indices 10-35), blank (index 36)
CHAR_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["blank"]
BLANK_CLASS_INDEX = 36

# Keep for backwards compatibility
ALPHABET_CLASSES = CHAR_CLASSES


def _fix_emnist_orientation(t: torch.Tensor) -> torch.Tensor:
    # EMNIST images are stored transposed relative to natural writing orientation.
    # The correct fix is a plain transpose (permute spatial dims), not a rotation.
    return t.permute(0, 2, 1)


def _build_transform(
    img_size: int, augment: bool, mean: float, std: float
) -> transforms.Compose:
    ops: list[Any] = [transforms.Grayscale(num_output_channels=1)]
    if augment:
        ops.extend([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.88, 1.12),
                shear=8,
                fill=255,
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25, fill=255),
            transforms.ColorJitter(brightness=0.20, contrast=0.20),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ])
    ops.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # EMNIST images are stored transposed relative to normal orientation.
        transforms.Lambda(_fix_emnist_orientation),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    if augment:
        ops.append(transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), value=1.0))
    return transforms.Compose(ops)


class _EMNISTSplitDataset(Dataset):
    """Wraps a single EMNIST split with a label-remapping function."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        train: bool,
        label_fn,  # callable: raw_label -> class_index
        transform: transforms.Compose,
    ) -> None:
        self.emnist = EMNIST(root=str(root), split=split, train=train, download=True)
        self.label_fn = label_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.emnist)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.emnist[idx]
        return {
            "image": self.transform(image),
            "label": torch.tensor(self.label_fn(int(label)), dtype=torch.long),
        }


def _build_combined_dataset(
    root: str | Path,
    train: bool,
    img_size: int,
    augment: bool,
    mean: float,
    std: float,
    blank_size: int = 10000,
) -> ConcatDataset:
    """
    Concatenates EMNIST 'digits', 'letters', and synthetic blanks into a 37-class dataset.

    Digit labels (0-9) → indices 0-9.
    Letter labels (1-26, A-Z) → indices 10-35.
    Blank (synthetic white images) → index 36.
    """
    transform = _build_transform(img_size, augment, mean, std)
    digit_ds = _EMNISTSplitDataset(root, "digits", train, lambda l: l, transform)
    letter_ds = _EMNISTSplitDataset(root, "letters", train, lambda l: l - 1 + 10, transform)
    blank_ds = BlankDataset(blank_size, img_size, mean, std, augment=augment)
    return ConcatDataset([digit_ds, letter_ds, blank_ds])


class BlankDataset(Dataset):
    """Synthetic white images representing blank/empty boxes (class index 36)."""

    def __init__(self, size: int, img_size: int, mean: float, std: float, augment: bool = False) -> None:
        self.size = size
        self.img_size = img_size
        # At inference, a blank box is white paper (1.0) → polarity inversion (1-arr) → 0.0 → normalize → -1.0.
        # Training blanks must match that so the model sees the same representation.
        self.blank_val = (0.0 - mean) / std
        # In augment mode, add noise scaled to normalized space to simulate scan noise
        self.noise_std = 0.05 / std if augment else 0.0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _: int) -> dict[str, Any]:
        img = torch.full((1, self.img_size, self.img_size), self.blank_val)
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std
        return {
            "image": img,
            "label": torch.tensor(BLANK_CLASS_INDEX, dtype=torch.long),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"image": images, "label": labels}


def build_dataloaders(cfg: dict[str, Any], device: str = "cpu") -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    img_size = int(model_cfg.get("img_size", 64))
    mean = float(model_cfg.get("mean", 0.5))
    std = float(model_cfg.get("std", 0.5))
    root = data_cfg.get("root", "data/emnist")
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 4))
    pin_memory = device.startswith("cuda")

    blank_train = int(data_cfg.get("blank_train", 10000))
    blank_val = int(data_cfg.get("blank_val", 2000))

    train_ds = _build_combined_dataset(root, train=True, img_size=img_size, augment=True, mean=mean, std=std, blank_size=blank_train)
    val_ds = _build_combined_dataset(root, train=False, img_size=img_size, augment=False, mean=mean, std=std, blank_size=blank_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=False,
    )
    return train_loader, val_loader


def preprocess_crop_for_inference(
    image: Any,  # PIL Image
    img_size: int = 64,
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    """Preprocess a real crop for inference (no EMNIST orientation fix)."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    return transform(image)


__all__ = [
    "CHAR_CLASSES",
    "ALPHABET_CLASSES",
    "build_dataloaders",
    "preprocess_crop_for_inference",
]
