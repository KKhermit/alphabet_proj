from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST

# EMNIST "letters" split: labels 1–26 map to A–Z
ALPHABET_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _fix_emnist_orientation(t: torch.Tensor) -> torch.Tensor:
    # EMNIST images are stored transposed; this corrects to normal orientation.
    return t.permute(0, 2, 1).flip(1)


class AlphabetDataset(Dataset):
    """Wraps EMNIST 'letters' with orientation fix and optional augmentation."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        img_size: int = 64,
        augment: bool = False,
        mean: float = 0.5,
        std: float = 0.5,
    ) -> None:
        self.emnist = EMNIST(root=str(root), split="letters", train=train, download=True)
        self.transform = self._build_transform(img_size, augment, mean, std)

    def _build_transform(
        self, img_size: int, augment: bool, mean: float, std: float
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
            # This corrects them so the model sees normally-oriented letters,
            # matching the real letter crops it will receive at inference time.
            transforms.Lambda(_fix_emnist_orientation),
            transforms.Normalize(mean=[mean], std=[std]),
        ])
        if augment:
            ops.append(transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), value=1.0))
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.emnist)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.emnist[idx]
        tensor = self.transform(image)
        # EMNIST letters are 1-indexed (1=A … 26=Z); convert to 0-indexed
        label_idx = int(label) - 1
        return {
            "image": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
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

    train_ds = AlphabetDataset(root, train=True, img_size=img_size, augment=True, mean=mean, std=std)
    val_ds = AlphabetDataset(root, train=False, img_size=img_size, augment=False, mean=mean, std=std)

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
    """Preprocess a real letter crop for inference (no EMNIST orientation fix)."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    return transform(image)


__all__ = ["ALPHABET_CLASSES", "AlphabetDataset", "build_dataloaders", "preprocess_crop_for_inference"]
