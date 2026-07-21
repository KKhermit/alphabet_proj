from __future__ import annotations

import math
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report

from alphabet.dataset import ALPHABET_CLASSES, build_dataloaders
from alphabet.export import export_from_config
from alphabet.model import build_model, count_parameters
from alphabet.utils import ensure_dir, load_yaml, save_json, set_seed, write_csv


class EarlyStopper:
    def __init__(self, patience: int = 8, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _forward_context(device: str, amp_enabled: bool):
    if amp_enabled and device.startswith("cuda"):
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _forward_context(device, amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.cpu().numpy())
        losses.append(float(loss.item()))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(
        y_true, y_pred,
        target_names=ALPHABET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {"accuracy": accuracy, "report": report}
    return float(np.mean(losses)) if losses else math.nan, metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def train_from_config(config_path: str | Path, device: str | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    save_dir = ensure_dir(cfg["train"].get("save_dir", "outputs/checkpoints"))

    train_loader, val_loader = build_dataloaders(cfg, device=device)
    model = build_model(cfg["model"]).to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training on {device} | "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    early = EarlyStopper(patience=int(cfg["train"].get("early_patience", 8)), mode="max")

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        score = val_metrics["accuracy"]
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={score:.4f}"
        )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(score),
        }
        history.append(row)

        save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, epoch, cfg, row)

        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, epoch, cfg, row)
            save_json(val_metrics, save_dir / "best_metrics.json")

        if early.step(score):
            print(f"Early stopping at epoch {epoch}.")
            break

    write_csv(history, save_dir / "train_log.csv")

    if best_state is not None:
        model.load_state_dict(best_state)

    _, final_metrics = validate(model, val_loader, criterion, device)
    save_json(final_metrics, save_dir / "final_metrics.json")

    summary = {
        "device": device,
        "params": count_parameters(model),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "best_val_accuracy": best_score,
        "final_metrics": final_metrics,
    }
    save_json(summary, save_dir / "summary.json")

    if bool(cfg.get("export", {}).get("after_train", True)):
        export_from_config(config_path, save_dir / "best.pt", device=device)

    return summary


__all__ = ["train_from_config"]
