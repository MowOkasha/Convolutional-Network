from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import resolve_device
from src.part1.model2_cnn import Model2CNN
from src.utils.seeding import set_seed


@dataclass
class Model2Config:
    data_root: str
    classes: list[str]
    image_size: int
    max_train_per_class: int | None
    max_val_per_class: int | None
    max_test_per_class: int | None
    kernel_sizes: list[int]
    channels: list[int]
    hidden_units: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Part 1 Model 2 CNN.")
    parser.add_argument("--config", type=str, required=True, help="Path to part1_model2 YAML.")
    return parser.parse_args()


def load_config(path: str) -> Model2Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return Model2Config(
        data_root=raw["data"]["root_dir"],
        classes=list(raw["data"]["classes"]),
        image_size=int(raw["data"]["image_size"]),
        max_train_per_class=(
            None
            if raw["data"].get("max_train_per_class") is None
            else int(raw["data"]["max_train_per_class"])
        ),
        max_val_per_class=(
            None
            if raw["data"].get("max_val_per_class") is None
            else int(raw["data"]["max_val_per_class"])
        ),
        max_test_per_class=(
            None
            if raw["data"].get("max_test_per_class") is None
            else int(raw["data"]["max_test_per_class"])
        ),
        kernel_sizes=[int(x) for x in raw["model"]["kernel_sizes"]],
        channels=[int(x) for x in raw["model"]["channels"]],
        hidden_units=int(raw["model"]["hidden_units"]),
        epochs=int(raw["training"]["epochs"]),
        batch_size=int(raw["training"]["batch_size"]),
        learning_rate=float(raw["training"]["learning_rate"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        num_workers=int(raw["training"]["num_workers"]),
        seed=int(raw["training"]["seed"]),
        device=str(raw["training"]["device"]),
        output_dir=raw["output"]["dir"],
    )


def cap_per_class(dataset: ImageFolder, max_per_class: int | None, seed: int):
    """Optionally cap samples per class for faster local experimentation."""
    if max_per_class is None:
        return dataset

    rng = np.random.default_rng(seed)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    selected_indices: list[int] = []

    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        if len(class_indices) > max_per_class:
            class_indices = rng.choice(class_indices, size=max_per_class, replace=False)
        selected_indices.extend(class_indices.tolist())

    selected_indices.sort()
    return Subset(dataset, selected_indices)


def make_dataloaders(cfg: Model2Config):
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ]
    )

    root = Path(cfg.data_root)
    train_ds = ImageFolder(root / "train", transform=transform)
    val_ds = ImageFolder(root / "val", transform=transform)

    test_root = root / "test"
    test_ds = ImageFolder(test_root, transform=transform) if test_root.exists() else None

    class_names = train_ds.classes

    train_ds_capped = cap_per_class(train_ds, cfg.max_train_per_class, cfg.seed)
    val_ds_capped = cap_per_class(val_ds, cfg.max_val_per_class, cfg.seed)
    test_ds_capped = None if test_ds is None else cap_per_class(test_ds, cfg.max_test_per_class, cfg.seed)

    # Sanity check: assignment asks for four classes.
    if len(class_names) != 4:
        print(f"Warning: expected 4 classes, found {len(class_names)} classes: {class_names}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds_capped,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds_capped,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_ds_capped is not None:
        test_loader = DataLoader(
            test_ds_capped,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader, class_names


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        probs = model(images)
        # NLLLoss expects log-probabilities.
        log_probs = torch.log(probs + 1e-8)
        loss = criterion(log_probs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

        preds = probs.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())

    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    output_dir = Path(cfg.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = make_dataloaders(cfg)
    model = Model2CNN(
        num_classes=len(class_names),
        input_size=cfg.image_size,
        kernel_sizes=cfg.kernel_sizes,
        channels=cfg.channels,
        hidden_units=cfg.hidden_units,
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": cfg.image_size,
                    "kernel_sizes": cfg.kernel_sizes,
                    "channels": cfg.channels,
                    "hidden_units": cfg.hidden_units,
                    "best_val_acc": best_val_acc,
                },
                checkpoints_dir / "best.pt",
            )

    # Save history CSV for later plotting and report usage.
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Evaluate the best checkpoint on test split if available, otherwise on validation split.
    checkpoint = torch.load(checkpoints_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_loader = test_loader if test_loader is not None else val_loader
    eval_split = "test" if test_loader is not None else "val"
    with torch.no_grad():
        eval_loss, eval_acc = run_epoch(model, eval_loader, optimizer=None, criterion=criterion, device=device)

    metrics = {
        "best_val_acc": best_val_acc,
        f"{eval_split}_loss": eval_loss,
        f"{eval_split}_acc": eval_acc,
        "classes": class_names,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Model 2 complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
