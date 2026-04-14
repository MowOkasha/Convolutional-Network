from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import load_config, resolve_device
from src.data.build import build_dataloaders
from src.engine.eval_loop import evaluate
from src.engine.train_loop import train_one_epoch
from src.models.factory import build_model
from src.utils.io import ensure_dir, save_json
from src.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Reproducibility first so data split and augmentation randomness are stable.
    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)

    output_dir = ensure_dir(cfg.paths.output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    train_loader, val_loader, _, class_names = build_dataloaders(cfg)

    model = build_model(
        model_name=cfg.model.name,
        num_classes=len(class_names),
        pretrained=cfg.model.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    best_val_acc = -1.0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
        )

        # Save best model by validation accuracy.
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_path = checkpoints_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": cfg.model.name,
                    "image_size": cfg.model.image_size,
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )

    # Keep class names as plain JSON too, useful for quick inspection.
    save_json(class_names, output_dir / "class_names.json")

    # Write CSV history so you can quickly plot curves in spreadsheet or Python.
    history_path = Path(output_dir) / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {checkpoints_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
