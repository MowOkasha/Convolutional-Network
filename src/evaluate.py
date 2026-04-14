from __future__ import annotations

import argparse

import torch
from torch import nn

from src.config import load_config, resolve_device
from src.data.build import build_dataloaders
from src.engine.eval_loop import evaluate
from src.models.factory import build_model
from src.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.training.seed)
    device = resolve_device(cfg.training.device)

    _, val_loader, test_loader, class_names = build_dataloaders(cfg)
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_name = "test" if test_loader is not None else "val"

    model = build_model(
        model_name=cfg.model.name,
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_names = checkpoint.get("class_names", class_names)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, eval_loader, criterion, device)

    print(f"{eval_name} loss: {metrics['loss']:.4f}")
    print(f"{eval_name} acc:  {metrics['acc']:.4f}")

    # Optional pretty report if scikit-learn exists.
    try:
        from sklearn.metrics import classification_report

        print("\nClassification report:\n")
        print(
            classification_report(
                metrics["targets"],
                metrics["predictions"],
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )
    except Exception:
        print("scikit-learn report skipped (package missing or report generation failed).")


if __name__ == "__main__":
    main()
