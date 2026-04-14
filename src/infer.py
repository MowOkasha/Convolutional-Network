from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.config import load_config, resolve_device
from src.data.transforms import build_eval_transforms
from src.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to display.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.training.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", [])
    if not class_names:
        raise ValueError("Checkpoint is missing class_names. Re-train with this skeleton.")

    model = build_model(
        model_name=checkpoint.get("model_name", cfg.model.name),
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use eval transforms to match training-time normalization statistics.
    transform = build_eval_transforms(checkpoint.get("image_size", cfg.model.image_size))

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits.squeeze(0), dim=0)

    k = min(max(args.top_k, 1), len(class_names))
    top_probs, top_indices = torch.topk(probs, k=k)

    print("Top predictions:")
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        label = class_names[int(idx)]
        print(f"{rank}. {label}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
