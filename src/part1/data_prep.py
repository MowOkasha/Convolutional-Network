from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split


FER_LABEL_TO_NAME = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}
FER_NAME_TO_LABEL = {name: label for label, name in FER_LABEL_TO_NAME.items()}


@dataclass
class PrepConfig:
    source: str
    hf_dataset_id: str
    cache_dir: str
    selected_labels: list[str]
    output_root: str
    image_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    remove_exact_duplicates: bool
    seed: int
    max_images_per_class: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Part 1 emotion dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to part1_data_prep YAML.")
    return parser.parse_args()


def load_config(path: str) -> PrepConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    ratio_sum = raw["split"]["train_ratio"] + raw["split"]["val_ratio"] + raw["split"]["test_ratio"]
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    selected_labels = [str(x) for x in raw["dataset"]["selected_labels"]]

    return PrepConfig(
        source=str(raw["dataset"]["source"]),
        hf_dataset_id=str(raw["dataset"]["hf_dataset_id"]),
        cache_dir=str(raw["dataset"]["cache_dir"]),
        selected_labels=selected_labels,
        output_root=raw["output"]["root_dir"],
        image_size=int(raw["output"]["image_size"]),
        train_ratio=float(raw["split"]["train_ratio"]),
        val_ratio=float(raw["split"]["val_ratio"]),
        test_ratio=float(raw["split"]["test_ratio"]),
        remove_exact_duplicates=bool(raw["preprocessing"]["remove_exact_duplicates"]),
        seed=int(raw["runtime"]["seed"]),
        max_images_per_class=(
            None
            if raw["runtime"]["max_images_per_class"] is None
            else int(raw["runtime"]["max_images_per_class"])
        ),
    )


def fetch_fer2013_hf(dataset_id: str, cache_dir: str) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Download FER2013 from Hugging Face datasets.

    We merge all available splits first, then create assignment-required
    70/20/10 splits ourselves.
    """
    ds = load_dataset(dataset_id, cache_dir=cache_dir)

    images: list[np.ndarray] = []
    labels: list[str] = []

    for split_name in ds.keys():
        split = ds[split_name]
        for row in split:
            image = np.asarray(row["image"], dtype=np.uint8)
            if image.ndim == 3:
                image = image[:, :, 0]

            raw_label = row.get("emotion", row.get("emotion_name"))
            if isinstance(raw_label, str):
                if raw_label.isdigit():
                    label = raw_label
                else:
                    lowered = raw_label.strip().lower()
                    if lowered not in FER_NAME_TO_LABEL:
                        raise ValueError(f"Unknown emotion label in dataset: {raw_label}")
                    label = FER_NAME_TO_LABEL[lowered]
            else:
                label = str(int(raw_label))

            images.append(image)
            labels.append(label)

    return images, np.asarray(labels, dtype=str)


def choose_classes(
    images: list[np.ndarray],
    labels: np.ndarray,
    selected_labels: list[str],
    max_images_per_class: int | None,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Select the required 4 classes and optionally cap samples per class."""
    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    for label in selected_labels:
        class_indices = np.where(labels == label)[0]
        if max_images_per_class is not None and len(class_indices) > max_images_per_class:
            class_indices = rng.choice(class_indices, size=max_images_per_class, replace=False)
        selected_indices.extend(class_indices.tolist())

    selected_indices = np.array(selected_indices, dtype=np.int64)
    selected_images = [images[int(i)] for i in selected_indices.tolist()]
    selected_labels_np = labels[selected_indices]
    return selected_images, selected_labels_np


def gray_to_rgb_512(gray: np.ndarray, image_size: int) -> np.ndarray:
    """Convert grayscale face image into resized 512x512x3 image array."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    pil_image = Image.fromarray(rgb)
    pil_image = pil_image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def deduplicate_images(images: list[np.ndarray], labels: list[str]) -> tuple[list[np.ndarray], list[str], int]:
    """Remove exact duplicates by image hash so redundant samples are dropped."""
    seen_hashes: set[str] = set()
    unique_images: list[np.ndarray] = []
    unique_labels: list[str] = []
    removed = 0

    for image, label in zip(images, labels):
        image_hash = hashlib.sha1(image.tobytes()).hexdigest()
        if image_hash in seen_hashes:
            removed += 1
            continue

        seen_hashes.add(image_hash)
        unique_images.append(image)
        unique_labels.append(label)

    return unique_images, unique_labels, removed


def stratified_split(
    images: list[np.ndarray],
    labels: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    """Split into train/val/test with exact assignment ratio intent: 70/20/10."""
    del test_ratio  # We use train+val remainder relation below.

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images,
        labels,
        test_size=0.1,
        random_state=seed,
        stratify=labels,
    )

    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def reset_output_dirs(root: Path, class_names: list[str]) -> None:
    """Clear old split folders so reruns do not mix old and new files."""
    for split in ["train", "val", "test"]:
        split_root = root / split
        if split_root.exists():
            shutil.rmtree(split_root)

        for class_name in class_names:
            (split_root / class_name).mkdir(parents=True, exist_ok=True)


def save_split(root: Path, split: str, images: list[np.ndarray], labels: list[str]) -> None:
    """Write images to data/raw/split/class folders."""
    counters: dict[str, int] = {}

    for image, label in zip(images, labels):
        class_name = FER_LABEL_TO_NAME[label]
        counters.setdefault(class_name, 0)
        counters[class_name] += 1

        filename = f"{split}_{class_name}_{counters[class_name]:05d}.jpg"
        target = root / split / class_name / filename
        Image.fromarray(image).save(target, format="JPEG", quality=95)


def count_per_class(labels: list[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for label in labels:
        class_name = FER_LABEL_TO_NAME[label]
        summary[class_name] = summary.get(class_name, 0) + 1
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.source != "huggingface_fer2013":
        raise ValueError("Unsupported dataset source. Expected 'huggingface_fer2013'.")

    images_gray, labels = fetch_fer2013_hf(dataset_id=cfg.hf_dataset_id, cache_dir=cfg.cache_dir)
    images_gray, labels = choose_classes(
        images_gray,
        labels,
        selected_labels=cfg.selected_labels,
        max_images_per_class=cfg.max_images_per_class,
        seed=cfg.seed,
    )

    labels_list: list[str] = labels.tolist()

    # Deduplicate before resizing to avoid unnecessary processing.
    removed_duplicates = 0
    if cfg.remove_exact_duplicates:
        images_gray, labels_list, removed_duplicates = deduplicate_images(images_gray, labels_list)

    images_512: list[np.ndarray] = [gray_to_rgb_512(img, cfg.image_size) for img in images_gray]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = stratified_split(
        images_512,
        labels_list,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    selected_class_names = [FER_LABEL_TO_NAME[label] for label in cfg.selected_labels]
    output_root = Path(cfg.output_root)
    reset_output_dirs(output_root, selected_class_names)

    save_split(output_root, "train", x_train, y_train)
    save_split(output_root, "val", x_val, y_val)
    save_split(output_root, "test", x_test, y_test)

    metadata = {
        "dataset": "FER2013 (huggingface)",
        "source": cfg.source,
        "hf_dataset_id": cfg.hf_dataset_id,
        "selected_labels": cfg.selected_labels,
        "selected_classes": selected_class_names,
        "image_size": cfg.image_size,
        "split_ratios": {
            "train": cfg.train_ratio,
            "val": cfg.val_ratio,
            "test": cfg.test_ratio,
        },
        "duplicates_removed": removed_duplicates,
        "counts": {
            "train": count_per_class(y_train),
            "val": count_per_class(y_val),
            "test": count_per_class(y_test),
        },
    }

    metadata_path = output_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Data preparation complete.")
    print(f"Output root: {output_root}")
    print(f"Metadata: {metadata_path}")
    print("Counts:")
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
