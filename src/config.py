from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml


@dataclass
class DatasetConfig:
    # Dataset mode: "imagefolder" expects train/val/test folders;
    # "cifar10" downloads and uses torchvision's built-in dataset.
    name: str = "imagefolder"
    root: str = "data/raw"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str = "test"


@dataclass
class ModelConfig:
    # Model name (for timm or fallback torchvision mapping).
    name: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    # device can be: "auto", "cpu", "cuda", "mps".
    device: str = "auto"


@dataclass
class PathsConfig:
    output_dir: str = "outputs"


@dataclass
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(config_path: str | Path) -> ProjectConfig:
    """Load YAML config into dataclasses so field names are explicit and typed."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ProjectConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        paths=PathsConfig(**raw.get("paths", {})),
    )


def resolve_device(preferred: str) -> str:
    """
    Pick the best available hardware device.

    - auto: cuda -> mps -> cpu
    - explicit: trust user input but validate common options
    """
    preferred = preferred.lower().strip()
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    if preferred in {"cpu", "cuda", "mps"}:
        return preferred

    raise ValueError("device must be one of: auto, cpu, cuda, mps")
