from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, ImageFolder

from src.data.transforms import build_eval_transforms, build_train_transforms


def _build_imagefolder_datasets(cfg) -> tuple[ImageFolder, ImageFolder, ImageFolder | None, list[str]]:
    """
    Build datasets from folder structure:
    root/train/class_x/*.jpg, root/val/class_x/*.jpg, root/test/class_x/*.jpg (optional)
    """
    root = Path(cfg.dataset.root)
    train_root = root / cfg.dataset.train_dir
    val_root = root / cfg.dataset.val_dir
    test_root = root / cfg.dataset.test_dir

    if not train_root.exists():
        raise FileNotFoundError(f"Missing training folder: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(
            f"Missing validation folder: {val_root}. "
            "Create one or switch to dataset.name=cifar10 for quick start."
        )

    train_transform = build_train_transforms(cfg.model.image_size)
    eval_transform = build_eval_transforms(cfg.model.image_size)

    train_dataset = ImageFolder(train_root, transform=train_transform)
    val_dataset = ImageFolder(val_root, transform=eval_transform)

    # Test set is optional; if absent, evaluation script will use validation set.
    test_dataset = ImageFolder(test_root, transform=eval_transform) if test_root.exists() else None

    class_names = train_dataset.classes
    return train_dataset, val_dataset, test_dataset, class_names


def _build_cifar10_datasets(cfg) -> tuple[Subset, Subset, CIFAR10, list[str]]:
    """
    Build CIFAR-10 datasets.

    We split CIFAR-10 train into train/val so the training loop always has
    a real validation stage each epoch.
    """
    root = Path(cfg.dataset.root)
    train_transform = build_train_transforms(cfg.model.image_size)
    eval_transform = build_eval_transforms(cfg.model.image_size)

    train_full = CIFAR10(root=str(root), train=True, transform=train_transform, download=True)
    val_full = CIFAR10(root=str(root), train=True, transform=eval_transform, download=True)
    test_dataset = CIFAR10(root=str(root), train=False, transform=eval_transform, download=True)

    n_total = len(train_full)
    n_train = int(0.9 * n_total)

    # Fixed generator seed keeps split stable between runs.
    generator = torch.Generator().manual_seed(cfg.training.seed)
    shuffled = torch.randperm(n_total, generator=generator).tolist()
    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:]

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    class_names = train_full.classes
    return train_dataset, val_dataset, test_dataset, class_names


def build_dataloaders(cfg):
    """Create train/val/test dataloaders plus class names."""
    if cfg.dataset.name.lower() == "cifar10":
        train_dataset, val_dataset, test_dataset, class_names = _build_cifar10_datasets(cfg)
    else:
        train_dataset, val_dataset, test_dataset, class_names = _build_imagefolder_datasets(cfg)

    # pin_memory can speed up host->GPU transfers when CUDA is available.
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader, class_names
