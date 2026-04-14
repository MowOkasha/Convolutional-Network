from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


@dataclass
class Model1Config:
    data_root: str
    classes: list[str]
    activation: str
    pooling_type: str
    feature_dim: int
    kmeans_max_iter: int
    kmeans_tol: float
    seed: int
    max_train_per_class: int
    max_val_per_class: int
    max_test_per_class: int
    output_dir: str


class ConvLayer:
    """
    Convolution layer implemented from scratch with NumPy.

    Requirement mapping:
    - Supports random initialization by shape.
    - Supports explicit initialization with provided 3D/4D weight matrix.
    """

    def __init__(
        self,
        num_filters: int,
        filter_size: int,
        input_channels: int,
        filters: np.ndarray | None = None,
    ) -> None:
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels

        if filters is None:
            rng = np.random.default_rng(0)
            self.filters = rng.normal(
                loc=0.0,
                scale=0.05,
                size=(num_filters, filter_size, filter_size, input_channels),
            ).astype(np.float32)
        else:
            filters = np.asarray(filters, dtype=np.float32)
            expected = (num_filters, filter_size, filter_size, input_channels)
            if filters.shape != expected:
                raise ValueError(f"Expected filter shape {expected}, got {filters.shape}")
            self.filters = filters

    @classmethod
    def random_init(
        cls,
        num_filters: int,
        filter_size: int,
        input_channels: int,
        seed: int,
    ) -> "ConvLayer":
        """Initialization method 1: random filters from shape parameters."""
        rng = np.random.default_rng(seed)
        filters = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(num_filters, filter_size, filter_size, input_channels),
        ).astype(np.float32)
        return cls(num_filters, filter_size, input_channels, filters=filters)

    @classmethod
    def from_weights(cls, weights: np.ndarray) -> "ConvLayer":
        """Initialization method 2: explicit user-provided filter matrix."""
        if weights.ndim != 4:
            raise ValueError("weights must be a 4D array [num_filters, k, k, channels]")
        n, k1, k2, c = weights.shape
        if k1 != k2:
            raise ValueError("filters must be square")
        return cls(num_filters=n, filter_size=k1, input_channels=c, filters=weights)

    def iterate_regions(self, image: np.ndarray):
        """Yield every filter-sized patch in raster order over the full image."""
        h, w, _ = image.shape
        k = self.filter_size
        for i in range(h - k + 1):
            for j in range(w - k + 1):
                yield image[i : i + k, j : j + k, :], i, j

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Run a valid convolution forward pass.

        This implementation keeps the algorithm from scratch while using NumPy vectorization
        to keep execution practical for 512x512 images.
        """
        if image.ndim != 3 or image.shape[2] != self.input_channels:
            raise ValueError(
                f"Expected input shape [H,W,{self.input_channels}], got {image.shape}"
            )

        k = self.filter_size
        windows = np.lib.stride_tricks.sliding_window_view(
            image,
            (k, k, self.input_channels),
        )
        windows = np.squeeze(windows, axis=2)  # [H-k+1, W-k+1, k, k, C]

        output = np.tensordot(
            windows,
            self.filters,
            axes=([2, 3, 4], [1, 2, 3]),
        )  # [H-k+1, W-k+1, num_filters]

        return output.astype(np.float32)


class PoolingLayer:
    """Pooling layer from scratch with configurable type and window size."""

    def __init__(self, pool_size: int = 2, pooling_type: str = "max") -> None:
        self.pool_size = pool_size
        pooling_type = pooling_type.lower().strip()
        if pooling_type not in {"max", "average"}:
            raise ValueError("pooling_type must be 'max' or 'average'")
        self.pooling_type = pooling_type

    def iterate_regions(self, feature_map: np.ndarray):
        """Yield non-overlapping pooling windows over feature maps."""
        h, w, _ = feature_map.shape
        p = self.pool_size
        h_out = h // p
        w_out = w // p
        for i in range(h_out):
            for j in range(w_out):
                yield feature_map[i * p : (i + 1) * p, j * p : (j + 1) * p, :], i, j

    def forward(self, feature_map: np.ndarray) -> np.ndarray:
        """Run pooling forward pass with stride equal to window size."""
        p = self.pool_size
        h, w, c = feature_map.shape

        h_out = h // p
        w_out = w // p
        clipped = feature_map[: h_out * p, : w_out * p, :]

        reshaped = clipped.reshape(h_out, p, w_out, p, c).transpose(0, 2, 1, 3, 4)
        if self.pooling_type == "max":
            pooled = reshaped.max(axis=(2, 3))
        else:
            pooled = reshaped.mean(axis=(2, 3))

        return pooled.astype(np.float32)


def simple_activation(x: np.ndarray, kind: str = "relu") -> np.ndarray:
    """Simple activation after pooling as required in Model 1."""
    kind = kind.lower().strip()
    if kind == "relu":
        return np.maximum(0.0, x)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if kind == "tanh":
        return np.tanh(x)
    raise ValueError("activation must be one of: relu, sigmoid, tanh")


class ConvolutionBlock:
    """Conv -> Pool -> Activation."""

    def __init__(self, conv: ConvLayer, pool: PoolingLayer, activation: str) -> None:
        self.conv = conv
        self.pool = pool
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv.forward(x)
        x = self.pool.forward(x)
        x = simple_activation(x, self.activation)
        return x


def build_predefined_filters(input_channels: int) -> np.ndarray:
    """
    Build 5 predefined filters (3x3xC).

    The base kernels are common edge/texture operators and are expanded to the
    required channel count so they can be applied without training.
    """
    base_kernels = np.array(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # blur
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],  # sharpen
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # sobel-x
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # sobel-y
            [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],  # emboss
        ],
        dtype=np.float32,
    )

    filters = []
    for kernel in base_kernels:
        expanded = np.repeat(kernel[:, :, None], input_channels, axis=2)
        expanded = expanded / max(input_channels, 1)
        filters.append(expanded)

    return np.stack(filters, axis=0).astype(np.float32)


def downsample_to_vector(feature_map: np.ndarray, output_dim: int = 128) -> np.ndarray:
    """Flatten and downsample to the required 1x128 vector using interpolation."""
    flat = feature_map.reshape(-1).astype(np.float32)
    if len(flat) == output_dim:
        return flat

    src = np.arange(len(flat), dtype=np.float32)
    dst = np.linspace(0, len(flat) - 1, output_dim, dtype=np.float32)
    reduced = np.interp(dst, src, flat)
    return reduced.astype(np.float32)


class ScratchConvNet:
    """Three convolution blocks -> flatten -> downsample to 128 features."""

    def __init__(self, activation: str, pooling_type: str) -> None:
        block1 = ConvolutionBlock(
            conv=ConvLayer.from_weights(build_predefined_filters(input_channels=3)),
            pool=PoolingLayer(pool_size=2, pooling_type=pooling_type),
            activation=activation,
        )
        block2 = ConvolutionBlock(
            conv=ConvLayer.from_weights(build_predefined_filters(input_channels=5)),
            pool=PoolingLayer(pool_size=2, pooling_type=pooling_type),
            activation=activation,
        )
        block3 = ConvolutionBlock(
            conv=ConvLayer.from_weights(build_predefined_filters(input_channels=5)),
            pool=PoolingLayer(pool_size=2, pooling_type=pooling_type),
            activation=activation,
        )
        self.blocks = [block1, block2, block3]

    def extract_features(self, image: np.ndarray, output_dim: int) -> np.ndarray:
        x = image
        for block in self.blocks:
            x = block.forward(x)
        return downsample_to_vector(x, output_dim=output_dim)


class KMeansNumpy:
    """Simple K-means implementation from scratch using NumPy."""

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n_samples = x.shape[0]

        chosen = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = x[chosen].copy()

        for _ in range(self.max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)

            new_centroids = np.zeros_like(centroids)
            for cluster_idx in range(self.n_clusters):
                members = x[labels == cluster_idx]
                if len(members) == 0:
                    new_centroids[cluster_idx] = x[rng.integers(0, n_samples)]
                else:
                    new_centroids[cluster_idx] = members.mean(axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids = centroids
        return labels

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Call fit before predict")
        distances = ((x[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 Model 1 (from scratch).")
    parser.add_argument("--config", type=str, required=True, help="Path to part1_model1 YAML.")
    return parser.parse_args()


def load_config(path: str) -> Model1Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return Model1Config(
        data_root=raw["data"]["root_dir"],
        classes=list(raw["data"]["classes"]),
        activation=raw["model"]["activation"],
        pooling_type=raw["model"]["pooling_type"],
        feature_dim=int(raw["model"]["feature_dim"]),
        kmeans_max_iter=int(raw["kmeans"]["max_iter"]),
        kmeans_tol=float(raw["kmeans"]["tol"]),
        seed=int(raw["runtime"]["seed"]),
        max_train_per_class=int(raw["runtime"]["max_train_per_class"]),
        max_val_per_class=int(raw["runtime"]["max_val_per_class"]),
        max_test_per_class=int(raw["runtime"]["max_test_per_class"]),
        output_dir=raw["output"]["dir"],
    )


def collect_paths(split_dir: Path, class_name: str) -> list[Path]:
    if not split_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in split_dir.glob("*") if p.suffix.lower() in exts]
    files.sort()
    return files


def extract_split_features(
    split_root: Path,
    classes: list[str],
    max_per_class: int,
    extractor: ScratchConvNet,
    feature_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images for one split and extract 128-D feature vectors."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    file_records: list[str] = []

    for class_idx, class_name in enumerate(classes):
        class_dir = split_root / class_name
        paths = collect_paths(class_dir, class_name)
        if len(paths) == 0:
            continue

        if len(paths) > max_per_class:
            chosen_indices = rng.choice(len(paths), size=max_per_class, replace=False)
            paths = [paths[i] for i in sorted(chosen_indices.tolist())]

        for path in paths:
            image = Image.open(path).convert("RGB")
            arr = np.asarray(image, dtype=np.float32) / 255.0
            feat = extractor.extract_features(arr, output_dim=feature_dim)
            features.append(feat)
            labels.append(class_idx)
            file_records.append(str(path))

    if len(features) == 0:
        raise RuntimeError(f"No images found in split: {split_root}")

    return np.stack(features, axis=0), np.asarray(labels, dtype=np.int64), file_records


def cluster_to_label_map(cluster_ids: np.ndarray, labels: np.ndarray, n_clusters: int) -> dict[int, int]:
    """Map each cluster id to a class id by majority vote on training samples."""
    fallback = int(Counter(labels.tolist()).most_common(1)[0][0])
    mapping: dict[int, int] = {}
    for cluster_idx in range(n_clusters):
        members = labels[cluster_ids == cluster_idx]
        if len(members) == 0:
            mapping[cluster_idx] = fallback
        else:
            mapping[cluster_idx] = int(Counter(members.tolist()).most_common(1)[0][0])
    return mapping


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    extractor = ScratchConvNet(activation=cfg.activation, pooling_type=cfg.pooling_type)

    data_root = Path(cfg.data_root)
    x_train, y_train, train_paths = extract_split_features(
        split_root=data_root / "train",
        classes=cfg.classes,
        max_per_class=cfg.max_train_per_class,
        extractor=extractor,
        feature_dim=cfg.feature_dim,
        rng=rng,
    )
    x_val, y_val, val_paths = extract_split_features(
        split_root=data_root / "val",
        classes=cfg.classes,
        max_per_class=cfg.max_val_per_class,
        extractor=extractor,
        feature_dim=cfg.feature_dim,
        rng=rng,
    )
    x_test, y_test, test_paths = extract_split_features(
        split_root=data_root / "test",
        classes=cfg.classes,
        max_per_class=cfg.max_test_per_class,
        extractor=extractor,
        feature_dim=cfg.feature_dim,
        rng=rng,
    )

    kmeans = KMeansNumpy(
        n_clusters=len(cfg.classes),
        max_iter=cfg.kmeans_max_iter,
        tol=cfg.kmeans_tol,
        seed=cfg.seed,
    )
    train_clusters = kmeans.fit(x_train)
    mapping = cluster_to_label_map(train_clusters, y_train, n_clusters=len(cfg.classes))

    def decode(cluster_ids: np.ndarray) -> np.ndarray:
        return np.array([mapping[int(c)] for c in cluster_ids], dtype=np.int64)

    train_pred = decode(train_clusters)
    val_pred = decode(kmeans.predict(x_val))
    test_pred = decode(kmeans.predict(x_test))

    metrics = {
        "train_accuracy": accuracy(y_train, train_pred),
        "val_accuracy": accuracy(y_val, val_pred),
        "test_accuracy": accuracy(y_test, test_pred),
        "feature_dim": cfg.feature_dim,
        "num_clusters": len(cfg.classes),
        "class_names": cfg.classes,
        "cluster_to_class_map": {
            str(cluster): cfg.classes[class_id] for cluster, class_id in mapping.items()
        },
        "sample_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (output_dir / "predictions_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true_label", "pred_label"])
        for path, true_id, pred_id in zip(test_paths, y_test.tolist(), test_pred.tolist()):
            writer.writerow([path, cfg.classes[true_id], cfg.classes[pred_id]])

    print("Model 1 complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
