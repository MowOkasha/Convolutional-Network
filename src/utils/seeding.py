from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch for more reproducible training runs.

    Exact bit-for-bit determinism can still vary by hardware and CUDA kernels,
    but this function greatly reduces random variation between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode can reduce speed, but it helps in assignment settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
