from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Configure deterministic random seeds for reproducible experiments.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False