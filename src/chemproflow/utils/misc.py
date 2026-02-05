import json
import pickle
import random
from typing import Any, Dict

from lightning import pytorch as pl
import numpy as np
import torch


def set_seed(seed: int = 42, workers: bool = False):
    """Set random seeds and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=workers)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Use deterministic algorithms when available; may raise if unsupported ops are used.
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str) -> Dict:
    with open(path, "r") as fd:
        return json.load(fd)


def write_json(data: Dict, path: str):
    with open(path, "w") as fod:
        json.dump(data, fod)


def write_pickle(data: Any, path: str):
    with open(path, "wb") as fd:
        pickle.dump(data, fd)


def flatten_recursive(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_recursive(item))
        else:
            flattened.append(item)
    flattened = list(set(flattened))
    return flattened
