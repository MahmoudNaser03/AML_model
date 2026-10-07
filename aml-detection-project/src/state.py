"""
Checkpoint helpers — save/load intermediate pipeline state via pickle.
Keeps notebooks idempotent: re-run any notebook without rebuilding everything.
"""

import pickle
from pathlib import Path

from src.config import CHECKPOINTS_DIR


def save(name: str, obj, directory: Path = CHECKPOINTS_DIR) -> Path:
    path = directory / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"Saved checkpoint: {path}")
    return path


def load(name: str, directory: Path = CHECKPOINTS_DIR):
    path = directory / f"{name}.pkl"
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    print(f"Loaded checkpoint: {path}")
    return obj


def exists(name: str, directory: Path = CHECKPOINTS_DIR) -> bool:
    return (directory / f"{name}.pkl").exists()
