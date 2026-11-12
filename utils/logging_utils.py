"""Logging utilities for ATLAS-MERGE."""
import os
import json
import logging
from typing import Any, Dict
import numpy as np
import torch

def _to_serializable(obj: Any) -> Any:
    """Convert PyTorch tensors, NumPy arrays/scalars to JSON-serializable types."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def get_logger(name: str = "atlas_merge", level: int = logging.INFO) -> logging.Logger:
    """Create and configure a standardized application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_jsonl(record: Dict[str, Any], filepath: str) -> None:
    """Append a structured dictionary record to a JSONL file.
    
    Creates parent directories automatically if they do not exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable_record = _to_serializable(record)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable_record) + "\n")

