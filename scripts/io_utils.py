"""io_utils.py — JSON serialization helpers."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def json_serializable(obj):
    """Recursively convert objects to JSON-safe types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
        return str(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='list')
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: json_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_serializable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def save_json(obj, path: str | Path) -> None:
    """Save an object to JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(json_serializable(obj), f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)
