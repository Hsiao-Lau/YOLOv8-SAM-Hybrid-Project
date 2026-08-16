"""File-system and image I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_path", "mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def load_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def load_mask(path: str | Path) -> np.ndarray:
    mask = Image.open(path)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    out = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(out).save(path)


def save_probability(path: str | Path, prob: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    arr = np.clip(prob, 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
