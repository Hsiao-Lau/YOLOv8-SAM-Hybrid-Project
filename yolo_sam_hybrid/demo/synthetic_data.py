"""Synthetic data generator for a runnable smoke test."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from yolo_sam_hybrid.utils.io import ensure_dir


def create_synthetic_lesion_image(size: int = 256, seed: int = 17821) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = rng.normal(35, 5, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (int(rng.integers(size // 3, size * 2 // 3)), int(rng.integers(size // 3, size * 2 // 3)))
    axes = (int(rng.integers(size // 8, size // 5)), int(rng.integers(size // 10, size // 4)))
    angle = float(rng.integers(0, 180))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    image[mask > 0] = np.asarray([190, 190, 190], dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image, mask


def write_demo_dataset(output_dir: str | Path, n_images: int = 8, size: int = 256, seed: int = 17821) -> Path:
    output_dir = ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    rows = []
    for i in range(n_images):
        image, mask = create_synthetic_lesion_image(size=size, seed=seed + i)
        image_path = image_dir / f"demo_{i:03d}.png"
        mask_path = mask_dir / f"demo_{i:03d}.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask * 255).save(mask_path)
        rows.append({"image_path": str(image_path), "mask_path": str(mask_path), "case_id": f"demo_{i:03d}", "class_name": "lesion"})
    manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
