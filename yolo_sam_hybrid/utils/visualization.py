"""Visualization helpers for qualitative examples."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .io import ensure_dir


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.uint8).copy()
    color = np.zeros_like(image)
    color[..., 1] = 255
    m = mask.astype(bool)
    image[m] = (image[m] * (1 - alpha) + color[m] * alpha).astype(np.uint8)
    return image


def draw_boxes_and_points(image_rgb: np.ndarray, detections, prompts) -> np.ndarray:
    canvas = image_rgb.astype(np.uint8).copy()
    for det, prompt in zip(detections, prompts):
        x1, y1, x2, y2 = map(int, det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for x, y in prompt.positive_points:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
        for x, y in prompt.negative_points:
            cv2.circle(canvas, (int(x), int(y)), 4, (255, 0, 255), -1)
    return canvas


def save_visualization(path: str | Path, image_rgb: np.ndarray, mask: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    Image.fromarray(overlay_mask(image_rgb, mask)).save(path)
