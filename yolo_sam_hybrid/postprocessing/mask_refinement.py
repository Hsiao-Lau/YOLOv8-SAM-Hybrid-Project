"""Mask thresholding and lightweight morphological refinement."""
from __future__ import annotations

import cv2
import numpy as np


def postprocess_probability_mask(
    probability_mask: np.ndarray,
    threshold: float = 0.5,
    min_component_area: int = 64,
    opening_kernel: int = 3,
    closing_kernel: int = 5,
    gaussian_sigma: float = 1.0,
    retain_largest_component: bool = False,
) -> np.ndarray:
    prob = np.asarray(probability_mask, dtype=np.float32)
    binary = (prob >= threshold).astype(np.uint8)

    if opening_kernel and opening_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if closing_kernel and closing_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    binary = filter_connected_components(
        binary,
        min_component_area=min_component_area,
        retain_largest_component=retain_largest_component,
    )

    if gaussian_sigma and gaussian_sigma > 0:
        smoothed = cv2.GaussianBlur(binary.astype(np.float32), ksize=(0, 0), sigmaX=gaussian_sigma)
        binary = (smoothed >= 0.5).astype(np.uint8)
    return binary


def filter_connected_components(
    binary_mask: np.ndarray,
    min_component_area: int = 64,
    retain_largest_component: bool = False,
) -> np.ndarray:
    mask = (binary_mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask

    component_ids = list(range(1, n_labels))
    if retain_largest_component:
        component_ids = [max(component_ids, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])]

    out = np.zeros_like(mask)
    for idx in component_ids:
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            out[labels == idx] = 1
    return out
