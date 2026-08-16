"""Preprocessing for Prostate158 slices and ISIC 2018 dermoscopy images."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = float(np.min(image)), float(np.max(image))
    if hi - lo < eps:
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo + eps)


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    width, height = size
    interp = cv2.INTER_LINEAR if image.ndim == 3 else cv2.INTER_LINEAR
    out_img = cv2.resize(image, (width, height), interpolation=interp)
    out_mask = None
    if mask is not None:
        out_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        out_mask = (out_mask > 0).astype(np.uint8)
    return out_img, out_mask


def build_prostate_three_channel_input(
    t2w_slice: np.ndarray,
    dwi_slice: np.ndarray | None = None,
    adc_slice: np.ndarray | None = None,
    anatomical_zone_task: bool = False,
) -> np.ndarray:
    """Map prostate MRI slices to a 3-channel tensor expected by YOLOv8 and SAM.

    For CG/PZ zone segmentation, T2W is repeated into three channels. For PCa lesion
    segmentation, registered T2W/DWI/ADC are used when available; missing channels are
    filled by T2W to keep inference deterministic.
    """
    t2 = normalize_minmax(t2w_slice)
    if anatomical_zone_task:
        stacked = np.stack([t2, t2, t2], axis=-1)
    else:
        dwi = normalize_minmax(dwi_slice) if dwi_slice is not None else t2
        adc = normalize_minmax(adc_slice) if adc_slice is not None else t2
        stacked = np.stack([t2, dwi, adc], axis=-1)
    return (stacked * 255).clip(0, 255).astype(np.uint8)


def remove_dermoscopy_black_border(image_rgb: np.ndarray) -> np.ndarray:
    """Remove simple dark borders introduced by dermoscopy vignetting."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    non_dark = gray > 8
    ys, xs = np.where(non_dark)
    if len(xs) == 0 or len(ys) == 0:
        return image_rgb
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return image_rgb[y1 : y2 + 1, x1 : x2 + 1]


def preprocess_isic_rgb(image_rgb: np.ndarray, output_size: int = 1024) -> np.ndarray:
    image = remove_dermoscopy_black_border(image_rgb)
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8)
