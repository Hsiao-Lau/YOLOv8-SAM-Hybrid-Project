"""Segmentation metrics with explicit empty-mask handling."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


@dataclass
class SegmentationMetrics:
    dice: float
    jaccard: float
    sensitivity: float
    specificity: float
    hd: float
    hd95: float
    asd: float
    reference_empty: bool
    prediction_empty: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


def dice_score(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if not ref.any() and not pred.any():
        return 1.0
    if not ref.any() or not pred.any():
        return 0.0
    return float(2.0 * np.logical_and(pred, ref).sum() / (pred.sum() + ref.sum()))


def jaccard_index(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if not ref.any() and not pred.any():
        return 1.0
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def sensitivity_specificity(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    tp = np.logical_and(pred, ref).sum()
    fn = np.logical_and(~pred, ref).sum()
    tn = np.logical_and(~pred, ~ref).sum()
    fp = np.logical_and(pred, ~ref).sum()
    sens = 0.0 if tp + fn == 0 else float(tp / (tp + fn))
    spec = 0.0 if tn + fp == 0 else float(tn / (tn + fp))
    return sens, spec


def surface_distances(pred: np.ndarray, ref: np.ndarray, spacing: Optional[Tuple[float, float]] = None) -> np.ndarray:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if not pred.any() or not ref.any():
        return np.asarray([np.inf], dtype=np.float64)
    spacing = spacing or (1.0, 1.0)
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    ref_surface = np.logical_xor(ref, binary_erosion(ref))
    dt_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    d_pred_to_ref = dt_ref[pred_surface]
    d_ref_to_pred = dt_pred[ref_surface]
    return np.concatenate([d_pred_to_ref, d_ref_to_pred]).astype(np.float64)


def hausdorff_distance(pred: np.ndarray, ref: np.ndarray, spacing: Optional[Tuple[float, float]] = None) -> float:
    distances = surface_distances(pred, ref, spacing=spacing)
    return float(np.max(distances))


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: Optional[Tuple[float, float]] = None) -> float:
    distances = surface_distances(pred, ref, spacing=spacing)
    return float(np.percentile(distances, 95))


def average_surface_distance(pred: np.ndarray, ref: np.ndarray, spacing: Optional[Tuple[float, float]] = None) -> float:
    distances = surface_distances(pred, ref, spacing=spacing)
    return float(np.mean(distances))


def evaluate_mask_pair(
    pred: np.ndarray,
    ref: np.ndarray,
    spacing: Optional[Tuple[float, float]] = None,
) -> SegmentationMetrics:
    pred = (pred > 0).astype(np.uint8)
    ref = (ref > 0).astype(np.uint8)
    ref_empty = not bool(ref.any())
    pred_empty = not bool(pred.any())
    sens, spec = sensitivity_specificity(pred, ref)
    if ref_empty and pred_empty:
        hd = hd_95 = asd = 0.0
    elif ref_empty or pred_empty:
        hd = hd_95 = asd = float("inf")
    else:
        hd = hausdorff_distance(pred, ref, spacing=spacing)
        hd_95 = hd95(pred, ref, spacing=spacing)
        asd = average_surface_distance(pred, ref, spacing=spacing)
    return SegmentationMetrics(
        dice=dice_score(pred, ref),
        jaccard=jaccard_index(pred, ref),
        sensitivity=sens,
        specificity=spec,
        hd=hd,
        hd95=hd_95,
        asd=asd,
        reference_empty=ref_empty,
        prediction_empty=pred_empty,
    )
