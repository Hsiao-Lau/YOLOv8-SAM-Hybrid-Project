"""Lightweight detection metric helpers."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def recall_at_iou(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]], iou_thr: float = 0.5) -> float:
    if len(gt_boxes) == 0:
        return 1.0
    matched = 0
    for gt in gt_boxes:
        if any(box_iou_xyxy(pred, gt) >= iou_thr for pred in pred_boxes):
            matched += 1
    return matched / len(gt_boxes)
