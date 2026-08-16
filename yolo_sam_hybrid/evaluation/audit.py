"""Audit routines for detector failure and prompt validity."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from yolo_sam_hybrid.prompting.hybrid_prompt import HybridPrompt


def audit_positive_prompt_validity(prompts: Iterable[HybridPrompt], reference_mask: np.ndarray) -> dict:
    mask = (reference_mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    total = 0
    outside = 0
    for prompt in prompts:
        for x, y in prompt.positive_points:
            total += 1
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            if mask[yi, xi] == 0:
                outside += 1
    return {
        "positive_points_total": total,
        "positive_points_outside_reference": outside,
        "outside_reference_rate": 0.0 if total == 0 else outside / total,
    }


def detection_failure_summary(detections_per_image: List[int], references_non_empty: List[bool]) -> dict:
    total_positive_refs = sum(bool(x) for x in references_non_empty)
    missed = sum(bool(ref) and n_det == 0 for n_det, ref in zip(detections_per_image, references_non_empty))
    return {
        "positive_reference_images": total_positive_refs,
        "missed_detection_images": missed,
        "missed_detection_rate": 0.0 if total_positive_refs == 0 else missed / total_positive_refs,
    }
