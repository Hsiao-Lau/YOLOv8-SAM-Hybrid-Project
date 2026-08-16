"""Hybrid box/positive-point/negative-point prompt generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "target"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class HybridPrompt:
    box: Tuple[float, float, float, float]
    positive_points: List[Tuple[float, float]]
    negative_points: List[Tuple[float, float]]
    point_labels: List[int]

    def to_sam_points(self) -> tuple[np.ndarray, np.ndarray]:
        points = self.positive_points + self.negative_points
        labels = [1] * len(self.positive_points) + [0] * len(self.negative_points)
        return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class HybridPromptGenerator:
    """Generate detector-derived heuristic positive and negative SAM prompts.

    Positive points are sampled in the central region of each predicted box. They are
    not checked against ground truth at inference. Negative points are generated outside
    the box with a safety distance.
    """

    def __init__(
        self,
        t_medium: int,
        t_large: int,
        negative_points: int = 2,
        safety_distance: int = 8,
        deterministic: bool = True,
        seed: int = 17821,
    ) -> None:
        self.t_medium = int(t_medium)
        self.t_large = int(t_large)
        self.negative_points = int(negative_points)
        self.safety_distance = int(safety_distance)
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def positive_point_count(self, area: float) -> int:
        if area > self.t_large:
            return 1
        if area > self.t_medium:
            return 2
        return 3

    def generate(self, detection: Detection, image_shape: tuple[int, int]) -> HybridPrompt:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = self._clip_box(detection.xyxy, w, h)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        n_pos = self.positive_point_count(bw * bh)
        positive = self._positive_points(xc, yc, bw, bh, n_pos)
        negative = self._negative_points(x1, y1, x2, y2, w, h, self.negative_points)
        return HybridPrompt(
            box=(x1, y1, x2, y2),
            positive_points=positive,
            negative_points=negative,
            point_labels=[1] * len(positive) + [0] * len(negative),
        )

    def batch_generate(self, detections: Sequence[Detection], image_shape: tuple[int, int]) -> List[HybridPrompt]:
        return [self.generate(det, image_shape) for det in detections]

    @staticmethod
    def _clip_box(xyxy: Sequence[float], w: int, h: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = map(float, xyxy)
        x1 = float(np.clip(x1, 0, max(0, w - 1)))
        x2 = float(np.clip(x2, 0, max(0, w - 1)))
        y1 = float(np.clip(y1, 0, max(0, h - 1)))
        y2 = float(np.clip(y2, 0, max(0, h - 1)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _positive_points(self, xc: float, yc: float, bw: float, bh: float, n_pos: int) -> List[Tuple[float, float]]:
        if self.deterministic:
            if n_pos == 1:
                offsets = [(0.0, 0.0)]
            elif n_pos == 2:
                offsets = [(-0.12, -0.12), (0.12, 0.12)]
            else:
                offsets = [(0.0, 0.0), (-0.15, -0.15), (0.15, 0.15)]
            return [(xc + ox * bw, yc + oy * bh) for ox, oy in offsets[:n_pos]]
        offsets_x = self.rng.uniform(-bw / 4.0, bw / 4.0, size=n_pos)
        offsets_y = self.rng.uniform(-bh / 4.0, bh / 4.0, size=n_pos)
        return [(xc + float(dx), yc + float(dy)) for dx, dy in zip(offsets_x, offsets_y)]

    def _negative_points(self, x1: float, y1: float, x2: float, y2: float, w: int, h: int, n_neg: int) -> List[Tuple[float, float]]:
        if n_neg <= 0:
            return []
        delta = self.safety_distance
        candidates = [
            (x1 - delta, y1 - delta),
            (x2 + delta, y1 - delta),
            (x1 - delta, y2 + delta),
            (x2 + delta, y2 + delta),
            ((x1 + x2) / 2.0, y1 - delta),
            ((x1 + x2) / 2.0, y2 + delta),
            (x1 - delta, (y1 + y2) / 2.0),
            (x2 + delta, (y1 + y2) / 2.0),
        ]
        valid: List[Tuple[float, float]] = []
        for x, y in candidates:
            x = float(np.clip(x, 0, max(0, w - 1)))
            y = float(np.clip(y, 0, max(0, h - 1)))
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                valid.append((x, y))
            if len(valid) >= n_neg:
                break
        while len(valid) < n_neg:
            # Deterministic fallback on image corners.
            corner = [(0.0, 0.0), (w - 1.0, 0.0), (0.0, h - 1.0), (w - 1.0, h - 1.0)][len(valid) % 4]
            valid.append(corner)
        return valid[:n_neg]
