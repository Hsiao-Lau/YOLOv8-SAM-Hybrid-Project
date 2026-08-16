"""Construction of YOLO labels from segmentation masks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

from yolo_sam_hybrid.utils.io import ensure_dir


@dataclass
class YoloBoxLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    xyxy_pixels: tuple[int, int, int, int]


def connected_component_boxes(mask: np.ndarray, min_area: int = 1) -> List[tuple[int, int, int, int]]:
    mask = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[tuple[int, int, int, int]] = []
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return boxes


def mask_to_yolo_labels(mask: np.ndarray, class_id: int, min_area: int = 1) -> List[YoloBoxLabel]:
    h, w = mask.shape[:2]
    labels: List[YoloBoxLabel] = []
    for x1, y1, x2, y2 in connected_component_boxes(mask, min_area=min_area):
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        xc = x1 + bw / 2
        yc = y1 + bh / 2
        labels.append(
            YoloBoxLabel(
                class_id=class_id,
                x_center=xc / w,
                y_center=yc / h,
                width=bw / w,
                height=bh / h,
                xyxy_pixels=(x1, y1, x2, y2),
            )
        )
    return labels


def write_yolo_label_file(path: str | Path, labels: Sequence[YoloBoxLabel]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for item in labels:
            f.write(
                f"{item.class_id} {item.x_center:.8f} {item.y_center:.8f} "
                f"{item.width:.8f} {item.height:.8f}\n"
            )


def convert_mask_folder_to_yolo_labels(
    mask_paths: Iterable[str | Path],
    output_label_dir: str | Path,
    class_id: int = 0,
    min_area: int = 1,
) -> None:
    from PIL import Image

    output_label_dir = ensure_dir(output_label_dir)
    for mask_path in mask_paths:
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        labels = mask_to_yolo_labels(mask, class_id=class_id, min_area=min_area)
        write_yolo_label_file(output_label_dir / (Path(mask_path).stem + ".txt"), labels)
