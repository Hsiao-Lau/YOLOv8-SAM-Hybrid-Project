"""PyTorch datasets for SAM fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

from yolo_sam_hybrid.utils.io import load_image_rgb, load_mask


class SliceSegmentationDataset:
    """Minimal dataset returning image/mask pairs from a manifest CSV.

    The class avoids importing torch at module import time so that preprocessing and
    demo utilities run on systems without PyTorch.
    """

    def __init__(self, manifest_csv: str | Path, image_size: int = 1024, transform: Optional[Callable] = None):
        self.df = pd.read_csv(manifest_csv)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        import torch

        row = self.df.iloc[index]
        image = load_image_rgb(row["image_path"])
        mask = load_mask(row["mask_path"])
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy((mask > 0).astype(np.float32))[None, ...]
        return {
            "image": image_t,
            "mask": mask_t,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "case_id": row.get("case_id", Path(row["image_path"]).stem),
        }
