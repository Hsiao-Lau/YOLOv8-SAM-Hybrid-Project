"""Manifest helpers for slice-based and image-based experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def make_manifest(
    image_paths: Iterable[str | Path],
    mask_paths: Iterable[str | Path],
    output_csv: str | Path,
    case_ids: Optional[Iterable[str]] = None,
    class_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    image_paths = list(map(str, image_paths))
    mask_paths = list(map(str, mask_paths))
    if len(image_paths) != len(mask_paths):
        raise ValueError("image_paths and mask_paths must have the same length")
    n = len(image_paths)
    df = pd.DataFrame({"image_path": image_paths, "mask_path": mask_paths})
    df["case_id"] = list(case_ids) if case_ids is not None else [Path(p).stem for p in image_paths]
    df["class_name"] = list(class_names) if class_names is not None else ["target"] * n
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
