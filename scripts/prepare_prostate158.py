#!/usr/bin/env python
"""Prepare slice-based Prostate158 manifests and YOLO labels.

This script assumes pre-extracted 2D RGB/PNG slices and binary masks. If your raw
Prostate158 data are NIfTI volumes, first export registered T2W/DWI/ADC slices using
your institutional preprocessing script, then run this manifest builder.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from yolo_sam_hybrid.data.labels import mask_to_yolo_labels, write_yolo_label_file
from yolo_sam_hybrid.utils.io import load_mask

CLASS_TO_ID = {"CG": 0, "PZ": 1, "PCa": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare slice-based Prostate158 manifest and YOLO labels.")
    parser.add_argument("--slice_dir", required=True, help="Directory of preprocessed 2D slice images.")
    parser.add_argument("--mask_dir", required=True, help="Directory of binary target masks.")
    parser.add_argument("--class_name", required=True, choices=list(CLASS_TO_ID))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_ext", default=".png")
    parser.add_argument("--mask_ext", default=".png")
    parser.add_argument("--min_area", type=int, default=1)
    return parser.parse_args()


def infer_case_id(slice_stem: str) -> str:
    # Expected style: patient001_slice034 or patient001_CG_slice034.
    parts = slice_stem.split("_slice")
    return parts[0] if parts else slice_stem


def main() -> None:
    args = parse_args()
    slice_dir = Path(args.slice_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    class_id = CLASS_TO_ID[args.class_name]
    for image_path in tqdm(sorted(slice_dir.glob(f"*{args.image_ext}"))):
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}{args.mask_ext}"
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        labels = mask_to_yolo_labels(mask, class_id=class_id, min_area=args.min_area)
        write_yolo_label_file(label_dir / f"{stem}.txt", labels)
        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "case_id": infer_case_id(stem),
                "slice_id": stem,
                "class_name": args.class_name,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"manifest_{args.class_name}.csv", index=False)
    print(f"Prepared {len(rows)} prostate slices. Manifest: {output_dir / f'manifest_{args.class_name}.csv'}")


if __name__ == "__main__":
    main()
