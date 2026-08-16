#!/usr/bin/env python
"""Prepare an ISIC 2018-style manifest and YOLO labels from image/mask folders."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISIC 2018 manifest and YOLO labels.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--mask_suffix", default="_segmentation.png")
    parser.add_argument("--min_area", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in tqdm(sorted(image_dir.glob(f"*{args.image_ext}"))):
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}{args.mask_suffix}"
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        labels = mask_to_yolo_labels(mask, class_id=0, min_area=args.min_area)
        write_yolo_label_file(label_dir / f"{stem}.txt", labels)
        rows.append({"image_path": str(image_path), "mask_path": str(mask_path), "case_id": stem, "class_name": "lesion"})
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)
    print(f"Prepared {len(rows)} ISIC samples. Manifest: {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
