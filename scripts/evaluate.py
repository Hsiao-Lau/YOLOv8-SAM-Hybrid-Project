#!/usr/bin/env python
"""Evaluate predicted masks against reference masks in a manifest."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import pandas as pd

from yolo_sam_hybrid.evaluation.metrics import evaluate_mask_pair
from yolo_sam_hybrid.evaluation.statistics import bootstrap_ci, mean_std
from yolo_sam_hybrid.utils.io import load_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument("--manifest", required=True, help="CSV with mask_path and case_id.")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted PNG masks named as case_id.png.")
    parser.add_argument("--output_csv", default="evaluation_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    records = []
    for _, row in df.iterrows():
        case_id = str(row.get("case_id", Path(row["image_path"]).stem))
        pred_path = Path(args.pred_dir) / f"{case_id}.png"
        pred = load_mask(pred_path)
        ref = load_mask(row["mask_path"])
        metrics = evaluate_mask_pair(pred, ref).to_dict()
        records.append({"case_id": case_id, **metrics})
    out = pd.DataFrame(records)
    out.to_csv(args.output_csv, index=False)
    for metric in ["dice", "jaccard", "sensitivity", "specificity"]:
        m, s = mean_std(out[metric])
        lo, hi = bootstrap_ci(out[metric])
        print(f"{metric}: {m:.4f} ± {s:.4f}; 95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"Saved per-case/image metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
