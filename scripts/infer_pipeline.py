#!/usr/bin/env python
"""Run full automatic YOLO-SAM hybrid inference on a manifest CSV."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from yolo_sam_hybrid.config import ExperimentConfig
from yolo_sam_hybrid.models.sam_peft import SAMSegmenter
from yolo_sam_hybrid.models.yolo_detector import YOLOv8Detector
from yolo_sam_hybrid.pipeline.yolo_sam_pipeline import YOLOSAMHybridPromptingFramework
from yolo_sam_hybrid.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO-SAM hybrid inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--yolo_weights", required=True)
    parser.add_argument("--sam_checkpoint", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_global_seed(cfg.runtime.seed)
    yolo = YOLOv8Detector(
        weights=args.yolo_weights,
        class_names={idx: name for idx, name in enumerate(cfg.dataset.classes)},
        confidence_threshold=cfg.detector.confidence_threshold,
        nms_iou_threshold=cfg.detector.nms_iou_threshold,
        device=cfg.runtime.device,
    )
    sam = SAMSegmenter(
        checkpoint=args.sam_checkpoint or cfg.sam.checkpoint,
        backbone=cfg.sam.backbone,
        device=cfg.runtime.device,
    )
    pipeline = YOLOSAMHybridPromptingFramework(yolo, sam, cfg.dataset, cfg.prompt)
    output_dir = args.output_dir or cfg.runtime.output_dir
    results = pipeline.run_manifest(args.manifest, output_dir)
    print(results.mean(numeric_only=True))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
