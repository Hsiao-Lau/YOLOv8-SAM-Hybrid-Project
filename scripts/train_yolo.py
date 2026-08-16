#!/usr/bin/env python
"""Train YOLOv8 detector using an Ultralytics data.yaml file."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from yolo_sam_hybrid.config import ExperimentConfig
from yolo_sam_hybrid.models.yolo_detector import YOLOv8Detector
from yolo_sam_hybrid.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    parser.add_argument("--data_yaml", required=True, help="Ultralytics YOLO data.yaml.")
    parser.add_argument("--name", default="yolov8_detector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_global_seed(cfg.runtime.seed)
    YOLOv8Detector.train(
        data_yaml=args.data_yaml,
        model_initialization=cfg.detector.initialization,
        image_size=cfg.detector.input_size,
        batch_size=cfg.detector.batch_size,
        epochs=cfg.detector.epochs,
        patience=cfg.detector.patience,
        learning_rate=cfg.detector.learning_rate,
        project=cfg.detector.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
