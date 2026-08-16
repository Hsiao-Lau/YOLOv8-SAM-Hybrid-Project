#!/usr/bin/env python
"""Fine-tune SAM prompt encoder and mask decoder with the manuscript loss."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from yolo_sam_hybrid.config import ExperimentConfig
from yolo_sam_hybrid.data.dataset import SliceSegmentationDataset
from yolo_sam_hybrid.models.sam_peft import fine_tune_sam_peft
from yolo_sam_hybrid.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM with PEFT strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    set_global_seed(cfg.runtime.seed)
    train_ds = SliceSegmentationDataset(args.train_manifest, image_size=cfg.dataset.image_size)
    val_ds = SliceSegmentationDataset(args.val_manifest, image_size=cfg.dataset.image_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.sam.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.sam.batch_size, shuffle=False, num_workers=2)
    info = fine_tune_sam_peft(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=args.sam_checkpoint or cfg.sam.checkpoint,
        backbone=cfg.sam.backbone,
        output_dir=args.output_dir,
        epochs=cfg.sam.epochs,
        learning_rate=cfg.sam.learning_rate,
        weight_decay=cfg.sam.weight_decay,
        patience=cfg.sam.patience,
        lambda_dice=cfg.sam.lambda_dice,
        lambda_focal=cfg.sam.lambda_focal,
        focal_alpha=cfg.sam.focal_alpha,
        focal_gamma=cfg.sam.focal_gamma,
    )
    print(info)


if __name__ == "__main__":
    main()
