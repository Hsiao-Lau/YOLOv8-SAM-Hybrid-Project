#!/usr/bin/env python
"""Run a complete smoke test without external YOLO/SAM weights."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from yolo_sam_hybrid.config import DatasetConfig, PromptConfig
from yolo_sam_hybrid.demo.synthetic_data import write_demo_dataset
from yolo_sam_hybrid.models.sam_peft import DummySAMSegmenter
from yolo_sam_hybrid.models.yolo_detector import SimpleThresholdDetector
from yolo_sam_hybrid.pipeline.yolo_sam_pipeline import YOLOSAMHybridPromptingFramework
from yolo_sam_hybrid.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo YOLO-SAM hybrid pipeline.")
    parser.add_argument("--output_dir", default="runs/demo", help="Output directory.")
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17821)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    out = Path(args.output_dir)
    manifest = write_demo_dataset(out / "data", n_images=args.n_images, seed=args.seed)
    dataset_config = DatasetConfig(
        name="demo_isic_like",
        task="demo_skin_lesion_segmentation",
        classes=["lesion"],
        image_size=256,
        detector_input_size=256,
        mask_threshold=0.45,
        min_component_area=32,
        opening_kernel=3,
        closing_kernel=5,
        gaussian_sigma=1.0,
    )
    prompt_config = PromptConfig(t_medium=1024, t_large=9216, negative_points=2, safety_distance=8, deterministic=True, seed=args.seed)
    pipeline = YOLOSAMHybridPromptingFramework(
        detector=SimpleThresholdDetector(class_name="lesion", class_id=0, min_area=64),
        sam_segmenter=DummySAMSegmenter(),
        dataset_config=dataset_config,
        prompt_config=prompt_config,
    )
    results = pipeline.run_manifest(manifest, out / "predictions")
    print(results.describe(include="all"))
    print(f"Demo finished. Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
