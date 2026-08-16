"""End-to-end YOLO-SAM hybrid prompting pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from yolo_sam_hybrid.config import DatasetConfig, PromptConfig
from yolo_sam_hybrid.evaluation.audit import audit_positive_prompt_validity
from yolo_sam_hybrid.evaluation.metrics import evaluate_mask_pair
from yolo_sam_hybrid.postprocessing.mask_refinement import postprocess_probability_mask
from yolo_sam_hybrid.prompting.hybrid_prompt import Detection, HybridPromptGenerator
from yolo_sam_hybrid.utils.io import ensure_dir, load_image_rgb, load_mask, save_json, save_mask, save_probability
from yolo_sam_hybrid.utils.timing import timer
from yolo_sam_hybrid.utils.visualization import save_visualization


@dataclass
class PipelineOutput:
    image_id: str
    detections: List[Detection]
    prediction_mask: np.ndarray
    probability_mask: np.ndarray
    timings: Dict[str, float]
    metrics: Optional[Dict[str, Any]] = None
    prompt_audit: Optional[Dict[str, Any]] = None


class YOLOSAMHybridPromptingFramework:
    """Full automatic detector-to-SAM segmentation pipeline.

    If no detection is produced, the prediction is an empty mask, preserving the
    detection failure in unconditional end-to-end scores.
    """

    def __init__(
        self,
        detector,
        sam_segmenter,
        dataset_config: DatasetConfig,
        prompt_config: PromptConfig,
    ) -> None:
        self.detector = detector
        self.sam_segmenter = sam_segmenter
        self.dataset_config = dataset_config
        self.prompt_generator = HybridPromptGenerator(
            t_medium=prompt_config.t_medium,
            t_large=prompt_config.t_large,
            negative_points=prompt_config.negative_points,
            safety_distance=prompt_config.safety_distance,
            deterministic=prompt_config.deterministic,
            seed=prompt_config.seed,
        )

    def predict_image(self, image_rgb: np.ndarray, reference_mask: Optional[np.ndarray] = None, image_id: str = "image") -> PipelineOutput:
        times: Dict[str, float] = {}
        with timer(times, "yolo_detection_sec"):
            detections = self.detector.predict(image_rgb)

        h, w = image_rgb.shape[:2]
        if not detections:
            probability = np.zeros((h, w), dtype=np.float32)
            pred = np.zeros((h, w), dtype=np.uint8)
            metrics = evaluate_mask_pair(pred, reference_mask).to_dict() if reference_mask is not None else None
            return PipelineOutput(image_id=image_id, detections=[], prediction_mask=pred, probability_mask=probability, timings=times, metrics=metrics)

        with timer(times, "prompt_generation_sec"):
            prompts = self.prompt_generator.batch_generate(detections, image_rgb.shape)

        probability_union = np.zeros((h, w), dtype=np.float32)
        with timer(times, "sam_and_postprocessing_sec"):
            for prompt in prompts:
                probability = self.sam_segmenter.predict(image_rgb, prompt)
                probability_union = np.maximum(probability_union, probability.astype(np.float32))
            pred = postprocess_probability_mask(
                probability_union,
                threshold=self.dataset_config.mask_threshold,
                min_component_area=self.dataset_config.min_component_area,
                opening_kernel=self.dataset_config.opening_kernel,
                closing_kernel=self.dataset_config.closing_kernel,
                gaussian_sigma=self.dataset_config.gaussian_sigma,
            )

        metrics = None
        audit = None
        if reference_mask is not None:
            metrics = evaluate_mask_pair(pred, reference_mask).to_dict()
            audit = audit_positive_prompt_validity(prompts, reference_mask)
        times["total_sec"] = sum(times.values())
        return PipelineOutput(
            image_id=image_id,
            detections=detections,
            prediction_mask=pred,
            probability_mask=probability_union,
            timings=times,
            metrics=metrics,
            prompt_audit=audit,
        )

    def run_manifest(self, manifest_csv: str | Path, output_dir: str | Path) -> pd.DataFrame:
        output_dir = ensure_dir(output_dir)
        pred_dir = ensure_dir(output_dir / "pred_masks")
        prob_dir = ensure_dir(output_dir / "probability_masks")
        vis_dir = ensure_dir(output_dir / "visualizations")
        meta_dir = ensure_dir(output_dir / "metadata")
        df = pd.read_csv(manifest_csv)
        records: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"]) if "mask_path" in row and pd.notna(row["mask_path"]) else None
            image_id = str(row.get("case_id", image_path.stem))
            image = load_image_rgb(image_path)
            ref = load_mask(mask_path) if mask_path is not None and mask_path.exists() else None
            out = self.predict_image(image, reference_mask=ref, image_id=image_id)

            save_mask(pred_dir / f"{image_id}.png", out.prediction_mask)
            save_probability(prob_dir / f"{image_id}.png", out.probability_mask)
            save_visualization(vis_dir / f"{image_id}.png", image, out.prediction_mask)
            save_json(
                meta_dir / f"{image_id}.json",
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "detections": [asdict(d) for d in out.detections],
                    "timings": out.timings,
                    "metrics": out.metrics,
                    "prompt_audit": out.prompt_audit,
                },
            )
            record: Dict[str, Any] = {"image_id": image_id, "n_detections": len(out.detections), **out.timings}
            if out.metrics:
                record.update(out.metrics)
            if out.prompt_audit:
                record.update(out.prompt_audit)
            records.append(record)

        results = pd.DataFrame(records)
        results.to_csv(output_dir / "image_level_results.csv", index=False)
        return results
