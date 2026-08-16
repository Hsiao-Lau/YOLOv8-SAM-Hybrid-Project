"""Configuration objects for the YOLO-SAM hybrid prompting framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str
    task: str
    classes: List[str]
    image_size: int = 1024
    detector_input_size: int = 640
    mask_threshold: float = 0.5
    min_component_area: int = 64
    opening_kernel: int = 3
    closing_kernel: int = 5
    gaussian_sigma: float = 1.0
    pixel_spacing_mm: Optional[List[float]] = None


@dataclass
class DetectorConfig:
    model_size: str = "yolov8s"
    initialization: str = "yolov8s.pt"
    input_size: int = 640
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 20
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.5
    project: str = "runs/yolo"


@dataclass
class PromptConfig:
    t_medium: int = 1024
    t_large: int = 9216
    negative_points: int = 2
    safety_distance: int = 8
    deterministic: bool = True
    seed: int = 17821


@dataclass
class SAMConfig:
    backbone: str = "vit_b"
    checkpoint: str = "weights/sam_vit_b_01ec64.pth"
    freeze_image_encoder: bool = True
    train_prompt_encoder: bool = True
    train_mask_decoder: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    patience: int = 10
    lambda_dice: float = 0.5
    lambda_focal: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 17821
    output_dir: str = "runs/yolo_sam_hybrid"
    save_probability: bool = True
    save_visualization: bool = True


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            dataset=DatasetConfig(**raw.get("dataset", {})),
            detector=DetectorConfig(**raw.get("detector", {})),
            prompt=PromptConfig(**raw.get("prompt", {})),
            sam=SAMConfig(**raw.get("sam", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset.__dict__,
            "detector": self.detector.__dict__,
            "prompt": self.prompt.__dict__,
            "sam": self.sam.__dict__,
            "runtime": self.runtime.__dict__,
        }
