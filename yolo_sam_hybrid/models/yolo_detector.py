"""YOLOv8 detector wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from yolo_sam_hybrid.prompting.hybrid_prompt import Detection


class YOLOv8Detector:
    """Thin wrapper around Ultralytics YOLOv8 for detection and training."""

    def __init__(
        self,
        weights: str,
        class_names: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError("Install ultralytics to use YOLOv8Detector: pip install ultralytics==8.0.196") from exc
        self.model = YOLO(weights)
        self.class_names = class_names or getattr(self.model, "names", {}) or {}
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = None if device == "auto" else device

    def predict(self, image_rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image_rgb,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
            device=self.device,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            conf = result.boxes.conf.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, conf, cls):
                detections.append(
                    Detection(
                        xyxy=tuple(map(float, box.tolist())),
                        confidence=float(score),
                        class_id=int(class_id),
                        class_name=str(self.class_names.get(int(class_id), int(class_id))),
                    )
                )
        return detections

    @staticmethod
    def train(
        data_yaml: str,
        model_initialization: str = "yolov8s.pt",
        image_size: int = 640,
        batch_size: int = 16,
        epochs: int = 100,
        patience: int = 20,
        learning_rate: float = 1e-3,
        project: str = "runs/yolo",
        name: str = "yolov8_detector",
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError("Install ultralytics to train YOLOv8: pip install ultralytics==8.0.196") from exc
        model = YOLO(model_initialization)
        return model.train(
            data=data_yaml,
            imgsz=image_size,
            batch=batch_size,
            epochs=epochs,
            patience=patience,
            lr0=learning_rate,
            optimizer="AdamW",
            project=project,
            name=name,
        )


class SimpleThresholdDetector:
    """Small deterministic detector used only for demo/testing without YOLO weights."""

    def __init__(self, class_name: str = "lesion", class_id: int = 0, confidence: float = 0.99, min_area: int = 16):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.min_area = min_area

    def predict(self, image_rgb: np.ndarray) -> List[Detection]:
        import cv2

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
        detections = []
        for idx in range(1, n_labels):
            x, y, w, h, area = stats[idx]
            if area < self.min_area:
                continue
            detections.append(
                Detection(
                    xyxy=(float(x), float(y), float(x + w - 1), float(y + h - 1)),
                    confidence=self.confidence,
                    class_id=self.class_id,
                    class_name=self.class_name,
                )
            )
        return detections
