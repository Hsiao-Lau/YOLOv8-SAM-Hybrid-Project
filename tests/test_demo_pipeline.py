from pathlib import Path

from yolo_sam_hybrid.config import DatasetConfig, PromptConfig
from yolo_sam_hybrid.demo.synthetic_data import write_demo_dataset
from yolo_sam_hybrid.models.sam_peft import DummySAMSegmenter
from yolo_sam_hybrid.models.yolo_detector import SimpleThresholdDetector
from yolo_sam_hybrid.pipeline.yolo_sam_pipeline import YOLOSAMHybridPromptingFramework


def test_demo_pipeline_runs(tmp_path: Path):
    manifest = write_demo_dataset(tmp_path / "data", n_images=2, size=128)
    cfg = DatasetConfig(name="demo", task="demo", classes=["lesion"], image_size=128, mask_threshold=0.45, min_component_area=16)
    prompt = PromptConfig(t_medium=256, t_large=2048, negative_points=2, safety_distance=6)
    pipe = YOLOSAMHybridPromptingFramework(SimpleThresholdDetector(min_area=8), DummySAMSegmenter(), cfg, prompt)
    results = pipe.run_manifest(manifest, tmp_path / "pred")
    assert len(results) == 2
    assert "dice" in results.columns
    assert (tmp_path / "pred" / "image_level_results.csv").exists()
