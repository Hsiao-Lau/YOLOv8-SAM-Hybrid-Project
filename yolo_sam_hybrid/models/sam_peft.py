"""SAM inference and parameter-efficient fine-tuning utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from yolo_sam_hybrid.prompting.hybrid_prompt import HybridPrompt


class SAMSegmenter:
    """SAM predictor wrapper using box, positive-point, and negative-point prompts."""

    def __init__(self, checkpoint: str, backbone: str = "vit_b", device: str = "auto") -> None:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise ImportError(
                "Install PyTorch and segment-anything to use SAMSegmenter. "
                "Example: pip install torch torchvision && pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.sam = sam_model_registry[backbone](checkpoint=checkpoint).to(device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    def predict(self, image_rgb: np.ndarray, prompt: HybridPrompt) -> np.ndarray:
        self.predictor.set_image(image_rgb)
        point_coords, point_labels = prompt.to_sam_points()
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords if len(point_coords) else None,
            point_labels=point_labels if len(point_labels) else None,
            box=np.asarray(prompt.box, dtype=np.float32),
            multimask_output=False,
        )
        return masks[0].astype(np.float32)


def configure_sam_peft_trainable_modules(
    sam_model,
    freeze_image_encoder: bool = True,
    train_prompt_encoder: bool = True,
    train_mask_decoder: bool = True,
) -> dict:
    """Freeze SAM image encoder and enable prompt encoder/mask decoder training."""
    for p in sam_model.parameters():
        p.requires_grad = False
    if not freeze_image_encoder:
        for p in sam_model.image_encoder.parameters():
            p.requires_grad = True
    if train_prompt_encoder:
        for p in sam_model.prompt_encoder.parameters():
            p.requires_grad = True
    if train_mask_decoder:
        for p in sam_model.mask_decoder.parameters():
            p.requires_grad = True
    total = sum(p.numel() for p in sam_model.parameters())
    trainable = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


class DummySAMSegmenter:
    """Demo SAM-like segmenter that converts prompts to an ellipse probability mask."""

    def predict(self, image_rgb: np.ndarray, prompt: HybridPrompt) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = map(int, prompt.box)
        prob = np.zeros((h, w), dtype=np.float32)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        ax = max(2, int((x2 - x1) / 2.1))
        ay = max(2, int((y2 - y1) / 2.1))
        cv2.ellipse(prob, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        for x, y in prompt.positive_points:
            cv2.circle(prob, (int(x), int(y)), max(2, min(ax, ay) // 4), 1.0, -1)
        for x, y in prompt.negative_points:
            cv2.circle(prob, (int(x), int(y)), max(2, min(ax, ay) // 5), 0.0, -1)
        prob = cv2.GaussianBlur(prob, (0, 0), sigmaX=2.0)
        return np.clip(prob, 0, 1)


def _mask_to_xyxy(mask_np: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.asarray([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)


def _train_one_sam_prompted_sample(sam, image_chw, mask_1hw, transform, loss_fn, device, freeze_image_encoder: bool):
    """Train SAM on one image using an oracle box derived from the reference mask.

    This is used for SAM PEFT training. Automatic YOLO-predicted prompts are used in
    end-to-end inference, while oracle boxes here provide stable supervised adaptation
    of the prompt encoder and mask decoder.
    """
    import torch

    image_np = (image_chw.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    mask_np = (mask_1hw.detach().cpu().squeeze(0).numpy() > 0).astype(np.uint8)
    box_np = _mask_to_xyxy(mask_np)
    if box_np is None:
        return None

    original_size = image_np.shape[:2]
    resized_image = transform.apply_image(image_np)
    input_size = resized_image.shape[:2]
    image_t = torch.as_tensor(resized_image, device=device).permute(2, 0, 1).contiguous().float()
    image_t = sam.preprocess(image_t[None, :, :, :])

    box_resized = transform.apply_boxes(box_np, original_size)
    box_t = torch.as_tensor(box_resized, dtype=torch.float32, device=device)
    target = mask_1hw[None, ...].to(device).float()

    if freeze_image_encoder:
        with torch.no_grad():
            image_embeddings = sam.image_encoder(image_t)
    else:
        image_embeddings = sam.image_encoder(image_t)

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=box_t, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    logits = sam.postprocess_masks(low_res_masks, input_size=input_size, original_size=original_size)
    return loss_fn(logits, target)


def fine_tune_sam_peft(
    train_loader,
    val_loader,
    checkpoint: str,
    backbone: str,
    output_dir: str,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    lambda_dice: float = 0.5,
    lambda_focal: float = 0.5,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    freeze_image_encoder: bool = True,
):
    """Fine-tune SAM prompt encoder and mask decoder with Dice + focal loss.

    The image encoder is frozen by default. Training prompts are oracle boxes derived
    from the reference masks; full automatic inference still uses YOLO-predicted boxes
    and detector-derived hybrid point prompts.
    """
    import torch
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    from yolo_sam_hybrid.models.loss import DiceFocalLoss
    from yolo_sam_hybrid.utils.io import ensure_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = ensure_dir(output_dir)
    sam = sam_model_registry[backbone](checkpoint=checkpoint).to(device)
    param_info = configure_sam_peft_trainable_modules(
        sam,
        freeze_image_encoder=freeze_image_encoder,
        train_prompt_encoder=True,
        train_mask_decoder=True,
    )
    optimizer = torch.optim.AdamW((p for p in sam.parameters() if p.requires_grad), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = DiceFocalLoss(lambda_dice=lambda_dice, lambda_focal=lambda_focal, alpha=focal_alpha, gamma=focal_gamma)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        sam.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for image_chw, mask_1hw in zip(batch["image"], batch["mask"]):
                loss = _train_one_sam_prompted_sample(
                    sam=sam,
                    image_chw=image_chw,
                    mask_1hw=mask_1hw,
                    transform=transform,
                    loss_fn=loss_fn,
                    device=device,
                    freeze_image_encoder=freeze_image_encoder,
                )
                if loss is not None:
                    batch_losses.append(loss)
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        sam.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                for image_chw, mask_1hw in zip(batch["image"], batch["mask"]):
                    loss = _train_one_sam_prompted_sample(
                        sam=sam,
                        image_chw=image_chw,
                        mask_1hw=mask_1hw,
                        transform=transform,
                        loss_fn=loss_fn,
                        device=device,
                        freeze_image_encoder=True,
                    )
                    if loss is not None:
                        val_losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses)) if train_losses else float("inf")
        print(f"Epoch {epoch:03d}: train_loss={np.mean(train_losses) if train_losses else float('nan'):.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model": sam.state_dict(), "param_info": param_info, "epoch": epoch}, output_dir / "sam_peft_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return {"best_val_loss": best_val, **param_info}
