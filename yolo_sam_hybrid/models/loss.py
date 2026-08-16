"""Dice and binary focal losses used for SAM PEFT fine-tuning."""
from __future__ import annotations


def dice_loss(prob, target, eps: float = 1.0):
    import torch

    prob = prob.float()
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    numerator = 2.0 * torch.sum(prob * target, dim=dims) + eps
    denominator = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims) + eps
    return 1.0 - torch.mean(numerator / denominator)


def binary_focal_loss(prob, target, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
    import torch

    prob = torch.clamp(prob.float(), eps, 1.0 - eps)
    target = target.float()
    fg = alpha * target * ((1.0 - prob) ** gamma) * torch.log(prob)
    bg = (1.0 - alpha) * (1.0 - target) * (prob ** gamma) * torch.log(1.0 - prob)
    return -torch.mean(fg + bg)


class DiceFocalLoss:
    def __init__(self, lambda_dice: float = 0.5, lambda_focal: float = 0.5, alpha: float = 0.25, gamma: float = 2.0):
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits_or_prob, target):
        import torch

        prob = torch.sigmoid(logits_or_prob) if logits_or_prob.min() < 0 or logits_or_prob.max() > 1 else logits_or_prob
        return self.lambda_dice * dice_loss(prob, target) + self.lambda_focal * binary_focal_loss(
            prob, target, alpha=self.alpha, gamma=self.gamma
        )
