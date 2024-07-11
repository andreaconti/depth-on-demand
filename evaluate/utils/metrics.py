"""
Used metrics in this project
"""

import torch
from torch import Tensor

__all__ = ["mae", "rmse", "sq_rel", "rel_thresh", "compute_metrics"]


def mae(pred: Tensor, gt: Tensor) -> Tensor:
    return torch.mean(torch.abs(pred - gt))


def rmse(pred: Tensor, gt: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(torch.square(pred - gt)))


def sq_rel(pred: Tensor, gt: Tensor, eps: float = 0.0) -> Tensor:
    return torch.mean(torch.square(pred - gt) / (gt + eps))


def abs_rel(pred: Tensor, gt: Tensor, eps: float = 0.0) -> Tensor:
    return torch.mean(torch.abs(pred - gt) / (gt + eps))


def rel_thresh(pred: Tensor, gt: Tensor, sigma: float) -> Tensor:
    rel = torch.maximum(gt / pred, pred / gt) < sigma
    rel = torch.mean(rel.float())
    return rel


def compute_metrics_depth(
    pred: Tensor, gt: Tensor, label: str = ""
) -> dict[str, Tensor]:
    label = label if not label else label + "/"
    return {
        f"{label}mae": mae(pred, gt),
        f"{label}rmse": rmse(pred, gt),
        f"{label}sq_rel": sq_rel(pred, gt),
        f"{label}abs_rel": abs_rel(pred, gt),
        f"{label}rel_thresh_1.05": rel_thresh(pred, gt, 1.05),
        f"{label}rel_thresh_1.25": rel_thresh(pred, gt, 1.25),
    }
