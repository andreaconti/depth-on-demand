"""
Evaluation procedure to compute metrics
on meshes
"""

import open3d as o3d
from pytorch3d.ops import knn_points
import numpy as np
from pathlib import Path
import torch
import warnings

__all__ = ["mesh_metrics"]


def mesh_metrics(
    mesh_prediction: o3d.geometry.TriangleMesh | str | Path,
    mesh_groundtruth: o3d.geometry.TriangleMesh | str | Path,
    world2grid: np.ndarray | str | Path | None = None,
    occlusion_mask: np.ndarray | str | Path | None = None,
    dist_threshold: float = 0.05,
    max_dist: float = 1.0,
    num_points_samples: int = 200000,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Takes in input the predicted and groundtruth meshes and computes a set
    of common metrics between them: accuracy and completion, precision and
    recall and their f1_score. To compute fair metrics the predicted mesh
    can be masked to remove areas not available in the groundtruth providing
    the occlusion mask and the world2grid parameters (both).
    """
    # read data if not already done
    assert (
        occlusion_mask is None
        and world2grid is None
        or occlusion_mask is not None
        and world2grid is not None
    ), "occlusion_mask and world2grid must be both provided"
    if isinstance(mesh_groundtruth, (str, Path)):
        mesh_groundtruth = o3d.io.read_triangle_mesh(str(mesh_groundtruth))
    if isinstance(mesh_prediction, (str, Path)):
        mesh_prediction = o3d.io.read_triangle_mesh(str(mesh_prediction))
    if world2grid is not None and isinstance(world2grid, (str, Path)):
        world2grid = np.loadtxt(world2grid, dtype=np.float32)
    if occlusion_mask is not None and isinstance(occlusion_mask, (str, Path)):
        occlusion_mask = np.load(occlusion_mask).astype(np.float32)

    # compute gt -> pred distance
    points_pred = torch.from_numpy(
        np.asarray(
            mesh_prediction.sample_points_uniformly(num_points_samples).points,
            dtype=np.float32,
        )
    )
    points_pred = points_pred.to(device)
    points_gt = torch.from_numpy(
        np.asarray(mesh_groundtruth.vertices, dtype=np.float32)
    ).to(device)
    world2grid = torch.from_numpy(world2grid).to(device)
    occlusion_mask = torch.from_numpy(occlusion_mask).to(device).float()
    gt2pred_dist = chamfer_distance(points_gt[None], points_pred[None], max_dist)

    # compute pred -> gt distance
    points_pred_filtered = filter_occluded_points(
        points_pred, world2grid, occlusion_mask, device
    )
    pred2gt_dist = chamfer_distance(
        points_pred_filtered[None], points_gt[None], max_dist
    )

    # compute metrics
    accuracy = torch.mean(pred2gt_dist)
    completion = torch.mean(gt2pred_dist)
    precision = (pred2gt_dist <= dist_threshold).float().mean()
    recall = (gt2pred_dist <= dist_threshold).float().mean()
    f1_score = 2 * precision * recall / (precision + recall)
    chamfer = 0.5 * (accuracy + completion)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "completion": completion,
        "chamfer": chamfer,
    }


def chamfer_distance(points1, points2, max_dist):
    l2dists = knn_points(points1, points2).dists
    sqdists = torch.where(l2dists > 0, torch.sqrt(l2dists), 0.0)
    sqdists = torch.minimum(sqdists, torch.full_like(sqdists, max_dist))
    return sqdists


def filter_occluded_points(points_pred, world2grid, occlusion_mask, device):
    dim_x = occlusion_mask.shape[0]
    dim_y = occlusion_mask.shape[1]
    dim_z = occlusion_mask.shape[2]
    num_points_pred = points_pred.shape[0]

    # Transform points to bbox space.
    R_world2grid = world2grid[:3, :3].view(1, 3, 3).expand(num_points_pred, -1, -1)
    t_world2grid = world2grid[:3, 3].view(1, 3, 1).expand(num_points_pred, -1, -1)

    points_pred_coords = (
        torch.matmul(R_world2grid, points_pred.view(num_points_pred, 3, 1))
        + t_world2grid
    ).view(num_points_pred, 3)

    # Normalize to [-1, 1]^3 space.
    # The world2grid transforms world positions to voxel centers, so we need to
    # use "align_corners=True".
    points_pred_coords[:, 0] /= dim_x - 1
    points_pred_coords[:, 1] /= dim_y - 1
    points_pred_coords[:, 2] /= dim_z - 1
    points_pred_coords = points_pred_coords * 2 - 1

    # Trilinearly interpolate occlusion mask.
    # Occlusion mask is given as (x, y, z) storage, but the grid_sample method
    # expects (c, z, y, x) storage.
    visibility_mask = 1 - occlusion_mask.view(dim_x, dim_y, dim_z)
    visibility_mask = visibility_mask.permute(2, 1, 0).contiguous()
    visibility_mask = visibility_mask.view(1, 1, dim_z, dim_y, dim_x)

    points_pred_coords = points_pred_coords.view(1, 1, 1, num_points_pred, 3)

    points_pred_visibility = torch.nn.functional.grid_sample(
        visibility_mask,
        points_pred_coords,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).to(device)

    points_pred_visibility = points_pred_visibility.view(num_points_pred)

    eps = 1e-5
    points_pred_visibility = points_pred_visibility >= 1 - eps

    # Filter occluded predicted points.
    if points_pred_visibility.sum() == 0:
        # If no points are visible, we keep the original points, otherwise
        # we would penalize the sample as if nothing is predicted.
        warnings.warn(
            "All points occluded, keeping all predicted points!", RuntimeWarning
        )
        points_pred_visible = points_pred.clone()
    else:
        points_pred_visible = points_pred[points_pred_visibility]

    return points_pred_visible
