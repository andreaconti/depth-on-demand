import torch
import random
import kornia
from torch import Tensor

__all__ = [
    "sparsify_depth",
    "project_depth",
    "project_pcd",
    "inv_pose",
]


def sparsify_depth(
    depth: torch.Tensor, hints_perc: float | tuple[float, float] | int = 0.03
) -> torch.Tensor:
    if isinstance(hints_perc, tuple | list):
        hints_perc = random.uniform(*hints_perc)

    if hints_perc < 1.0:
        sparse_map = torch.rand_like(depth) < hints_perc
        sparse_depth = torch.where(
            sparse_map, depth, torch.tensor(0.0, dtype=depth.dtype, device=depth.device)
        )
        return sparse_depth
    else:
        b = depth.shape[0]
        idxs = torch.nonzero(depth[:, 0])
        idxs = idxs[torch.randperm(len(idxs))]
        sparse_depth = torch.zeros_like(depth)
        for bi in range(b):
            bidxs = idxs[idxs[:, 0] == bi][:hints_perc]
            sparse_depth[bi, 0, bidxs[:, 1], bidxs[:, 2]] = depth[
                bi, 0, bidxs[:, 1], bidxs[:, 2]
            ]
        return sparse_depth


def project_depth(
    depth: Tensor,
    intrinsics_from: Tensor,
    intrinsics_to: Tensor,
    extrinsics_from_to: Tensor,
    depth_to: Tensor,
) -> Tensor:
    # project the depth in 3D
    batch, _, h, w = depth.shape
    xyz_pcd_from = kornia.geometry.depth_to_3d(depth, intrinsics_from).permute(
        0, 2, 3, 1
    )
    xyz_pcd_to = intrinsics_to @ (
        (
            extrinsics_from_to
            @ torch.nn.functional.pad(
                xyz_pcd_from.view(batch, -1, 3), [0, 1], "constant", 1.0
            ).permute(0, 2, 1)
        )[:, :3]
    )
    xyz_pcd_to = xyz_pcd_to.permute(0, 2, 1).view(batch, h, w, 3)

    # project depth to 2D
    h_to, w_to = depth_to.shape[-2:]
    u, v = torch.unbind(xyz_pcd_to[..., :2] / xyz_pcd_to[..., -1:], -1)
    u, v = torch.round(u).to(torch.long), torch.round(v).to(torch.long)
    mask = (u >= 0) & (v >= 0) & (u < w_to) & (v < h_to) & (depth[:, 0] > 0)

    for b in range(batch):
        used_mask = mask[b]
        used_u, used_v = u[b, used_mask], v[b, used_mask]
        prev_depths = depth_to[b, 0, used_v, used_u]
        new_depths = xyz_pcd_to[b, used_mask][:, -1]
        merged_depths = torch.where(prev_depths == 0, new_depths, prev_depths)
        depth_to[b, 0, used_v, used_u] = merged_depths

    return depth_to


def project_pcd(
    xyz_pcd: torch.Tensor,
    intrinsics_to: torch.Tensor,
    depth_to: torch.Tensor,
    extrinsics_from_to: torch.Tensor | None = None,
) -> torch.Tensor:
    # transform pcd
    batch, _, _ = xyz_pcd.shape
    if extrinsics_from_to is None:
        extrinsics_from_to = torch.eye(4, dtype=xyz_pcd.dtype, device=xyz_pcd.device)[
            None
        ].repeat(batch, 1, 1)
    xyz_pcd_to = (
        intrinsics_to
        @ (
            extrinsics_from_to
            @ torch.nn.functional.pad(xyz_pcd, [0, 1], "constant", 1.0).permute(0, 2, 1)
        )[:, :3]
    )

    # project depth to 2D
    h_to, w_to = depth_to.shape[-2:]
    u, v = torch.unbind(xyz_pcd_to[:, :2] / xyz_pcd_to[:, -1:], 1)
    u, v = torch.round(u).to(torch.long), torch.round(v).to(torch.long)
    mask = (u >= 0) & (v >= 0) & (u < w_to) & (v < h_to)

    for b in range(batch):
        used_mask = mask[b]
        used_u, used_v = u[b, used_mask], v[b, used_mask]
        prev_depths = depth_to[b, 0, used_v, used_u]
        new_depths = xyz_pcd_to[b, :, used_mask][-1]
        merged_depths = torch.where(
            (prev_depths == 0) & (new_depths > 0), new_depths, prev_depths
        )
        depth_to[b, 0, used_v, used_u] = merged_depths

    return depth_to


def inv_pose(pose: torch.Tensor) -> torch.Tensor:
    rot_inv = pose[:, :3, :3].permute(0, 2, 1)
    tr_inv = -rot_inv @ pose[:, :3, -1:]
    pose_inv = torch.eye(4, dtype=pose.dtype, device=pose.device)[None]
    pose_inv = pose_inv.repeat(pose.shape[0], 1, 1)
    pose_inv[:, :3, :3] = rot_inv
    pose_inv[:, :3, -1:] = tr_inv
    return pose_inv
