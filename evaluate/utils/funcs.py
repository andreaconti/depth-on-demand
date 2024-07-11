import torch
import kornia
from torch import Tensor
import random
import torch.nn.functional as F
from typing import Callable
from lib.visualize.matplotlib import GridFigure
from kornia.morphology import dilation
import io
import itertools
from contextlib import redirect_stdout
import kornia
import numpy as np
from kornia.geometry.quaternion import QuaternionCoeffOrder
from kornia.geometry.conversions import (
    euler_from_quaternion,
    rotation_matrix_to_quaternion,
)
import re
from lib.visualize.matplotlib import color_depth

__all__ = ["sparsity_depth", "prepare_input"]


def prepare_input(
    batch,
    hints_density: float | int = 500,
    hints_from_pcd: bool = False,
    source_postfix: str = "",
    hints_postfix: str = "",
    predict_pose: Callable | None = None,
    pose_noise_std_mult: float = 0.0,
):
    # prepare source and target views
    image_target = batch["image"]
    image_source = batch[f"image{source_postfix}"]
    intrinsics = torch.stack(
        [batch["intrinsics"], batch[f"intrinsics{source_postfix}"]], 1
    )

    # sample hints on their frame
    hints_hints = find_hints(batch, hints_postfix, hints_density, hints_from_pcd)
    if hints_postfix != source_postfix:
        hints_src = find_hints(batch, source_postfix, hints_density, hints_from_pcd)
    else:
        hints_src = hints_hints

    # prepare pose
    if predict_pose is None:
        pose_src_tgt = (
            inv_pose(batch["extrinsics"]) @ batch[f"extrinsics{source_postfix}"]
        )
        pose_hints_tgt = (
            inv_pose(batch["extrinsics"]) @ batch[f"extrinsics{hints_postfix}"]
        )
        if pose_noise_std_mult > 0.0:
            if source_postfix == hints_postfix:
                trasl_std = pose_noise_std_mult * pose_src_tgt[:, :3, -1].abs()
                rot_std = pose_noise_std_mult * pose_to_angles(pose_src_tgt).abs()
                pnoise = pose_noise(rot_std, trasl_std)
                pose_src_tgt = pnoise @ pose_src_tgt
                pose_hints_tgt = pnoise @ pose_hints_tgt
            else:
                trasl_std = pose_noise_std_mult * pose_src_tgt[:, :3, -1].abs()
                rot_std = pose_noise_std_mult * pose_to_angles(pose_src_tgt).abs()
                pnoise = pose_noise(rot_std, trasl_std)
                pose_src_tgt = pnoise @ pose_src_tgt

                trasl_std = pose_noise_std_mult * pose_hints_tgt[:, :3, -1].abs()
                rot_std = pose_noise_std_mult * pose_to_angles(pose_hints_tgt).abs()
                pnoise = pose_noise(rot_std, trasl_std)
                pose_hints_tgt = pnoise @ pose_hints_tgt

    else:
        pose_hints_tgt = predict_pose(
            image_target,
            batch[f"image{hints_postfix}"],
            hints_hints,
            batch["intrinsics"],
            batch[f"intrinsics{hints_postfix}"],
        )

        if hints_postfix != source_postfix:
            pose_src_tgt = predict_pose(
                image_target,
                batch[f"image{source_postfix}"],
                hints_src,
                batch["intrinsics"],
                batch[f"intrinsics{source_postfix}"],
            )
        else:
            pose_src_tgt = pose_hints_tgt

    # project hints
    if not hints_from_pcd:
        h, w = image_target.shape[-2:]
        hints, _ = project_depth(
            hints_hints.to(torch.float32),
            batch[f"intrinsics{hints_postfix}"].to(torch.float32),
            batch[f"intrinsics"].to(torch.float32),
            pose_hints_tgt.to(torch.float32),
            torch.zeros_like(hints_hints, dtype=torch.float32),
        )
    else:
        b, _, h, w = image_target.shape
        device, dtype = image_target.device, image_target.dtype
        hints = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        project_pcd(
            batch[f"hints_pcd{hints_postfix}"],
            batch[f"intrinsics"],
            hints,
            pose_hints_tgt,
        )

    return {
        "target": image_target,
        "source": image_source,
        "pose_src_tgt": pose_src_tgt,
        "intrinsics": intrinsics,
        "hints": hints,
        "source_hints": hints_src,
        "hints_hints": hints_hints,
    }


def prepare_input_onnx(
    batch,
    hints_density: float | int = 500,
    hints_from_pcd: bool = False,
    source_postfix: str = "",
    hints_postfix: str = "",
    predict_pose: Callable | None = None,
    pose_noise_std_mult: float = 0.0,
):

    # take the useful base outputs
    base_input = prepare_input(
        batch,
        hints_density,
        hints_from_pcd,
        source_postfix,
        hints_postfix,
        predict_pose,
        pose_noise_std_mult,
    )
    out = {
        "target": base_input["target"],
        "source": base_input["source"],
        "pose_src_tgt": base_input["pose_src_tgt"],
        "intrinsics": base_input["intrinsics"],
    }

    # compute the other outputs required by the onnx model
    b, _, h, w = base_input["hints_hints"].shape
    device = base_input["hints_hints"].device
    hints8, subpix8 = project_depth(
        base_input["hints_hints"],
        batch["intrinsics"],
        adjust_intrinsics(batch["intrinsics"], 1 / 8),
        torch.eye(4, device=device)[None],
        torch.zeros(
            1, 1, h // 8, w // 8, device=device, dtype=base_input["hints_hints"].dtype
        ),
    )
    init_depth = create_init_depth(hints8)
    out |= {
        "hints8": hints8,
        "subpix8": subpix8,
        "init_depth": init_depth,
    }
    return out, base_input


def adjust_intrinsics(intrinsics: Tensor, factor: float = 0.125):
    intrinsics = intrinsics.clone()
    intrinsics[..., :2, :] = intrinsics[..., :2, :] * factor
    return intrinsics


def create_init_depth(hints8, fallback_init_depth=2.0, min_hints_for_init=20):
    batch_size = hints8.shape[0]
    mean_hints = (
        torch.ones(batch_size, dtype=hints8.dtype, device=hints8.device)
        * fallback_init_depth
    )
    for bi in range(batch_size):
        hints_mask = hints8[bi, 0] > 0
        if hints_mask.sum() > min_hints_for_init:
            mean_hints[bi] = hints8[bi, 0][hints8[bi, 0] > 0].mean()
    depth = torch.ones_like(hints8) * mean_hints[:, None, None, None]
    depth = torch.where(hints8 > 0, hints8, depth)
    return depth


def find_hints(
    batch,
    postfix,
    hints_density,
    from_pcd: bool = False,
    project_pose: Tensor | None = None,
):
    if not from_pcd:
        if f"hints{postfix}" in batch:
            sparse_hints = batch[f"hints{postfix}"]
        else:
            sparse_hints = sparsify_depth(batch[f"depth{postfix}"], hints_density)
        return sparse_hints
    else:
        hints_pcd = batch[f"hints_pcd{postfix}"]
        device, dtype = hints_pcd.device, hints_pcd.dtype
        b, _, h, w = batch["image"].shape
        sparse_hints = project_pcd(
            batch[f"hints_pcd{postfix}"],
            batch["intrinsics"],
            torch.zeros(b, 1, h, w, device=device, dtype=dtype),
            project_pose,
        )
        return sparse_hints


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


def inv_pose(pose: torch.Tensor) -> torch.Tensor:
    rot_inv = pose[:, :3, :3].permute(0, 2, 1)
    tr_inv = -rot_inv @ pose[:, :3, -1:]
    pose_inv = torch.eye(4, dtype=pose.dtype, device=pose.device)[None]
    pose_inv = pose_inv.repeat(pose.shape[0], 1, 1)
    pose_inv[:, :3, :3] = rot_inv
    pose_inv[:, :3, -1:] = tr_inv
    return pose_inv


def pose_distance(
    pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot = pose[:, :3, :3]
    trasl = pose[:, :3, 3]
    R_trace = rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    r_measure = torch.sqrt(
        2 * (1 - torch.minimum(torch.ones_like(R_trace) * 3.0, R_trace) / 3)
    )
    t_measure = torch.norm(trasl, dim=1)
    combined_measure = torch.sqrt(t_measure**2 + r_measure**2)

    return combined_measure, r_measure, t_measure


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


def project_depth(
    depth: Tensor,
    intrinsics_from: Tensor,
    intrinsics_to: Tensor,
    extrinsics_from_to: Tensor,
    depth_to: Tensor,
) -> tuple[Tensor, Tensor]:
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
    u_subpix, v_subpix = torch.unbind(xyz_pcd_to[..., :2] / xyz_pcd_to[..., -1:], -1)
    u, v = torch.round(u_subpix).to(torch.long), torch.round(v_subpix).to(torch.long)
    mask = (u >= 0) & (v >= 0) & (u < w_to) & (v < h_to) & (depth[:, 0] > 0)
    subpix = torch.zeros(
        batch, 2, h_to, w_to, dtype=depth_to.dtype, device=depth_to.device
    )
    for b in range(batch):
        used_mask = mask[b]
        used_u, used_v = u[b, used_mask], v[b, used_mask]
        prev_depths = depth_to[b, 0, used_v, used_u]
        new_depths = xyz_pcd_to[b, used_mask][:, -1]
        merged_depths = torch.where(prev_depths == 0, new_depths, prev_depths)
        depth_to[b, 0, used_v, used_u] = merged_depths
        subpix[b, 0, used_v, used_u] = u_subpix[b, used_mask] / w_to
        subpix[b, 1, used_v, used_u] = v_subpix[b, used_mask] / h_to

    return depth_to, subpix


def prepare_plot(
    gt_depth,
    pred_depth,
    image_target,
    image_source,
    hints,
    gt_dilation: int | None = None,
    hints_dilation: int | None = None,
    depth_vmin: float = 0.0,
    depth_vmax: float = 10.0,
) -> GridFigure:
    b, _, h, w = image_source.shape
    grid = GridFigure(b, 4, size=(h // 2, w // 2))

    if hints_dilation:
        hints = dilation(
            hints, torch.ones((hints_dilation, hints_dilation), device=hints.device)
        )
    if gt_dilation:
        gt_depth = dilation(
            gt_depth, torch.ones((gt_dilation, gt_dilation), device=gt_depth.device)
        )

    dkwargs = {
        "vmin": depth_vmin,
        "vmax": depth_vmax,
        "cmap": "magma_r",
    }

    idx = 0
    for bi in range(b):

        # source view
        grid.imshow(idx := idx + 1, image_source[bi])

        # target view + hints
        img_target = image_target[bi]
        img_target = (img_target - img_target.min()) / (
            img_target.max() - img_target.min()
        )
        img_target = (
            (img_target.cpu().numpy() * 255).astype(np.uint8).transpose([1, 2, 0])
        )
        grid.imshow(idx := idx + 1, img_target, norm=False)
        hints_color = color_depth(hints[bi, 0].cpu().numpy(), **dkwargs)
        hints_color[..., :3] = hints_color[..., :3] / hints_color[..., :3].max()
        hints_color[..., -1:] = hints_color[..., -1:] / hints_color[..., -1:].max()
        grid.imshow(idx, hints_color, norm=False)

        # prediction
        grid.imshow(idx := idx + 1, pred_depth[bi], **dkwargs, norm=False)

        # gt view
        gt_depth_imshow = torch.where(gt_depth[bi] > 0, gt_depth[bi], depth_vmax)
        grid.imshow(idx := idx + 1, gt_depth_imshow, **dkwargs, norm=False)

    return grid


def pose_to_angles(pose: torch.Tensor):
    out_angles = []
    angles = euler_from_quaternion(
        *rotation_matrix_to_quaternion(
            pose[:, :3, :3].contiguous(), order=QuaternionCoeffOrder.WXYZ
        )[0]
    )
    angles = torch.stack(angles, 0)
    out_angles.append(angles)
    return torch.rad2deg(torch.stack(out_angles, 0))


def pose_noise(rot_std: torch.Tensor, trasl_std: torch.Tensor):
    std_vec = torch.cat([rot_std, trasl_std], 1)
    mean_vec = torch.zeros_like(std_vec)
    normal_noise = torch.normal(mean_vec, std_vec)
    noises = []
    for bi in range(rot_std.shape[0]):
        rot_noise = torch.eye(4).to(normal_noise.dtype).to(normal_noise.device)
        quat = kornia.geometry.quaternion_from_euler(
            torch.deg2rad(normal_noise[bi, 0]),
            torch.deg2rad(normal_noise[bi, 1]),
            torch.deg2rad(normal_noise[bi, 2]),
        )
        rot_noise[:3, :3] = kornia.geometry.quaternion_to_rotation_matrix(
            torch.stack(quat, -1), QuaternionCoeffOrder.WXYZ
        )
        rot_noise[:3, -1] += normal_noise[bi, 3:]
        noises.append(rot_noise)
    return torch.stack(noises, 0)


def find_parameters():
    ipy = get_ipython()  # type: ignore
    out = io.StringIO()
    with redirect_stdout(out):
        ipy.magic("history")
    x = out.getvalue().split("\n")
    param_lines = list(
        itertools.takewhile(
            lambda s: s != "##% END",
            itertools.dropwhile(lambda s: s != "##% PARAMETERS", x),
        )
    )
    params = []
    for param in param_lines:
        if match := re.match("([a-zA-Z_0-9]+).*=", param):
            name = match.groups()[0]
            params.append(name)
    return params
