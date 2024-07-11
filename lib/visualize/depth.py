"""
Utilities to visualize depth and pointclouds
"""

import numpy as np
from pathlib import Path
from torch import Tensor

try:
    import plyfile  # type: ignore
    from kornia.geometry.depth import depth_to_3d  # type: ignore
    from kornia.geometry import transform_points  # type: ignore
except ImportError:
    raise ImportError("To use depth visualize utilities install plyfile and kornia")

__all__ = [
    "save_depth_ply",
    "save_pcd_ply",
    "save_depth_overlap_ply",
]


def _norm_color(color):
    color = (color - color.min()) / (1e-6 + np.abs(color.max() - color.min()))
    return color * 255


def save_depth_ply(
    filename: str | Path,
    depth: Tensor,
    intrinsics: Tensor,
    color: Tensor | None = None,
    _output_plyelement: bool = False,
):
    """
    Saves a depth map with optionally colors in a ply file,
    depth is of shape 1 x H x W and colors 3 x H x W if provided,
    """
    mask = depth[0] > 0
    pcd = depth_to_3d(depth[None], intrinsics[None]).permute(0, 2, 3, 1)[0][mask]
    if color is not None:
        color = color.permute(1, 2, 0)[mask]
    return save_pcd_ply(filename, pcd, color, _output_plyelement)


def save_pcd_ply(
    filename: str | Path,
    pcd: Tensor,
    color: Tensor | None = None,
    _output_plyelement: bool = False,
):
    """
    Saves a a point cloud with optionally colors in a ply file,
    pcd is of shape N x 3 and colors N x 3 if provided
    """

    pcd = pcd.cpu().numpy()
    if color is not None:
        color = _norm_color(color.cpu().numpy())
    else:
        color = np.zeros_like(pcd)
        color[:, 0] += 255
    pcd = np.array(
        list(
            zip(
                pcd[:, 0],
                pcd[:, 1],
                pcd[:, 2],
                color[:, 0],
                color[:, 1],
                color[:, 2],
            )
        ),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )

    if _output_plyelement:
        return plyfile.PlyElement.describe(pcd, "vertex")
    else:
        plyfile.PlyData([plyfile.PlyElement.describe(pcd, "vertex")]).write(filename)


def save_depth_overlap_ply(
    filename: str | Path,
    depths: list[Tensor],
    colors: list[Tensor],
    intrinsics: list[Tensor] | None = None,
    extrinsics: list[Tensor] | None = None,
):
    """
    Takes a list of pointclouds or depth maps and their color map and saves all of them
    as a whole point cloud in ply format
    """

    if intrinsics is None:
        intrinsics = [None] * len(depths)

    elems = []
    for idx, (d, c, i) in enumerate(zip(depths, colors, intrinsics)):
        if d.dim() == 2:
            if extrinsics is None:
                elems.append(save_pcd_ply(None, d, c, True).data)
            else:
                d = transform_points(extrinsics[idx][None], d[None])[0]
                elems.append(save_pcd_ply(None, d, c, True).data)
        else:
            if extrinsics is None:
                elems.append(save_depth_ply(None, d, i, c, True).data)
            else:
                mask = d[0] > 0
                d = depth_to_3d(d[None], i[None]).permute(0, 2, 3, 1)[0][mask]
                d = transform_points(extrinsics[idx][None], d[None])[0]
                c = c.permute(1, 2, 0)[mask]
                elems.append(save_pcd_ply(None, d, c, True).data)

    plyelem = plyfile.PlyElement.describe(np.concatenate(elems, 0), "vertex")
    plyfile.PlyData([plyelem]).write(filename)
