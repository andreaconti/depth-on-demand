"""
Pose estimation by means of RGBD frame + RGB py means of
LightGlue and PnP + LO-RANSAC
"""

import torch
from torch import Tensor
from lightglue import DISK, LightGlue
import poselib
import numpy as np
from kornia.morphology import dilation

__all__ = ["PoseEstimation"]


class PoseEstimation(torch.nn.Module):
    def __init__(
        self,
        max_num_keypoints: int | None = None,
        dilate_depth: int | None = None,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.extractor = DISK(max_num_keypoints=max_num_keypoints).eval()
        self.matcher = LightGlue(features="disk").eval()
        if compile:
            self.matcher.compile()
        self.dilate_depth = dilate_depth

        # kwargs default
        if "ransac_max_reproj_error" not in kwargs:
            kwargs["ransac_max_reproj_error"] = 1.0

        self.kwargs = kwargs

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        image0: Tensor,
        image1: Tensor,
        depth1: Tensor,
        intrinsics0: Tensor,
        intrinsics1: Tensor,
    ):
        # depth dilation for easier matching
        if self.dilate_depth is not None:
            depth1 = dilation(
                depth1,
                torch.ones(
                    [self.dilate_depth, self.dilate_depth],
                    device=depth1.device,
                    dtype=depth1.dtype,
                ),
            )

        with torch.no_grad():
            batch_size = image0.shape[0]
            poses = []
            for b in range(batch_size):
                feats0 = self.extractor.extract(image0[b])
                feats1 = self.extractor.extract(image1[b])
                matches01 = self.matcher({"image0": feats0, "image1": feats1})
                matches01 = matches01["matches"]
                matches0 = _to_numpy(feats0["keypoints"][0][matches01[0][:, 0]])
                matches1 = _to_numpy(feats1["keypoints"][0][matches01[0][:, 1]])

                depths1 = _to_numpy(depth1)[
                    b,
                    0,
                    matches1[:, 1].round().astype(int),
                    matches1[:, 0].round().astype(int),
                ]
                valid_mask = depths1 > 0
                pose = np.eye(4)
                pose[:3] = poselib.estimate_absolute_pose(
                    matches0[valid_mask],
                    _depth_to_3d(
                        depths1[valid_mask],
                        matches1[valid_mask],
                        _to_numpy(intrinsics1[b]),
                    ),
                    {
                        "model": "SIMPLE_PINHOLE",
                        "width": image0.shape[-1],
                        "height": image0.shape[-2],
                        "params": [
                            intrinsics0[b, 0, 0].item(),  # fx
                            intrinsics0[b, 0, 2].item(),  # cx
                            intrinsics0[b, 1, 2].item(),  # cy
                        ],
                    },
                    {
                        "_".join(k.split("_")[1:]): v
                        for k, v in self.kwargs.items()
                        if k.startswith("ransac_")
                    },
                    {
                        "_".join(k.split("_")[1:]): v
                        for k, v in self.kwargs.items()
                        if k.startswith("ransac_")
                    },
                )[0].Rt
                poses.append(pose)
            return (
                torch.from_numpy(np.stack(poses, 0)).to(image0.device).to(image0.dtype)
            )


# utils


def _to_numpy(x: Tensor):
    return x.detach().cpu().numpy()


def _depth_to_3d(depths: np.ndarray, coords: np.ndarray, intrinsics: np.ndarray):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x3d = depths * (coords[:, 0] - cx) / fx
    y3d = depths * (coords[:, 1] - cy) / fy
    return np.stack([x3d, y3d, depths], -1)


def _norm(img: Tensor):
    return (img - img.min()) / (img.max() - img.min())
