"""
Dataloader for TartanAir
"""

from pathlib import Path
import imageio.v3 as imageio
import json
import numpy as np
from ._utils import GenericDataModule, LoadDataset


__all__ = ["TartanairDataModule"]


class TartanairDataModule(GenericDataModule):
    def __init__(
        self,
        *args,
        root: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 100.0,
        **kwargs,
    ):
        super().__init__(
            "tartanair",
            *args,
            root=root,
            min_depth=min_depth,
            max_depth=max_depth,
            load_dataset_cls=TartanairLoadSample,
            **kwargs,
        )


class TartanairLoadSample(LoadDataset):
    def load_sample(
        self,
        scan: str,
        idx: str,
        root: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 100.0,
        suffix: str = "",
    ) -> dict:
        root = Path(root)
        img = imageio.imread(root / scan / f"{idx}.image.jpg")

        depth = imageio.imread(root / scan / f"{idx}.depth.png")[..., None].astype(
            np.float32
        )
        drange = json.load(open(root / scan / f"{idx}.depth_range.json", "rt"))
        vmin, vmax = drange["vmin"], drange["vmax"]
        depth = vmin + (depth / 65535) * (vmax - vmin)
        depth[depth < min_depth] = 0.0
        depth[depth > max_depth] = 0.0

        intrinsics = np.load(root / scan / f"{idx}.intrinsics.npy")
        extrinsics = np.load(root / scan / f"{idx}.position.npy")
        extrinsics = _ned_to_cam.dot(extrinsics).dot(_ned_to_cam.T)

        return {
            f"image{suffix}": img,
            f"depth{suffix}": depth,
            f"intrinsics{suffix}": intrinsics,
            f"extrinsics{suffix}": extrinsics,
        }


_ned_to_cam = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
