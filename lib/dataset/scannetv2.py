"""
Dataloader for ScanNetV2
"""

from pathlib import Path
import imageio.v3 as imageio
import numpy as np

__all__ = ["ScanNetV2DataModule"]

from ._utils import GenericDataModule, LoadDataset


class ScanNetV2DataModule(GenericDataModule):
    def __init__(
        self,
        *args,
        root: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            "scannetv2",
            *args,
            root=root,
            min_depth=min_depth,
            max_depth=max_depth,
            load_dataset_cls=ScanNetV2LoadSample,
            **kwargs,
        )


class ScanNetV2LoadSample(LoadDataset):
    def load_sample(
        self,
        scan: str,
        idx: str,
        root,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        suffix: str = "",
    ) -> dict:
        root = Path(root)
        img = imageio.imread(root / scan / f"{idx}.image.jpg")
        depth = (
            imageio.imread(root / scan / f"{idx}.depth.png")[..., None] / 1000
        ).astype(np.float32)
        depth[(depth < min_depth) | (depth > max_depth)] = 0.0
        intrinsics = np.load(root / scan / f"{idx}.intrinsics.npy")
        extrinsics = np.load(root / scan / f"{idx}.extrinsics.npy")

        out = {}
        if not suffix:
            ply_path = root / f"{scan}_vh_clean.ply"
            occl_path = root / f"{scan}_occlusion_mask.npy"
            world2grid_path = root / f"{scan}_world2grid.txt"
            if ply_path.exists():
                out["gt_mesh_path"] = str(ply_path)
            if occl_path.exists():
                out["gt_mesh_occl_path"] = str(occl_path)
            if world2grid_path.exists():
                out["gt_mesh_world2grid_path"] = str(world2grid_path)

        return out | {
            f"image{suffix}": img,
            f"depth{suffix}": depth,
            f"intrinsics{suffix}": intrinsics,
            f"extrinsics{suffix}": extrinsics,
        }
