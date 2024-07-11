"""
Dataloader for 7Scenes
"""

from pathlib import Path
import imageio.v3 as imageio
import numpy as np

__all__ = ["SevenScenesDataModule"]

from ._utils import GenericDataModule, LoadDataset


class SevenScenesDataModule(GenericDataModule):
    def __init__(
        self,
        *args,
        root: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            "sevenscenes",
            *args,
            root=root,
            min_depth=min_depth,
            max_depth=max_depth,
            load_dataset_cls=SceneScenesLoadSample,
            **kwargs,
        )


class SceneScenesLoadSample(LoadDataset):
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

        return {
            f"image{suffix}": img,
            f"depth{suffix}": depth,
            f"intrinsics{suffix}": intrinsics,
            f"extrinsics{suffix}": extrinsics,
        }
