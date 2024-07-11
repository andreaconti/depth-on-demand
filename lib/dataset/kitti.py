"""
Dataloader for TartanAir
"""

from pathlib import Path
import imageio.v3 as imageio
import numpy as np
from ._utils import GenericDataModule, LoadDataset
import pykitti


__all__ = ["KittiDataModule"]


class KittiDataModule(GenericDataModule):
    def __init__(
        self,
        *args,
        root_raw: str | Path,
        root_completion: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        load_hints_pcd: bool = False,
        **kwargs,
    ):
        # prepare base class
        super().__init__(
            "kitti",
            *args,
            root_raw=root_raw,
            root_completion=root_completion,
            min_depth=min_depth,
            max_depth=max_depth,
            load_hints_pcd=load_hints_pcd,
            load_dataset_cls=KittiLoadSample,
            **kwargs,
        )

    @staticmethod
    def augmenting_targets():
        out = {
            "image": "image",
            "depth": "depth",
            "hints": "depth",
            "extrinsics": "rt",
            "intrinsics": "intrinsics",
            "hints_pcd": "pcd",
        }
        for i in range(20):
            out |= {
                f"image_prev_{i}": "image",
                f"depth_prev_{i}": "depth",
                f"hints_prev_{i}": "depth",
                f"extrinsics_prev_{i}": "rt",
                f"intrinsics_prev_{i}": "intrinsics",
                f"hints_pcd_prev_{i}": "pcd",
            }
        return out


class KittiLoadSample(LoadDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kitti_meta = {}

    def _get_meta(self, root_raw: str, scan: str):
        if scan not in self._kitti_meta:
            date = "_".join(scan.split("_")[:3])
            scan_id = scan.split("_")[4]
            self._kitti_meta[scan] = pykitti.raw(root_raw, date, scan_id)
        return self._kitti_meta[scan]

    def load_sample(
        self,
        scan: str,
        idx: str,
        root_raw: str | Path,
        root_completion: str | Path,
        min_depth: float = 1e-3,
        max_depth: float = 100.0,
        load_hints_pcd: bool = False,
        suffix: str = "",
    ) -> dict:
        root_raw = Path(root_raw)
        root_compl = Path(root_completion)
        date = "_".join(scan.split("_")[:3])
        compl_path = next(root_compl.glob(f"*/{scan}/proj_depth"))

        # img
        image = imageio.imread(root_raw / date / scan / f"image_02/data/{idx}.png")

        # hints (lidar projected)
        hints = (
            imageio.imread(compl_path / f"velodyne_raw/image_02/{idx}.png")[
                ..., None
            ].astype(np.float32)
            / 256.0
        )
        hints[hints < min_depth] = 0.0
        hints[hints > max_depth] = 0.0

        # gt
        depth = (
            imageio.imread(compl_path / f"groundtruth/image_02/{idx}.png")[
                ..., None
            ].astype(np.float32)
            / 256.0
        )
        depth[depth < min_depth] = 0.0
        depth[depth > max_depth] = 0.0

        # intrinsics
        scan_meta = self._get_meta(str(root_raw), scan)
        intrinsics = scan_meta.calib.K_cam2.astype(np.float32).copy()

        # extrinsics
        extrinsics = (
            scan_meta.oxts[int(idx)].T_w_imu
            @ np.linalg.inv(scan_meta.calib.T_velo_imu)
            @ np.linalg.inv(
                scan_meta.calib.R_rect_20 @ scan_meta.calib.T_cam0_velo_unrect
            )
        ).astype(np.float32)

        # lidar pcd
        pcd = None
        if load_hints_pcd:
            pcd = scan_meta.get_velo(int(idx))[:, :3]
            pcd = (
                scan_meta.calib.T_cam2_velo
                @ np.pad(pcd, [(0, 0), (0, 1)], constant_values=1.0).T
            ).T[:, :3]
            n_points = pcd.shape[0]
            padding = 130000 - n_points
            pcd = np.pad(pcd, [(0, padding), (0, 0)]).astype(np.float32)

        # crop frames
        h, w, _ = depth.shape
        lc = (w - 1216) // 2
        rc = w - 1216 - lc
        image = image[-256:, lc:-rc]
        depth = depth[-256:, lc:-rc]
        hints = hints[-256:, lc:-rc]
        intrinsics[0, -1] -= lc
        intrinsics[1, -1] -= h - 256

        # compose output dict
        out = {
            f"image{suffix}": image,
            f"hints{suffix}": hints,
            f"depth{suffix}": depth,
            f"intrinsics{suffix}": intrinsics,
            f"extrinsics{suffix}": extrinsics,
        }
        if pcd is not None:
            out[f"hints_pcd{suffix}"] = pcd

        return out
