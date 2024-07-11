"""
Utilities to build meshes by RGBD frames
"""

import open3d as o3d
import numpy as np
import imageio.v3 as imageio
from typing import Literal
from pathlib import Path

__all__ = ["TSDFFusion"]


class TSDFFusion:
    """
    Fusion utility to build a mesh from multiple RGBD calibrated and with pose
    frames through TSDF.
    """

    def __init__(
        self,
        engine: Literal["open3d", "open3d-tensor"] = "open3d",
        # volume info
        voxel_length=0.04,
        sdf_trunc=3 * 0.04,
        # defaults
        color_type: o3d.pipelines.integration.TSDFVolumeColorType = o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_scale: float = 1.0,
        depth_trunc: float = 3.0,
        convert_rgb_to_intensity: bool = False,
        device: str = "cuda:0",
    ):
        assert engine in ["open3d", "open3d-tensor"]
        self.engine = engine
        self.device = o3d.core.Device(device.upper())
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.voxel_length = voxel_length
        self.convert_rgb_to_intensity = convert_rgb_to_intensity

        if engine == "open3d":
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=voxel_length,
                sdf_trunc=sdf_trunc,
                color_type=color_type,
            )
        elif engine == "open3d-tensor":
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1,), (1,), (3,)),
                voxel_size=voxel_length,
                device=self.device,
            )

    def convert_intrinsics(
        self,
        intrinsics: np.ndarray,
        height: int | None = None,
        width: int | None = None,
    ) -> o3d.camera.PinholeCameraIntrinsic | o3d.core.Tensor:
        intrins = o3d.camera.PinholeCameraIntrinsic(
            width if width else int(intrinsics[0, 2] * 2),
            height if height else int(intrinsics[1, 2] * 2),
            intrinsics,
        )
        if self.engine == "open3d-tensor":
            intrins = o3d.core.Tensor(intrins.intrinsic_matrix, o3d.core.Dtype.Float64)
        return intrins

    def read_rgbd(
        self,
        image: str | Path,
        depth: str | Path,
        depth_trunc: float | None = None,
        convert_rgb_to_intensity: bool | None = None,
    ) -> o3d.geometry.RGBDImage | tuple[o3d.core.Tensor, o3d.core.Tensor]:
        if self.engine == "open3d":
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                imageio.imread(image),
                imageio.imread(depth),
                depth_trunc=self.depth_trunc if depth_trunc is None else depth_trunc,
                convert_rgb_to_intensity=self.convert_rgb_to_intensity
                if convert_rgb_to_intensity is None
                else convert_rgb_to_intensity,
            )
        else:
            color = o3d.t.io.read_image(image).to(self.device)
            depth = o3d.t.io.read_image(depth).to(self.device)
            return color, depth

    def convert_rgbd(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        depth_scale: float | None = None,
        depth_trunc: float | None = None,
        convert_rgb_to_intensity: bool | None = None,
    ) -> o3d.geometry.RGBDImage | tuple[o3d.core.Tensor, o3d.core.Tensor]:
        if self.engine == "open3d":
            if image.dtype != np.uint8:
                image = (image - image.min()) / (image.max() - image.min())
                image = np.round(image * 255).astype(np.uint8)
            image = o3d.geometry.Image(np.ascontiguousarray(image))
            depth = o3d.geometry.Image(depth[..., 0])
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                image,
                depth,
                depth_scale=self.depth_scale if depth_scale is None else depth_scale,
                depth_trunc=self.depth_trunc if depth_trunc is None else depth_trunc,
                convert_rgb_to_intensity=self.convert_rgb_to_intensity
                if convert_rgb_to_intensity is None
                else convert_rgb_to_intensity,
            )
            return rgbd
        else:
            image = o3d.t.geometry.Image(
                o3d.core.Tensor(
                    np.ascontiguousarray(image).astype(np.float32), device=self.device
                )
            )
            depth = o3d.t.geometry.Image(
                o3d.core.Tensor(depth[..., 0], device=self.device)
            )
            return image, depth

    def integrate_rgbd(
        self,
        image: np.ndarray | str | Path,
        depth: np.ndarray | str | Path,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        **kwargs,
    ):
        if isinstance(image, np.ndarray) and isinstance(depth, np.ndarray):
            rgbd = self.convert_rgbd(image, depth, **kwargs)
        elif isinstance(image, (str, Path)) and isinstance(depth, (str, Path)):
            rgbd = self.read_rgbd(image, depth, **kwargs)
        else:
            raise ValueError(
                f"not supported image and depth types ({type(image), type(depth)})"
            )

        if self.engine == "open3d":
            w, h = rgbd.color.get_max_bound().astype(int)
            self.volume.integrate(
                rgbd,
                self.convert_intrinsics(intrinsics, h, w),
                np.linalg.inv(extrinsics),
            )
        else:
            color, depth = rgbd
            extrins = o3d.core.Tensor(np.linalg.inv(extrinsics), o3d.core.Dtype.Float64)
            intrins = o3d.core.Tensor(intrinsics, o3d.core.Dtype.Float64)
            coords = self.volume.compute_unique_block_coordinates(
                depth,
                intrins,
                extrins,
                depth_scale=self.depth_scale,
                depth_max=self.depth_trunc,
            )
            self.volume.integrate(
                coords,
                depth,
                color,
                intrins,
                extrins,
                depth_scale=float(self.depth_scale),
                depth_max=float(self.depth_trunc),
            )

    def write_triangle_mesh(self, path: str | Path):
        return o3d.io.write_triangle_mesh(str(path), self.triangle_mesh())

    def triangle_mesh(self):
        if self.engine == "open3d":
            return self.volume.extract_triangle_mesh()
        else:
            return self.volume.extract_triangle_mesh().to_legacy()

    def reset(self):
        if self.engine == "open3d":
            self.volume.reset()
        else:
            self.volume = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
                attr_channels=((1,), (1,), (3,)),
                voxel_size=self.voxel_length,
                device=self.device,
            )
