from collections import defaultdict
from contextlib import contextmanager
from time import sleep
from typing import Any

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as ops
from kornia.geometry import depth_to_3d


class Visualize3D:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        self._server = viser.ViserServer(host, port)
        self.mesh = None
        self._steps = defaultdict(lambda: {})
        self._info = {}
        self._gui = defaultdict(lambda: {})
        self._gui_funcs = _GuiFuncs(self._server)

    def _convert_pose(self, pose: np.ndarray | None = None):
        if pose is None:
            return {}
        else:
            pose = ops.SE3.from_matrix(pose)
            return {
                "wxyz": pose.wxyz_xyz[:4],
                "position": pose.wxyz_xyz[4:],
            }

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> list[int]:
        steps = []
        for step in self._steps.keys():
            steps.append(int(step.split("_")[-1]))
        return sorted(steps)

    def has_step(self, step: int, name: str):
        with self._server.atomic():
            step_data = self._steps.get(f"step_{step}", None)
            if not step_data:
                raise KeyError(f"step {step} not exists")
            return name in step_data

    def step_visible(
        self,
        step: int,
        frame: bool | None = None,
        camera: bool | None = None,
        point_cloud: bool | None = None,
        mesh: bool | None = None,
        all: bool | None = None,
    ):
        with self._server.atomic():
            step_data = self._steps.get(f"step_{step}", None)
            if not step_data:
                raise KeyError(f"step {step} not exists")

            if all is not None:
                if "frame" in step_data:
                    step_data["frame"].visible = all
                if "camera" in step_data:
                    step_data["camera"].visible = all
                if "point_cloud" in step_data:
                    step_data["point_cloud"].visible = all
                if "mesh" in step_data:
                    step_data["mesh"].visible = all

            if frame is not None and "frame" in step_data:
                step_data["frame"].visible = frame
            if camera is not None and "camera" in step_data:
                step_data["camera"].visible = camera
            if point_cloud is not None and "point_cloud" in step_data:
                step_data["point_cloud"].visible = point_cloud
            if mesh is not None and "mesh" in step_data:
                step_data["mesh"].visible = mesh

    def remove_step(self, step: int):
        step_data = self._steps[f"step_{step}"]
        for v in step_data.values():
            v.remove()

    def reset_scene(self):
        self._server.reset_scene()
        self._steps = defaultdict(lambda: {})
        self._info = {}

    def add_info(self, label: str, value: Any):
        self._info[label] = value

    def get_info(self, label: str) -> Any:
        return self._info[label]

    def add_frame(self, step: int, pose: np.ndarray | None = None, **kwargs):
        frame = self._server.add_frame(
            f"step_{step}_frame", **self._convert_pose(pose), **kwargs
        )
        self._steps[f"step_{step}"]["frame"] = frame

    def add_camera(
        self,
        step: int,
        fov: float,
        image: np.ndarray,
        scale: float = 0.3,
        color: tuple[int, int, int] = (20, 20, 20),
        pose: np.ndarray | None = None,
        visible: bool = True,
    ):
        camera = self._server.add_camera_frustum(
            f"step_{step}_camera",
            fov=np.deg2rad(fov),
            image=image,
            aspect=image.shape[1] / image.shape[0],
            color=color,
            scale=scale,
            visible=visible,
            **self._convert_pose(pose),
        )
        self._steps[f"step_{step}"]["camera"] = camera

    def add_depth_point_cloud(
        self,
        step: int,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        color: np.ndarray | None = None,
        pose: np.ndarray | None = None,
        point_size: float = 0.01,
        visible: bool = True,
    ):
        pcd = (
            depth_to_3d(
                torch.from_numpy(depth)[None, None],
                torch.from_numpy(intrinsics)[None],
            )[0]
            .permute(1, 2, 0)
            .numpy()
        )
        pcd = pcd[depth > 0]
        if color is not None:
            color = color[depth > 0]
        handle = self._server.add_point_cloud(
            f"step_{step}_point_cloud",
            points=pcd,
            colors=color,
            point_size=point_size,
            **self._convert_pose(pose),
            visible=visible,
        )
        self._steps[f"step_{step}"]["point_cloud"] = handle

    def add_mesh(
        self,
        step: int,
        vertices: np.ndarray,
        triangles: np.ndarray,
        vertex_colors: np.ndarray | None = None,
        pose: np.ndarray | None = None,
        visible: bool = True,
    ):
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=triangles, vertex_colors=vertex_colors
        )
        handle = self._server.add_mesh_trimesh(
            f"step_{step}_mesh", mesh, **self._convert_pose(pose), visible=visible
        )
        self._steps[f"step_{step}"]["mesh"] = handle

    # GUI handles

    @property
    def gui(self) -> viser.ViserServer:
        return self._gui_funcs

    @contextmanager
    def atomic(self):
        with self._server.atomic():
            yield

    # wait for gui

    def wait(self):
        try:
            while True:
                sleep(10.0)
        except KeyboardInterrupt:
            self._server.stop()


class _GuiFuncs:
    def __init__(self, server):
        self._server = server

    def __getattr__(self, name: str) -> Any:
        if name.startswith("add_gui") and hasattr(self._server, name):
            return getattr(self._server, name)
