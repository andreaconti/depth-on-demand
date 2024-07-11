import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import trimesh
from depth_on_demand import Model

from lib.benchmark.mesh import TSDFFusion
from lib.dataset import list_scans, load_datamodule
from lib.dataset.utils import inv_pose, project_depth, sparsify_depth
from lib.visualize.gui import Visualize3D


class BuildScene:
    def __init__(
        self,
        viz: Visualize3D,
        device: str = "cuda:0",
    ):
        self.viz = viz
        self.device = device
        self.to_clear = False

        # select the model
        with viz.gui.add_gui_folder("Model"):
            self.model_pretrained = viz.gui.add_gui_dropdown(
                "Model Pretrained", ["scannetv2"], initial_value="scannetv2"
            )
            self.load_model(self.model_pretrained.value)
            self.model_pretrained.on_update(self.load_model)

        with viz.gui.add_gui_folder("Dataset"):
            self.dataset = viz.gui.add_gui_dropdown(
                "Name", ["sevenscenes", "scannetv2"], initial_value="scannetv2"
            )
            self.dataset_root = viz.gui.add_gui_text(
                "Root", initial_value="data/scannetv2"
            )
            self.dataset.on_update(self.on_update_dataset)

        # create build scene panel
        with viz.gui.add_gui_folder("Build Scene"):
            self.scene_select = viz.gui.add_gui_dropdown(
                "Scene", scenes[self.dataset.value]
            )
            self.hints_interval = viz.gui.add_gui_slider(
                "Hints Interval", min=1, max=10, step=1, initial_value=5
            )
            self.hints_density = viz.gui.add_gui_slider(
                "Hints Density", min=10, max=2000, step=10, initial_value=500
            )
            self.pcd_points_size = viz.gui.add_gui_slider(
                "Points Size", min=0.01, max=1.0, step=0.01, initial_value=0.05
            )
            self.frustum_color = viz.gui.add_gui_dropdown(
                "Frustum Color", list(matplotlib.colormaps.keys()), initial_value="jet"
            )
            self.frustum_color_range = viz.gui.add_gui_vector2(
                "Color Map Range", (0.25, 0.5), (0.0, 0.0), (1.0, 1.0), 0.05
            )
            self.overlap_time = viz.gui.add_gui_slider(
                "frames overlap", 0, 1, 0.01, 0.25
            )
            self.build_scene = viz.gui.add_gui_button("Build Scene")

        with (folder := viz.gui.add_gui_folder("TSDF Parameters")):
            self.tsdf_folder = folder
            self.voxel_length = viz.gui.add_gui_slider(
                "Voxel Length", 0.01, 0.5, 0.01, 0.04
            )
            self.sdf_trunc = viz.gui.add_gui_slider(
                "SDF Trunc", 0.01, 0.5, 0.01, 3 * 0.04
            )
            self.depth_trunc = viz.gui.add_gui_slider("Depth Trunc", 0.5, 10, 0.1, 3.0)
            self.integrate_only_slow = viz.gui.add_gui_checkbox(
                "integrate only when hints", False
            )

        self.build_scene.on_click(self.build_scene_callback)

    def on_update_dataset(self, *args, **kwargs):
        self.scene_select.options = scenes[self.dataset.value]
        self.voxel_length.visible = self.dataset.value != "kitti"
        self.sdf_trunc.visible = self.dataset.value != "kitti"
        self.depth_trunc.visible = self.dataset.value != "kitti"
        self.integrate_only_slow.visible = self.dataset.value != "kitti"
        self.dataset_root.value = f"data/{self.dataset.value}"

    def load_model(self, *args, **kwargs):
        self.model = Model(self.model_pretrained.value, device=self.device)

    def build_scene_callback(self, event):
        viz = self.viz
        scene_data = load_data(
            self.dataset.value, self.scene_select.value, root=self.dataset_root.value
        )
        interval = self.hints_interval.value
        density = self.hints_density.value

        colors = matplotlib.colormaps[self.frustum_color.value](
            np.linspace(*self.frustum_color_range.value, len(scene_data))
        )
        colors = (colors[:, :3] * 255).astype(np.uint8)

        tsdf = TSDFFusion(
            voxel_length=self.voxel_length.value,
            sdf_trunc=self.sdf_trunc.value,
            depth_trunc=self.depth_trunc.value,
            device=self.device,
        )
        viz.reset_scene()
        viz.add_info("dataset", self.dataset.value)
        viz.add_info("scan", self.scene_select.value)
        viz.add_info("frames", [])

        for step, ex in enumerate(scene_data):
            ex = {
                k: v.to(self.device)
                for k, v in ex.items()
                if isinstance(v, torch.Tensor)
            }
            if step % interval == 0:
                buffer = ex.copy()
                buffer_hints = sparsify_depth(buffer["depth"], density).to(self.device)
            with torch.no_grad():
                rel_pose = inv_pose(ex["extrinsics"]) @ buffer["extrinsics"]
                hints = project_depth(
                    buffer_hints,
                    buffer["intrinsics"],
                    ex["intrinsics"],
                    rel_pose,
                    torch.zeros_like(ex["image"][:, :1]),
                )
                depth = self.model(
                    ex["image"],
                    buffer["image"],
                    hints,
                    rel_pose,
                    torch.stack([ex["intrinsics"], buffer["intrinsics"]], 1),
                )

            img = img_to_rgb(ex["image"])
            depth = depth.cpu().numpy()[0, 0, ..., None]
            self.viz.get_info("frames").append(
                {
                    "target": img,
                    "source": img_to_rgb(buffer["image"]),
                    "target_hints": hints.cpu().numpy()[0, 0, ..., None],
                    "source_hints": buffer_hints.cpu().numpy()[0, 0, ..., None],
                    "depth": depth,
                }
            )
            pose = ex["extrinsics"][0].cpu().numpy()
            intrinsics = ex["intrinsics"][0].cpu().numpy()
            hints_cpu = buffer_hints[0, 0].cpu().numpy()
            hints_cpu = np.where(hints_cpu > self.depth_trunc.value, 0.0, hints_cpu)

            # viz.add_step(step)
            # viz.step_visible(step, container=False)
            if step % interval == 0:
                camera_color = (255, 0, 0)
            else:
                camera_color = tuple(colors[step])
            viz.add_camera(
                step, 60.0, img, color=camera_color, pose=pose, visible=False
            )
            if step % interval == 0:
                red = np.zeros_like(img)
                red[..., 0] = 255
                viz.add_depth_point_cloud(
                    step,
                    hints_cpu,
                    intrinsics,
                    red,
                    point_size=self.pcd_points_size.value,
                    pose=pose,
                    visible=False,
                )

            if self.integrate_only_slow.value and step % interval != 0:
                pass
            else:
                tsdf.integrate_rgbd(img, depth, intrinsics, pose)
            mesh = tsdf.triangle_mesh()
            viz.add_info(
                "mesh",
                trimesh.Trimesh(
                    vertices=np.asarray(mesh.vertices),
                    faces=np.asarray(mesh.triangles),
                    vertex_colors=np.asarray(mesh.vertex_colors),
                ),
            )
            viz.add_mesh(
                step,
                vertices=np.asarray(mesh.vertices),
                triangles=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors),
                # pose=np.linalg.inv(pose),
                visible=False,
            )

            if step > 0:
                viz.step_visible(step - 1, all=False, mesh=True)
            viz.step_visible(step, camera=True, point_cloud=True, mesh=True)
            if step > 0:
                time.sleep(self.overlap_time.value)
                viz.step_visible(step - 1, all=False)


def load_data(dataset: str, scene: str, root: Path | str):
    root = {"root": root}
    if dataset == "kitti":
        root = {
            "root_raw": Path(root) / "raw",
            "root_completion": Path(root) / "depth_completion",
        }

    dm = load_datamodule(
        dataset,
        load_prevs=0,
        keyframes="standard",
        **root,
        split_test_scans_loaders=True,
        filter_scans=[scene],
    )
    dm.prepare_data()
    dm.setup("test")
    return next(iter(dm.test_dataloader()))


scenes = {
    "scannetv2": list_scans("scannetv2", "standard", "test"),
    "sevenscenes": list_scans("sevenscenes", "standard", "test"),
    "kitti": list_scans("kitti", "standard", "test"),
    "tartanair": list_scans("tartanair", "standard", "test"),
}


def img_to_rgb(image: torch.Tensor):
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    return image
