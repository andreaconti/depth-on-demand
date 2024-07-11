import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
import viser

from lib.visualize.gui import Visualize3D


class RunDemo:
    def __init__(self, viz: Visualize3D, renders_folder: Path):
        self.viz = viz
        self.renders_folder = Path(renders_folder)

        # create gui
        with viz.gui.add_gui_folder("Demo"):
            self.speed = viz.gui.add_gui_slider("fps", 1, 20, 1, 5)
            self.overlap_time = viz.gui.add_gui_slider(
                "frames overlap", 0, 1, 0.01, 0.05
            )
            self.keep_mesh = viz.gui.add_gui_checkbox("Keep Background Mesh", False)
            self.run_button = viz.gui.add_gui_button("Run Demo")
            self.height = viz.gui.add_gui_number("Render Height", 960, 100, 2160, 1)
            self.width = viz.gui.add_gui_number("Render Width", 1280, 100, 2160, 1)
            self.render_button = viz.gui.add_gui_button("Render Video")
            self.save_frames = viz.gui.add_gui_checkbox("Save Frames", False)
        self.run_button.on_click(self.run_demo_clbk)
        self.render_button.on_click(self.render_video_clbk)

    def render_video_clbk(self, event: viser.GuiEvent):
        client = event.client
        assert client is not None
        client.camera.get_render
        self.run_demo_clbk(None, client)

    def run_demo_clbk(self, event, render_client=None):
        keep_mesh = self.keep_mesh.value
        keys = ["point_cloud", "frame", "camera", "mesh"]
        disable_all = {k: False for k in keys}
        enable_all = {k: True for k in keys}
        viz = self.viz

        # hide all
        for step in viz.steps:
            viz.step_visible(
                step,
                **(
                    disable_all | {"mesh": True}
                    if keep_mesh and step == viz.steps[-1]
                    else disable_all
                ),
            )

        # start animation
        images = []
        for step in viz.steps:
            viz.step_visible(
                step,
                **(
                    enable_all | {"mesh": False}
                    if keep_mesh and step != viz.steps[-1]
                    else enable_all
                ),
            )
            time.sleep(1 / self.speed.value)
            if step != viz.steps[-1]:
                viz.step_visible(step, **disable_all)

            if render_client is not None:
                images.append(
                    render_client.camera.get_render(
                        height=self.height.value, width=self.width.value
                    )
                )

        # save the animation
        if render_client is not None:
            scan_name = viz.get_info("scan")
            render_folder = self.renders_folder / viz.get_info("dataset") / scan_name
            render_folder.mkdir(exist_ok=True, parents=True)

            # write the video
            h, w, _ = images[0].shape
            writer = cv2.VideoWriter(
                str(render_folder / "video.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.speed.value,
                (w, h),
            )
            for frame in images:
                writer.write(frame[..., ::-1].astype(np.uint8))
            writer.release()

            # save the mesh
            mesh: trimesh.Trimesh = viz.get_info("mesh")
            mesh.vertices = mesh.vertices - mesh.centroid
            _ = mesh.export(
                str(render_folder / "mesh.glb"),
                file_type="glb",
            )

            # save frames
            if self.save_frames.value:
                for idx, dict_data in enumerate(self.viz.get_info("frames")):
                    with h5py.File(render_folder / f"data-{idx:0>6d}.h5", "w") as f:
                        f["target"] = dict_data["target"]
                        f["source"] = dict_data["source"]
                        f["depth"] = dict_data["depth"]
                        f["target_hints"] = dict_data["target_hints"]
                        f["source_hints"] = dict_data["source_hints"]
