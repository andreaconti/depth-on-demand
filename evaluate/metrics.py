"""
Script to reproduce the DoD Paper metrics on various datasets used
"""

import sys

# In[] Imports
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import shutil
import tempfile
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import pandas as pd
import torch
import utils.funcs as funcs
from depth_on_demand import Model as DepthOnDemand
from omegaconf import OmegaConf
from torchmetrics import MeanMetric
from tqdm import tqdm
from utils.metrics import compute_metrics_depth

from lib.benchmark.mesh import TSDFFusion, mesh_metrics
from lib.dataset import load_datamodule

warnings.filterwarnings("ignore", category=UserWarning)

rootdir = Path(__file__).parents[1]
thisdir = Path(__file__).parent

# In[] Args

parser = ArgumentParser("Test DoD Model")
parser.add_argument(
    "--dataset",
    choices=["kitti", "scannetv2", "sevenscenes", "tartanair"],
    default="sevenscenes",
)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--f")
args = parser.parse_args()

# In[] Load dataset
cfg = OmegaConf.load(thisdir / "config" / (args.dataset + ".yaml"))
dm = load_datamodule(
    args.dataset,
    **cfg.args,
    split_test_scans_loaders=True,
)
dm.prepare_data()
dm.setup("test")
scans = dm.test_dataloader()

# In[] Load the model
model = DepthOnDemand(
    pretrained={
        "sevenscenes": "scannetv2",
        "scannetv2": "scannetv2",
        "tartanair": "tartanair",
        "kitti": "kitti",
    }[args.dataset],
    device=args.device,
)

# In[] Testing
predict_pose = None
if cfg.inference.depth_hints.pnp_pose:

    from utils.pose_estimation import PoseEstimation

    predict_pose = PoseEstimation(dilate_depth=3)
    predict_pose.to(args.device)

metrics = defaultdict(lambda: MeanMetric().to(args.device))
for scan in tqdm(scans):

    buffer_hints = {}
    buffer_source = {}
    if args.dataset == "scannetv2":
        tsdf_volume = TSDFFusion(**cfg.tsdf_fusion)

    for idx, batch in enumerate(tqdm(scan, leave=False)):

        batch = {
            k: v.to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if idx % cfg.inference.depth_hints.interval == 0:
            buffer_hints = deepcopy(batch)
        if idx % cfg.inference.source.interval == 0:
            buffer_source = deepcopy(batch)

        input = funcs.prepare_input(
            batch
            | {k + "_prev_0": v for k, v in buffer_hints.items()}
            | {k + "_prev_1": v for k, v in buffer_source.items()},
            hints_density=cfg.inference.depth_hints.density,
            hints_from_pcd=cfg.inference.depth_hints.hints_from_pcd,
            hints_postfix="_prev_0",
            source_postfix="_prev_1",
            predict_pose=predict_pose,
        )
        buffer_source["hints"] = input.pop("source_hints")
        buffer_hints["hints"] = input.pop("hints_hints")

        # inference
        gt = batch["depth"].to(args.device)
        mask = gt > 0
        with torch.no_grad():
            pred_depth = model(
                **input,
                n_cycles=cfg.inference.model_eval_params.n_cycles,
            )

        # metrics
        for k, v in compute_metrics_depth(pred_depth[mask], gt[mask], "test").items():
            metrics[k].update(v)
        if args.dataset == "scannetv2":
            tsdf_volume.integrate_rgbd(
                batch["image"][0].permute(1, 2, 0).cpu().numpy(),
                pred_depth[0].permute(1, 2, 0).detach().cpu().numpy(),
                batch["intrinsics"][0].cpu().numpy(),
                batch["extrinsics"][0].cpu().numpy(),
            )

    if args.dataset == "scannetv2":
        mesh_pred = tsdf_volume.triangle_mesh()
        for k, v in mesh_metrics(
            mesh_pred,
            batch["gt_mesh_path"][0],
            batch["gt_mesh_world2grid_path"][0],
            batch["gt_mesh_occl_path"][0],
            device=args.device,
        ).items():
            metrics["test/" + k].update(v)
            tmpdir = Path(tempfile.mkdtemp())
            scan_name = "_".join(Path(batch["gt_mesh_path"][0]).stem.split("_")[:2])
            mesh_path = tmpdir / (scan_name + ".ply")
            tsdf_volume.write_triangle_mesh(mesh_path)
            shutil.rmtree(tmpdir, ignore_errors=True)

# %%

print(f"== metrics for {args.dataset} ==")
for k, v in metrics.items():
    print(f"{k.ljust(25)}: {v.compute():0.5f}")
