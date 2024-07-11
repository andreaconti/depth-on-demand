"""
Common interface to load the datasets, simply import the ``load_datamodule`` function,
provide the name of the dataset to load and optionally the specific arguments for that
dataset
"""

import torch
from .scannetv2 import ScanNetV2DataModule
from .sevenscenes import SevenScenesDataModule
from .tartanair import TartanairDataModule
from .kitti import KittiDataModule
from ._utils import list_scans
from lightning import LightningDataModule
import torchvision.transforms.functional as TF

__all__ = ["load_datamodule", "list_scans"]


def load_datamodule(name: str, /, **kwargs) -> LightningDataModule:
    match name:
        case "scannetv2":
            DataModule = ScanNetV2DataModule
        case "sevenscenes":
            DataModule = SevenScenesDataModule
        case "tartanair":
            DataModule = TartanairDataModule
        case "kitti":
            DataModule = KittiDataModule
        case other:
            raise ValueError(f"dataset {other} not available")

    return DataModule(**kwargs, eval_transform=_Preprocess(name))


class _Preprocess:
    def __init__(self, dataset: str):
        self.dataset = dataset

    def __call__(self, sample: dict) -> dict:
        for k in sample:
            if "path" not in k:
                if "intrinsics" not in k and "extrinsics" not in k and "pcd" not in k:
                    sample[k] = TF.to_tensor(sample[k])
                else:
                    sample[k] = torch.from_numpy(sample[k])
            if self.dataset == "scannetv2":
                if "image" in k or "depth" in k:
                    sample[k] = TF.center_crop(sample[k], [464, 624])
                if "intrinsics" in k:
                    intrins = sample[k].clone()
                    intrins[0, -1] -= 8
                    intrins[1, -1] -= 8
                    sample[k] = intrins
            if "image" in k:
                sample[k] = TF.normalize(
                    sample[k], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
        return sample
