from pathlib import Path
from typing import Literal, Callable, Type
import torchdata.datapipes as dp
from torch import Tensor
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from collections import defaultdict
import torchdata.datapipes as dp
import numpy as np

__all__ = ["GenericDataModule", "LoadDataset"]


class GenericDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        # dataset specific
        keyframes: str = "standard",
        load_prevs: int = 7,
        cycle: bool = True,
        filter_scans: list[str] | None = None,
        val_on_test: bool = False,
        split_test_scans_loaders: bool = False,
        # dataloader specific
        batch_size: int = 1,
        shuffle: bool = True,
        shuffle_keyframes: bool = False,
        order_sources_by_pose: bool = False,
        num_workers: int = 8,
        pin_memory: bool = True,
        load_dataset_cls: Type["LoadDataset"] | None = None,
        eval_transform: Callable[[dict], dict] | None = None,
        **sample_params,
    ):
        super().__init__()

        if load_prevs < 0:
            raise ValueError(f"0 <= load_prevs, not {load_prevs}")

        self.dataset = dataset
        self.keyframes = keyframes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_keyframes = shuffle_keyframes
        self.num_workers = num_workers
        self.eval_transform = eval_transform
        self.cycle = cycle
        self.filter_scans = filter_scans
        self.load_prevs = load_prevs
        self.val_on_test = val_on_test
        self.split_test_scans_dataloaders = split_test_scans_loaders
        self.order_sources_by_pose = order_sources_by_pose
        self.pin_memory = pin_memory
        self._load_dataset_cls = load_dataset_cls
        self._sample_params = sample_params
        self._other_params = {}

    def _filter_traj(self, split) -> str | list[str]:
        if self.filter_scans is None:
            return split
        else:
            split_scans = list_scans(self.dataset, self.keyframes, split)
            return [traj for traj in self.filter_scans if traj in split_scans]

    def _dataloader(self, split, scan=None, num_workers=None):
        dl_builder = self._load_dataset_cls(
            self.dataset,
            keyframes=self.keyframes,
            load_prevs=self.load_prevs,
            cycle=False,
            batch_size=1,
            shuffle=False,
            shuffle_keyframes=False,
            order_sources_by_pose=self.order_sources_by_pose,
            transform=self.eval_transform,
            num_workers=self.num_workers if num_workers is None else num_workers,
            pin_memory=self.pin_memory,
            **self._sample_params,
        )
        return dl_builder.build_dataloader(
            self._filter_traj(split) if scan is None else scan,
        )

    def setup(self, stage: str | None = None):
        if stage not in ["test", None]:
            raise ValueError(f"stage {stage} invalid")

        if stage in ["test", None]:
            if not self.split_test_scans_dataloaders:
                self._test_dl = self._dataloader("test")
            else:
                keyframes = build_scan_frames_mapping(
                    self.dataset,
                    self.keyframes,
                    "test",
                    self.load_prevs,
                )
                if self.filter_scans:
                    keyframes = {
                        k: v for k, v in keyframes.items() if k in self.filter_scans
                    }
                self._test_dl = [
                    self._dataloader("test", {key: value}, num_workers=1)
                    for key, value in keyframes.items()
                ]

    def test_dataloader(self):
        return self._test_dl


class LoadDataset:
    """
    This function embodies the whole creation of a datalaoder, the unique method
    to be overloaded is `load_sample` which loads a single sample of a sequence
    """

    def __init__(
        self,
        dataset: str,
        # dataset specific
        keyframes: Literal["standard", "dense", "offline"] = "standard",
        load_prevs: int = 7,
        cycle: bool = True,
        # dataloader specific
        batch_size: int = 1,
        shuffle: bool = False,
        shuffle_keyframes: bool = False,
        order_sources_by_pose: bool = True,
        transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        **kwargs,
    ):
        self.dataset = dataset
        self.keyframes = keyframes
        self.load_prevs = load_prevs
        self.cycle = cycle
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_keyframes = shuffle_keyframes
        self.order_sources_by_pose = order_sources_by_pose
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sample_kwargs = kwargs

    def build_dataloader(
        self,
        split: Literal["train", "val", "test"] | str | list[str] | dict = "train",
    ):
        # checks
        if self.load_prevs < 0:
            raise ValueError(f"0 <= load_prevs, not {self.load_prevs}")

        # load the keyframe file
        if split in ["train", "val", "test"]:
            keyframes_path = (
                Path(__file__).parent
                / f"_resources/{self.dataset}/keyframes/{self.keyframes}/{split}_split.txt"
            )
            if not keyframes_path.exists():
                raise ValueError(
                    f"split {split} for keyframes {self.keyframes} not available"
                )
            with open(keyframes_path, "rt") as f:
                keyframes_dict = defaultdict(_empty_dict)
                for ln in f:
                    scene, keyframe, *src_frames = ln.split()
                    keyframes_dict[scene][keyframe] = src_frames[: self.load_prevs]
        elif isinstance(split, (str, list)):
            split = [split] if isinstance(split, str) else split
            all_lines = []
            for file_path in (
                Path(__file__).parent
                / f"_resources/{self.dataset}/keyframes/{self.keyframes}"
            ).glob("*_split.txt"):
                with open(file_path, "rt") as f:
                    all_lines.extend(f.readlines())
            keyframes_dict = defaultdict(_empty_dict)
            for scene in split:
                for ln in all_lines:
                    if ln.startswith(scene):
                        scene, keyframe, *src_frames = ln.split()
                        keyframes_dict[scene][keyframe] = src_frames[: self.load_prevs]
        elif isinstance(split, dict):
            keyframes_dict = split
        else:
            raise ValueError(f"split type not allowed")

        # loading and processing pipeline
        keyframes_list = []
        for scene in keyframes_dict:
            keyframes_list.extend(
                [
                    {"scan": scene, "keyframe": keyframe, "sources": sources}
                    for keyframe, sources in keyframes_dict[scene].items()
                ]
            )

        ds = dp.map.SequenceWrapper(keyframes_list)
        ds = ds.shuffle() if self.shuffle else ds
        if self.cycle:
            if self.shuffle:
                ds = ds.cycle()
            else:
                ds = dp.iter.IterableWrapper(ds).cycle()
        ds = ds.sharding_filter() if self.shuffle or self.cycle else ds
        ds = ds.map(self._load)
        if self.order_sources_by_pose:
            ds = ds.map(order_by_pose)
        ds = ds.map(self.transform) if self.transform else ds
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def load_sample(self, scan: str, idx: str, suffix: str = "") -> dict:
        raise NotImplementedError("please implement load sample")

    def _load(self, sample):
        out = {}

        # shuffle keyframes
        if self.shuffle_keyframes:
            all_frames = [sample["keyframe"]] + sample["sources"]
            keyframe, sources = all_frames[0], all_frames[1:]
            sample = {"scan": sample["scan"], "keyframe": keyframe, "sources": sources}

        # load data
        out = self.load_sample(sample["scan"], sample["keyframe"], **self.sample_kwargs)
        for idx, src in enumerate(sample["sources"]):
            out |= self.load_sample(
                sample["scan"], src, suffix=f"_prev_{idx}", **self.sample_kwargs
            )
        return out


####


def list_scans(dataset: str, keyframes: str, split: str | None = None):
    """
    List all the scans used in a specific split of a dataset for a specific
    keyframe setting.
    """
    if split is not None:
        keyframes_path = (
            Path(__file__).parent
            / f"_resources/{dataset}/keyframes/{keyframes}/{split}_split.txt"
        )
        scans = []
        if keyframes_path.exists():
            scans = [ln.split()[0] for ln in open(keyframes_path, "rt")]
    else:
        scans = []
        for keyframes_path in (
            Path(__file__).parent / f"_resources/{dataset}/keyframes/{keyframes}"
        ).glob("*_split.txt"):
            scans.extend([ln.split()[0] for ln in open(keyframes_path, "rt")])
    return list(np.unique(scans))


def _empty_dict():
    return {}


def build_scan_frames_mapping(
    dataset: str,
    keyframes: str,
    split: str,
    load_prevs: int | None = None,
) -> dict[Path, dict[str, list[str]]]:
    """
    Given a dataset, its root, specified keyframes and split it build a dictionary
    mapping each scan in such split to a dictionary indexed by the target view and
    containing a list of valid source views.
    """
    keyframes_path = (
        Path(__file__).parent
        / f"_resources/{dataset}/keyframes/{keyframes}/{split}_split.txt"
    )
    if not keyframes_path.exists():
        raise ValueError(f"split {split} for keyframes {keyframes} not available")
    with open(keyframes_path, "rt") as f:
        keyframes_dict = defaultdict(_empty_dict)
        for ln in f:
            scene, keyframe, *src_frames = ln.split()
            if load_prevs is not None:
                src_frames = src_frames[:load_prevs]
            keyframes_dict[scene][keyframe] = src_frames
    return keyframes_dict


def order_by_pose(ex):
    """
    Takes an example and orders source views according with their distance
    from the target view
    """
    srcs = sorted({int(k.split("prev_")[-1]) for k in ex.keys() if "prev_" in k})
    tgt_pose = ex["extrinsics"]
    distances = np.array(
        [
            _pose_distance(_inv_pose(tgt_pose) @ ex[f"extrinsics_prev_{src}"])[0]
            for src in srcs
        ]
    )
    order = np.argsort(distances)
    out = {k: v for k, v in ex.items() if "_prev_" not in k}
    for new_idx, prev_idx in enumerate(order):
        for k, v in ex.items():
            if f"prev_{prev_idx}" in k:
                out[k.replace(f"prev_{prev_idx}", f"prev_{new_idx}")] = v
    return out


def _inv_pose(pose: np.ndarray):
    inverted = np.eye(4)
    inverted[:3, :3] = pose[:3, :3].T
    inverted[:3, 3:] = -inverted[:3, :3] @ pose[:3, 3:]
    return inverted


def _pose_distance(pose: np.ndarray):
    rot = pose[:3, :3]
    trasl = pose[:3, 3]
    R_trace = rot.diagonal().sum()
    r_measure = np.sqrt(2 * (1 - np.minimum(np.ones_like(R_trace) * 3.0, R_trace) / 3))
    t_measure = np.linalg.norm(trasl)
    combined_measure = np.sqrt(t_measure**2 + r_measure**2)
    return combined_measure, r_measure, t_measure
