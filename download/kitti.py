"""
Script to download the KITTI Depth Completion
validation set
"""

import os
import re
import shutil
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile
import requests
from argparse import ArgumentParser


calib_scans = [
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_29_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_28_calib.zip",
]

scenes = list(
    map(
        lambda x: "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
        + f"{x}/{x}_sync.zip",
        [
            "2011_09_26_drive_0020",
            "2011_09_26_drive_0036",
            "2011_09_26_drive_0002",
            "2011_09_26_drive_0013",
            "2011_09_26_drive_0005",
            "2011_09_26_drive_0113",
            "2011_09_26_drive_0023",
            "2011_09_26_drive_0079",
            "2011_09_29_drive_0026",
            "2011_09_30_drive_0016",
            "2011_10_03_drive_0047",
            "2011_09_26_drive_0095",
            "2011_09_28_drive_0037",
        ],
    )
)


def download_file(url, save_path, chunk_size=1024, verbose=True):
    """
    Downloads a zip file from an `url` into a zip file in the
    provided `save_path`.
    """
    r = requests.get(url, stream=True)
    zip_name = url.split("/")[-1]

    content_length = int(r.headers["Content-Length"]) / 10**6

    if verbose:
        bar = tqdm(total=content_length, unit="Mb", desc="download " + zip_name)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
            if verbose:
                bar.update(chunk_size / 10**6)

    if verbose:
        bar.close()


def raw_download(root_path: str):

    date_match = re.compile("[0-9]+_[0-9]+_[0-9]+")
    drive_match = re.compile("[0-9]+_[0-9]+_[0-9]+_drive_[0-9]+_sync")

    def download_unzip(url):
        date = date_match.findall(url)[0]
        drive = drive_match.findall(url)[0]
        os.makedirs(os.path.join(root_path, date), exist_ok=True)
        download_file(url, os.path.join(root_path, date, drive + ".zip"), verbose=False)
        with ZipFile(os.path.join(root_path, date, drive + ".zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(root_path, date, drive + "_tmp"))
        os.rename(
            os.path.join(root_path, date, drive + "_tmp", date, drive),
            os.path.join(root_path, date, drive),
        )
        shutil.rmtree(os.path.join(root_path, date, drive + "_tmp"))
        os.remove(os.path.join(root_path, date, drive + ".zip"))

    for scene in tqdm(scenes, desc="download kitti (raw)"):
        download_unzip(scene)


def dc_download(root_path: str, progress_bar=True):
    """
    Downloads and scaffold depth completion dataset in `root_path`
    """

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    # urls
    data_depth_selection_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip"
    )
    data_depth_velodyne_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_velodyne.zip"
    )
    data_depth_annotated_url = (
        "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    )

    # download of zips
    download_file(
        data_depth_selection_url,
        os.path.join(root_path, "data_depth_selection.zip"),
        verbose=progress_bar,
    )

    download_file(
        data_depth_velodyne_url,
        os.path.join(root_path, "data_depth_velodyne.zip"),
        verbose=progress_bar,
    )
    download_file(
        data_depth_annotated_url,
        os.path.join(root_path, "data_depth_annotated.zip"),
        verbose=progress_bar,
    )

    # unzip and remove zips
    with ZipFile(os.path.join(root_path, "data_depth_selection.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)
        os.rename(
            os.path.join(
                root_path, "depth_selection", "test_depth_completion_anonymous"
            ),
            os.path.join(root_path, "test_depth_completion_anonymous"),
        )
        os.rename(
            os.path.join(
                root_path, "depth_selection", "test_depth_prediction_anonymous"
            ),
            os.path.join(root_path, "test_depth_prediction_anonymous"),
        )
        os.rename(
            os.path.join(root_path, "depth_selection", "val_selection_cropped"),
            os.path.join(root_path, "val_selection_cropped"),
        )
        os.rmdir(os.path.join(root_path, "depth_selection"))
    with ZipFile(os.path.join(root_path, "data_depth_velodyne.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)
    with ZipFile(os.path.join(root_path, "data_depth_annotated.zip"), "r") as zip_ref:
        zip_ref.extractall(root_path)

    # remove zip files
    os.remove(os.path.join(root_path, "data_depth_selection.zip"))
    os.remove(os.path.join(root_path, "data_depth_annotated.zip"))
    os.remove(os.path.join(root_path, "data_depth_velodyne.zip"))


def calib_download(root_path: str):
    """
    Downloads and scaffolds calibration files
    """

    Path(root_path).mkdir(exist_ok=True)

    for repo in calib_scans:
        calib_zip_path = os.path.join(root_path, "calib.zip")
        download_file(repo, calib_zip_path)
        with open(calib_zip_path, "rb") as f:
            ZipFile(f).extractall(root_path)
        os.remove(calib_zip_path)


if __name__ == "__main__":
    parser = ArgumentParser("download the KITTI test set for evaluation")
    parser.add_argument("--root", type=Path, default=Path("data/kitti"))
    args = parser.parse_args()
    raw_download(str(args.root / "raw"))
    calib_download(str(args.root / "raw"))
    dc_download(str(args.root / "depth_completion"))
