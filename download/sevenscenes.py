from _utils import github_download_unzip_assets
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser("download the 7Scenes test set for evaluation")
    parser.add_argument("--root", type=Path, default=Path("data/sevenscenes"))
    args = parser.parse_args()
    github_download_unzip_assets(
        "andreaconti",
        "depth-on-demand",
        ["155673151"],
        args.root,
    )
