from _utils import github_download_unzip_assets
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser("download the TartanAir test set for evaluation")
    parser.add_argument("--root", type=Path, default=Path("data/tartanair"))
    args = parser.parse_args()
    github_download_unzip_assets(
        "andreaconti",
        "depth-on-demand",
        ["156760244", "156760245", "156760243", "156760242"],
        args.root,
    )
