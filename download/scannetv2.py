from _utils import github_download_unzip_assets
from argparse import ArgumentParser
from zipfile import ZipFile
from pathlib import Path
import os
import gdown


def download_and_unzip_meshes(root: str | Path):
    root = Path(root)

    # from TransformerFusion
    # https://github.com/AljazBozic/TransformerFusion?tab=readme-ov-file
    url = "https://drive.usercontent.google.com/download?id=1-nto65_JTNs1vyeHycebidYFyQvE6kt4&authuser=0&confirm=t&uuid=05aba470-c11c-48f9-8ba3-162560ab15bb&at=APZUnTUsLTVanQWS8dVSGr5pA3hJ%3A1710523361429"
    out_zip = str(Path(root) / ".meshes.zip")
    gdown.download(url, out_zip, quiet=False)

    # unzip data in the correct format
    with ZipFile(out_zip, "r") as f:
        for file in f.infolist():
            if not file.is_dir():
                _, scene, fname = file.filename.split("/")
                new_path = root / (
                    f"{scene}_vh_clean.ply"
                    if fname == "mesh_gt.ply"
                    else f"{scene}_{fname}"
                )
                data = f.read(file)
                with open(new_path, "wb") as out:
                    out.write(data)

    # remove the zip
    os.remove(out_zip)


if __name__ == "__main__":
    parser = ArgumentParser("download the TartanAir test set for evaluation")
    parser.add_argument("--root", type=Path, default=Path("data/scannetv2"))
    args = parser.parse_args()
    github_download_unzip_assets(
        "andreaconti",
        "depth-on-demand",
        ["156761793", "156761795", "156761794"],
        args.root,
    )
    download_and_unzip_meshes(args.root)
