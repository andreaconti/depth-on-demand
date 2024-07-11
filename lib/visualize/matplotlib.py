"""
Visualization utilities related to matplotlib
"""

from torch import Tensor

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

from skimage.transform import resize as img_resize
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

try:
    from IPython import get_ipython as _get_ipython

    _is_interactive = lambda: _get_ipython() is not None
except ImportError:
    _is_interactive = lambda: None


__all__ = ["GridFigure", "make_gif", "color_depth"]


class GridFigure:
    """
    Utility class to plot a grid of images, really useful
    in training code to log qualitatives
    """

    figure: Figure
    rows: int
    cols: int

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        size: tuple[int, int] | None = None,
        tight_layout: bool = True,
    ):
        figsize = None
        if size:
            h, w = size
            px = 1 / plt.rcParams["figure.dpi"]
            figsize = (cols * w * px, rows * h * px)
        self.figure = plt.figure(tight_layout=tight_layout, figsize=figsize)
        self.rows = rows
        self.cols = cols

    def imshow(
        self,
        pos: int,
        image: Tensor,
        /,
        *,
        show_axis: bool = False,
        norm: bool = True,
        **kwargs,
    ):
        ax = self.figure.add_subplot(self.rows, self.cols, pos)
        if not show_axis:
            ax.set_axis_off()
        ax.imshow(_to_numpy(image, norm=norm), **kwargs)

    def show(self):
        self.figure.show()

    def close(self):
        plt.close(self.figure)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, ext_tb):
        self.show()
        if not _is_interactive():
            self.close()


def make_gif(
    imgs: Iterable[Union[np.ndarray, Figure]],
    path: Union[Path, str],
    axis: bool = False,
    close_figures: bool = True,
    bbox_inches: Optional[str] = "tight",
    pad_inches: float = 0,
    **kwargs,
):
    """
    Renders each image using matplotlib and the compose them into a GIF saved
    in path, kwargs are arguments for ``plt.imshow``
    """

    tmpdir = Path(tempfile.mkdtemp())
    for i, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            plt.axis({False: "off", True: "on"}[axis])
            plt.imshow(img, **kwargs)
            plt.savefig(
                tmpdir / f"{i}.png", bbox_inches=bbox_inches, pad_inches=pad_inches
            )
            plt.close()
        elif isinstance(img, Figure):
            img.savefig(
                tmpdir / f"{i}.png", bbox_inches=bbox_inches, pad_inches=pad_inches
            )
            if close_figures:
                plt.close(img)

    with imageio.get_writer(path, mode="I") as writer:
        paths = sorted(tmpdir.glob("*.png"), key=lambda x: int(x.name.split(".")[0]))
        for path in paths:
            image = imageio.imread(path)
            writer.append_data(image)
    shutil.rmtree(tmpdir, ignore_errors=True)


def color_depth(depth: np.ndarray, cmap="magma_r", **kwargs):
    px = 1 / plt.rcParams["figure.dpi"]
    h, w = depth.shape[:2]
    fig = plt.figure(figsize=(w * px, h * px))
    plt.axis("off")
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.imshow(depth, cmap=cmap, **kwargs)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = img_resize(data, depth.shape)
    alpha = (depth[..., None] > 0).astype(np.uint8) * 255
    data = np.concatenate([data, alpha], -1)
    plt.close(fig)
    return data

# internal utils


def _to_numpy(image: Tensor | np.ndarray, norm: bool = True):
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy().astype(np.float32)
        if image.ndim == 3:
            image = image.transpose([1, 2, 0])
        elif image.ndim != 2:
            raise ValueError(f"Unsupported torch.Tensor of shape {image.shape}")
    if norm:
        nan_mask = np.isnan(image)
        if not np.all(nan_mask):
            img_min, img_max = image[~nan_mask].min(), image[~nan_mask].max()
            image = (image - img_min) / (img_max - img_min)
    return image
