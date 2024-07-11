"""
Visualization tools implemented in plotly
"""

import pandas as pd

try
    from plotly.graph_objects import Scatter3d, Figure  # type: ignore
except ImportError:
    raise ImportError("To use plotly visualize utilities install plotly")


def point_cloud(
    data: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color: list[str] | str | None = None,
    point_size: float = 1.0,
) -> Scatter3d:
    """
    Utility function to visualize a Point Cloud with colors
    """

    marker_opts = {
        "size": point_size,
    }

    if color is not None:
        if isinstance(color, str):
            marker_opts["color"] = [color] * len(data[x])
        else:
            marker_opts["color"] = [
                f"rgb({r}, {g}, {b})"
                for r, g, b in zip(data[color[0]], data[color[1]], data[color[2]])
            ]

    scatter = Scatter3d(
        x=data[x], y=data[y], z=data[z], mode="markers", marker=marker_opts
    )
    return scatter
