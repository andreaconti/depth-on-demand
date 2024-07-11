"""
Visualizer 3D of a scene built with my network
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
import argparse
import warnings

from callbacks.build_scene import BuildScene
from callbacks.run_demo import RunDemo

from lib.visualize.gui import Visualize3D

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--renders-folder", default="./renders")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    viz = Visualize3D(host=args.host, port=args.port)
    BuildScene(viz, args.device)
    RunDemo(viz, args.renders_folder)
    viz.wait()


if __name__ == "__main__":
    main()
