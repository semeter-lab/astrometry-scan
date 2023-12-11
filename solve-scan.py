#!/usr/bin/env python3
"""
Try to solve an arbitrary citizen science sky field using arbitrary thresholds.
Goal is to be able to process about 1000 images in 24 hours on a modern laptop computer.
By "process" we mean either solve an images plate scale,
or determine that it is not solvable with this technique.
"""

from pathlib import Path
import subprocess
import shutil
import logging
import argparse
import itertools
import tempfile

import imageio.v3 as iio

from matplotlib.pyplot import figure, draw, pause
import matplotlib.patches as patches
import matplotlib.colors as mplcolors

import astrometry_azel as ael

OPTS = ["--downsample", "4", "--scale-low", "10", "--verbose"]
"""
from practical experience, we have seen that solve-field works at least down to 256x256 pixel images
"""
X_MIN = 256
Y_MIN = 256


def subimage_dims(x: int, y: int) -> tuple[int, int]:
    """
    Determine dimensions of subimage
    Something like 1/3 of the image.
    Minimum size in a dimension is 256 pixels.
    Too small an image, and solve-field has too little info to solve.

    Parameters
    ----------

    x: int
        Image width (pixels)
    y: int
        Image height (pixels)

    Returns
    -------
    sx: int
        Subimage width (pixels)
    sy: int
        Subimage height (pixels)
    """

    return max(X_MIN, x // 3), max(Y_MIN, y // 3)


def subimage_centers(
    x: int, y: int, sx: int, sy: int, Nx: int, Ny: int, x_margin: int, y_margin: int
) -> list[tuple[int, int]]:
    """
    Parameters
    ----------

    x: int
        Image width (pixels)
    y: int
        Image height (pixels)
    sx: int
        Subimage width (pixels)
    sy: int
        Subimage height (pixels)
    Nx: int
        Number of subimages in x direction
    Ny: int
        Number of subimages in y direction
    x_margin: int
        x-pixels to stay away from edge of image
    y_margin: int
        y-pixels to stay away from edge of image
    """

    centers = []

    good_width = x - 2 * x_margin
    good_height = y - 2 * y_margin
    for i in range(Nx):
        center_x = x_margin + good_width * (i + 1) // (Nx + 1)
        for j in range(Ny):
            center_y = y_margin + good_height * (j + 1) // (Ny + 1)

            centers.append((center_x, center_y))

    return centers


def process_image(file: Path, Nx: int, Ny: int, x_margin: int, y_margin: int) -> None:
    img = iio.imread(file)
    y, x = img.shape[:2]
    if x < X_MIN or y < Y_MIN:
        logging.error(
            f"{file} size {x} x {y} too small:  Minimium image size {X_MIN} x {Y_MIN}"
        )
        return

    print(f"Processing {file} {x}x{y} pixels")

    sx, sy = subimage_dims(x, y)

    centers = subimage_centers(x, y, sx, sy, Nx, Ny, x_margin, y_margin)

    plot_subimage_box(img, centers, sx, sy)

    for center in centers:
        subimg = img[
            center[1] - sy // 2 : center[1] + sy // 2,
            center[0] - sx // 2 : center[0] + sx // 2,
        ]

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as f:
            subimage_file = Path(f) / file.name
            iio.imwrite(subimage_file, subimg)

            if ael.plate_scale(subimage_file, args=OPTS):
                new_dir = file.parent / f"{file.stem}-{center[0]}x{center[1]}/"
                print(f"{file}  Solved subimage center {center} copy to {new_dir}")
                shutil.copytree(f, new_dir)
                return

    logging.error(f"{file}  Failed to solve subimages")


def plot_subimage_box(img, centers: list[tuple[int, int]], sx: int, sy: int) -> None:
    """
    Show subimage overlay on original image
    """
    fig = figure()
    ax = fig.gca()
    ax.imshow(img)

    color_cycle = itertools.cycle(mplcolors.TABLEAU_COLORS)

    for (cx, cy), ec in zip(centers, color_cycle):
        ax.plot(cx, cy, color=ec, marker="+")
        rect = patches.Rectangle(
            (cx - sx // 2, cy - sy // 2),
            width=sx,
            height=sy,
            linewidth=1,
            edgecolor=ec,
            facecolor="none",
        )
        ax.add_patch(rect)

    draw()
    pause(0.05)


p = argparse.ArgumentParser(description="Process image(s) automatically")
p.add_argument("image_files", nargs="+", help="Image file(s) to process")
p.add_argument(
    "Nx",
    help="number of subimages in x and y directions",
    type=int,
    nargs="?",
    default=2,
)
p.add_argument(
    "Ny",
    help="number of subimages in x and y directions",
    type=int,
    nargs="?",
    default=2,
)
p.add_argument(
    "x_margin",
    help="number of pixels to stay away from x-edges",
    type=int,
    nargs="?",
    default=100,
)
p.add_argument(
    "y_margin",
    help="number of pixels to stay away from y-edges",
    type=int,
    nargs="?",
    default=100,
)
P = p.parse_args()

for file in P.image_files:
    file = Path(file).expanduser().resolve()
    if not file.is_file():
        logging.error(f"File not found {file}")
        continue

    process_image(file, P.Nx, P.Ny, x_margin=P.x_margin, y_margin=P.y_margin)
