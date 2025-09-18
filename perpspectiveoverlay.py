"""Utility functions to project an image on to a planar surface using a perspective transform."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

PointList = Iterable[Tuple[float, float]]


def project_texture(
    background: np.ndarray,
    texture: np.ndarray,
    dst_points: PointList,
) -> np.ndarray:
    """Project ``texture`` on to ``background`` inside the quad ``dst_points``.

    Parameters
    ----------
    background:
        Image representing the scene (e.g. a façade). The array is left unchanged.
    texture:
        Image that must be placed on the wall. The four corners of the image are
        assumed to correspond to the four points in ``dst_points`` in this exact
        order: top-left, top-right, bottom-right, bottom-left.
    dst_points:
        Sequence of four 2D points expressed in pixel coordinates on the
        background image.

    Returns
    -------
    ``numpy.ndarray``
        A new image where the texture is composited on the background with the
        proper perspective.
    """

    dst_tuple = tuple(dst_points)
    if len(dst_tuple) != 4:
        raise ValueError("dst_points must contain exactly four points")

    h, w = texture.shape[:2]

    src = np.array(
        [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float32
    )
    dst = np.array(dst_tuple, dtype=np.float32)

    homography = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(texture, homography, (background.shape[1], background.shape[0]))

    mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, homography, (background.shape[1], background.shape[0]))

    background_masked = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(warped_mask))
    return cv2.add(background_masked, warped)


def load_and_project(
    background_path: Path,
    texture_path: Path,
    dst_points: PointList,
) -> np.ndarray:
    """Convenience wrapper to load the images before projection."""
    background = cv2.imread(str(background_path))
    texture = cv2.imread(str(texture_path))
    if background is None:
        raise ValueError(f"Unable to load background image: {background_path}")
    if texture is None:
        raise ValueError(f"Unable to load texture image: {texture_path}")

    return project_texture(background, texture, dst_points)


if __name__ == "__main__":
    # Example usage with dummy points. Replace the coordinates with the pixel
    # locations of the wall corners in your image.
    result = load_and_project(
        Path("maison.jpg"),
        Path("poster.jpg"),
        dst_points=[
            (320, 180),   # top-left corner on the wall
            (620, 140),   # top-right corner on the wall
            (640, 380),   # bottom-right corner on the wall
            (300, 420),   # bottom-left corner on the wall
        ],
    )
    cv2.imwrite("maison_avec_poster.jpg", result)