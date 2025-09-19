"""Utility to estimate the visible quadrilateral of a wall mask.

The module exposes :func:`estimate_wall_quadrilateral` which implements the
strategy discussed with the user:

1. "Nettoyage" du mur grâce à l'enveloppe convexe (suppression des portes,
   fenêtres, etc.).
2. Recherche des directions dominantes du mur via Hough sur les contours.
3. Calcul des droites de support parallèles à ces directions pour reconstituer
   les bords manquants.
4. Intersection de ces droites pour former le quadrilatère.

En cas d'échec de la détection des directions (photo très bruitée ou trop peu
de segments), on retombe sur un simple rectangle orienté (`cv2.minAreaRect`).

Le code ne dépend que de NumPy et (optionnellement) d'OpenCV. Si OpenCV n'est
pas installé, on lève une erreur explicite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - we only need to fail gracefully when OpenCV is absent
    import cv2
except ImportError as exc:  # pragma: no cover - explicitly inform the caller
    raise ImportError(
        "OpenCV (cv2) est requis pour estimate_wall_quadrilateral."
    ) from exc


@dataclass
class LineCluster:
    """Small helper storing Hough lines that share roughly the same direction."""

    angles: List[float]
    lines: List[np.ndarray]

    @property
    def direction_angle(self) -> float:
        """Return the averaged direction (in radians, modulo π).

        We average on the doubled angle to make θ and θ+π equivalent.
        """

        if not self.angles:
            raise ValueError("Cluster without angles")

        doubled = np.array(self.angles) * 2.0
        mean_cos = float(np.mean(np.cos(doubled)))
        mean_sin = float(np.mean(np.sin(doubled)))
        return 0.5 * math.atan2(mean_sin, mean_cos)


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest contour of the binary mask."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Aucun contour détecté dans le masque fourni.")

    largest = max(contours, key=cv2.contourArea)
    return largest


def _convex_hull(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    return hull.reshape(-1, 2)


def _hough_lines(image: np.ndarray, mask: np.ndarray) -> List[np.ndarray]:
    """Detect dominant line segments in the region of interest.

    The Hough transform works best on edges; we therefore run it on a mask
    combining the supplied image (if any) and the convex hull mask.
    """

    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    h, w = edges.shape
    min_len = max(30, int(0.25 * min(h, w)))
    max_gap = max(5, int(0.02 * max(h, w)))
    threshold = max(30, int(0.15 * max(h, w)))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=max(20, min_len),
        maxLineGap=max(5, max_gap),
    )

    if lines is None:
        return []
    return [line[0] for line in lines]


def _angle_from_line(line: Sequence[int]) -> float:
    x1, y1, x2, y2 = line
    angle = math.atan2(y2 - y1, x2 - x1)
    angle = angle % math.pi  # ignore direction (θ == θ + π)
    return angle


def _split_orientations(lines: Sequence[np.ndarray]) -> List[LineCluster]:
    """Split the detected lines into (ideally) two dominant orientation groups."""

    if not lines:
        return []

    angles = np.array([_angle_from_line(line) for line in lines])
    features = np.column_stack((np.cos(2 * angles), np.sin(2 * angles)))

    if len(lines) < 2:
        return [LineCluster(angles=angles.tolist(), lines=list(lines))]

    # Initialise centres by taking the most distant pair on the unit circle.
    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    idx1, idx2 = np.unravel_index(np.argmax(dists), dists.shape)
    centre1 = features[idx1]
    centre2 = features[idx2]

    # Classic 2-means on the doubled-angle representation.
    for _ in range(15):
        dist1 = np.linalg.norm(features - centre1, axis=1)
        dist2 = np.linalg.norm(features - centre2, axis=1)
        labels = dist1 <= dist2
        if labels.all() or (~labels).all():
            break
        centre1 = features[labels].mean(axis=0)
        centre2 = features[~labels].mean(axis=0)
        # normalise to keep numerical stability
        if np.linalg.norm(centre1) > 1e-6:
            centre1 /= np.linalg.norm(centre1)
        if np.linalg.norm(centre2) > 1e-6:
            centre2 /= np.linalg.norm(centre2)

    cluster1 = LineCluster(
        angles=angles[labels].tolist(),
        lines=[lines[i] for i in np.nonzero(labels)[0]],
    )

    cluster2_indices = np.nonzero(~labels)[0]
    if cluster2_indices.size == 0:
        return [cluster1]
    cluster2 = LineCluster(
        angles=angles[cluster2_indices].tolist(),
        lines=[lines[i] for i in cluster2_indices],
    )
    return [cluster1, cluster2]


def _support_lines(points: np.ndarray, direction_angle: float) -> Tuple[np.ndarray, float, float]:
    """Return the unit normal and the two offsets (min/max) for a given direction."""

    # A direction θ corresponds to an edge tangent; the outward normal is rotated by 90°.
    normal = np.array([-math.sin(direction_angle), math.cos(direction_angle)], dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise ValueError("Normal vector too small for stable computation.")
    normal /= norm

    projections = points @ normal
    return normal, float(np.min(projections)), float(np.max(projections))


def _intersect(n1: np.ndarray, c1: float, n2: np.ndarray, c2: float) -> np.ndarray:
    """Solve the intersection of n1·x=c1 and n2·x=c2."""

    A = np.vstack([n1, n2])
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        raise ValueError("Les droites de support sont quasi parallèles, intersection instable.")
    b = np.array([c1, c2])
    return np.linalg.solve(A, b)


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners ordered clockwise starting from top-left."""

    if corners.shape != (4, 2):
        raise ValueError("Quatre sommets sont nécessaires pour ordonner le quadrilatère.")

    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1).reshape(-1)

    rect[0] = corners[np.argmin(s)]  # top-left
    rect[2] = corners[np.argmax(s)]  # bottom-right
    rect[1] = corners[np.argmin(diff)]  # top-right
    rect[3] = corners[np.argmax(diff)]  # bottom-left
    return rect


def _min_area_rect(points: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    return _order_corners(box)


def _quadrilateral_from_hull(points: np.ndarray) -> Optional[np.ndarray]:
    """Try to approximate the convex hull by a 4-vertex polygon."""

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Les points doivent être fournis en 2D (N, 2).")

    contour = points.reshape((-1, 1, 2)).astype(np.float32)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 1e-6:
        return None

    # Explore several approximation tolerances, from detailed to coarse. We
    # keep the first quadrilateral found, which favours following the hull
    # geometry instead of collapsing to a rectangle.
    for epsilon_ratio in np.linspace(0.01, 0.08, num=8):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2))

    return None


def _save_overlay_image(
    image: np.ndarray,
    quad: np.ndarray,
    path: str,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Save ``image`` with ``quad`` drawn on top as a closed polyline."""

    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            canvas = image.copy()
        elif image.shape[2] == 4:
            canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(
                "L'image doit avoir 1, 3 ou 4 canaux pour l'enregistrement de la superposition."
            )
    else:
        raise ValueError(
            "L'image doit être 2D ou 3D pour l'enregistrement de la superposition."
        )

    pts = quad.reshape((-1, 1, 2)).astype(np.int32)
    bgr_color = tuple(int(c) for c in color)
    cv2.polylines(canvas, [pts], isClosed=True, color=bgr_color, thickness=int(thickness), lineType=cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def estimate_wall_quadrilateral(
    wall_mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    *,
    return_debug: bool = False,
    overlay_output_path: Optional[str] = None,
    overlay_color: Tuple[int, int, int] = (0, 255, 0),
    overlay_thickness: int = 3,
) -> np.ndarray | Tuple[np.ndarray, dict]:
    """Estimate the quadrilateral corresponding to a wall masked in the image.

    Parameters
    ----------
    wall_mask:
        Binary mask (np.uint8) where the wall pixels are 255. Only the largest
        connected component is considered.
    image:
        Optional original image (RGB or BGR). When provided we use it to
        stabilise line detection.
    return_debug:
        When ``True`` the function also returns intermediate data useful for
        diagnostics.
    overlay_output_path:
        When provided, save the original image with the estimated quadrilateral
        drawn on top (requires ``image`` to be supplied).
    overlay_color:
        BGR colour used to draw the quadrilateral on the saved overlay image.
    overlay_thickness:
        Thickness, in pixels, of the quadrilateral polyline when saving the
        overlay image.

    Returns
    -------
    np.ndarray of shape (4, 2)
        The quadrilateral points ordered clockwise starting from the top-left
        corner.
    dict (optional)
        Intermediate artefacts such as the convex hull, detected lines, etc.

    Raises
    ------
    ValueError
        If the mask does not contain any contour.
    """

    if wall_mask.ndim != 2:
        raise ValueError("Le masque doit être en niveaux de gris (2D).")
    if wall_mask.dtype != np.uint8:
        wall_mask = wall_mask.astype(np.uint8)

    if overlay_output_path is not None:
        if image is None:
            raise ValueError(
                "overlay_output_path nécessite l'image d'origine (paramètre image)."
            )
        if overlay_thickness <= 0:
            raise ValueError("overlay_thickness doit être strictement positif.")
        if len(overlay_color) != 3:
            raise ValueError("overlay_color doit contenir trois composantes BGR.")

    largest = _largest_contour(wall_mask)
    hull_points = _convex_hull(largest)

    # Build a clean mask for the hull to guide Hough.
    hull_mask = np.zeros_like(wall_mask)
    cv2.drawContours(
        hull_mask,
        [hull_points.astype(np.int32).reshape((-1, 1, 2))],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    if image is None:
        image_for_edges = wall_mask
    else:
        if image.ndim == 3 and image.shape[2] == 3:
            image_for_edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_for_edges = image

    lines = _hough_lines(image_for_edges, hull_mask)
    clusters = _split_orientations(lines)

    debug_data = {
        "largest_contour": largest,
        "convex_hull": hull_points,
        "lines": lines,
        "clusters": clusters,
    }

    def _finalize_result(quad: np.ndarray, *, mode: Optional[str] = None):
        if overlay_output_path is not None:
            assert image is not None  # checked above when overlay_output_path is set
            _save_overlay_image(
                image,
                quad,
                overlay_output_path,
                overlay_color,
                overlay_thickness,
            )
            if return_debug:
                debug_data["overlay_output_path"] = overlay_output_path
                debug_data["overlay_color"] = tuple(int(c) for c in overlay_color)
                debug_data["overlay_thickness"] = int(overlay_thickness)
        if return_debug:
            if mode is not None:
                debug_data["mode"] = mode
            return quad, debug_data
        return quad

    if len(clusters) < 2:
        quad = _quadrilateral_from_hull(hull_points)
        if quad is None:
            quad = _min_area_rect(hull_points)
            mode = "min_area_rect"
        else:
            mode = "hull_approx"
        return _finalize_result(quad, mode=mode)

    # Select the two clusters with most lines.
    clusters = sorted(clusters, key=lambda c: len(c.lines), reverse=True)[:2]
    angle1, angle2 = clusters[0].direction_angle, clusters[1].direction_angle

    # Ensure angles are sufficiently distinct; otherwise fall back.
    angle_diff = abs(angle1 - angle2) % math.pi
    if angle_diff > math.pi / 2:
        angle_diff = math.pi - angle_diff
    if angle_diff < math.radians(10):
        quad = _quadrilateral_from_hull(hull_points)
        mode = "hull_approx"
        if quad is None:
            quad = _min_area_rect(hull_points)
            mode = "min_area_rect"
        if return_debug:
            debug_data["angle_diff"] = angle_diff
        return _finalize_result(quad, mode=mode)

    normal1, offset1_min, offset1_max = _support_lines(hull_points, angle1)
    normal2, offset2_min, offset2_max = _support_lines(hull_points, angle2)

    try:
        p1 = _intersect(normal1, offset1_min, normal2, offset2_min)
        p2 = _intersect(normal1, offset1_min, normal2, offset2_max)
        p3 = _intersect(normal1, offset1_max, normal2, offset2_max)
        p4 = _intersect(normal1, offset1_max, normal2, offset2_min)
    except ValueError:
        quad = _quadrilateral_from_hull(hull_points)
        mode = "hull_approx"
        if quad is None:
            quad = _min_area_rect(hull_points)
            mode = "min_area_rect"
        return _finalize_result(quad, mode=mode)

    corners = np.stack([p1, p2, p3, p4]).astype(np.float32)
    ordered = _order_corners(corners)

    if return_debug:
        debug_data.update(
            {
                "angle1": angle1,
                "angle2": angle2,
                "support_normals": (normal1, normal2),
                "offsets": ((offset1_min, offset1_max), (offset2_min, offset2_max)),
                "raw_corners": corners,
            }
        )
    return _finalize_result(ordered, mode="hough_support")


__all__ = ["estimate_wall_quadrilateral"]