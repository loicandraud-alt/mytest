"""Utility functions to project an image on to a planar surface using a perspective transform."""

from __future__ import annotations

from typing import Iterable, Tuple
from typing import List
from typing import List, Tuple, Optional
import math

import cv2
import numpy as np



Point = Tuple[float, float]
Line = Tuple[Point, Point]

_EPSILON = 1e-9
_VERTICAL_MIN_DEGREES = 75.0
_VERTICAL_MAX_DEGREES = 105.0




def quadrilateral_from_lines(
    line1: Optional[Line] = None,
    line2: Optional[Line] = None,
    extension_factor: float = 100.0,
    contour: Optional[Iterable[Point]] = None,
) -> List[Point]:
    """Construit un quadrilatère contenant un ensemble de points.

    Par défaut, on retrouve l'ancien comportement : le quadrilatère est bâti à
    partir de deux segments inclinés et de deux droites verticales.  Lorsque
    ``contour`` est fourni, la méthode applique l'algorithme des gabarits
    tournants (rotating calipers) sur l'enveloppe convexe pour obtenir le plus
    petit quadrilatère englobant tous les points du contour.

    Args:
        line1: Premier segment (non vertical) défini par deux points (x, y).
            Optionnel si ``contour`` est fourni.
        line2: Second segment (non vertical) défini par deux points (x, y).
            Optionnel si ``contour`` est fourni.
        extension_factor: Facteur d'allongement appliqué à chaque segment lorsque
            l'on utilise les deux lignes.
        contour: Points du contour (ou de son enveloppe convexe) à englober.

    Returns:
        Les quatre sommets du quadrilatère dans l'ordre suivant :
        haut-gauche, bas-gauche, bas-droite, haut-droite.

    Raises:
        ValueError: si ``contour`` est vide ou si les lignes fournies sont
            verticales, dégénérées ou le facteur invalide.
    """

    if contour is not None:
        return _minimum_bounding_quadrilateral(contour)

    if line1 is None or line2 is None:
        raise ValueError("line1 et line2 doivent être fournis si aucun contour n'est donné.")

    _validate_line(line1)
    _validate_line(line2)

    if extension_factor <= 0:
        raise ValueError("Le facteur d'extension doit être strictement positif.")

    length1 = _segment_length(line1)
    length2 = _segment_length(line2)
    longest_line = line1 if length1 >= length2 else line2

    min_x = min(longest_line[0][0], longest_line[1][0])
    max_x = max(longest_line[0][0], longest_line[1][0])

    extended_line1 = _extend_line(line1, extension_factor)
    extended_line2 = _extend_line(line2, extension_factor)

    slope1, intercept1 = _line_parameters(extended_line1)
    slope2, intercept2 = _line_parameters(extended_line2)

    left_points = [
        (min_x, slope1 * min_x + intercept1),
        (min_x, slope2 * min_x + intercept2),
    ]
    right_points = [
        (max_x, slope1 * max_x + intercept1),
        (max_x, slope2 * max_x + intercept2),
    ]

    left_points.sort(key=lambda pt: pt[1])
    right_points.sort(key=lambda pt: pt[1])

    top_left, bottom_left = left_points
    top_right, bottom_right = right_points

    return [top_left, bottom_left, bottom_right, top_right]

def _minimum_bounding_quadrilateral(points: Iterable[Point]) -> List[Point]:
    """Calcule le plus petit quadrilatère englobant avec les gabarits tournants."""

    points_array = _as_point_array(points)
    if points_array.size == 0:
        raise ValueError("Le contour doit contenir au moins un point.")

    if len(points_array) == 1:
        pt = tuple(points_array[0])
        return [pt, pt, pt, pt]

    hull = cv2.convexHull(points_array.astype(np.float32))
    hull_points = hull.reshape(-1, 2).astype(float)

    if len(hull_points) == 1:
        pt = tuple(hull_points[0])
        return [pt, pt, pt, pt]

    if len(hull_points) == 2:
        min_x = float(np.min(hull_points[:, 0]))
        max_x = float(np.max(hull_points[:, 0]))
        min_y = float(np.min(hull_points[:, 1]))
        max_y = float(np.max(hull_points[:, 1]))
        corners = np.array(
            [
                [min_x, min_y],
                [min_x, max_y],
                [max_x, max_y],
                [max_x, min_y],
            ]
        )
        return _order_points(corners)

    edges = np.diff(np.vstack([hull_points, hull_points[0]]), axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.mod(angles, math.pi / 2.0)
    unique_angles = np.unique(np.round(angles, decimals=12))

    best_area = math.inf
    best_corners: Optional[np.ndarray] = None

    for angle in unique_angles:
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated = hull_points @ rotation_matrix.T

        min_x = float(np.min(rotated[:, 0]))
        max_x = float(np.max(rotated[:, 0]))
        min_y = float(np.min(rotated[:, 1]))
        max_y = float(np.max(rotated[:, 1]))

        area = (max_x - min_x) * (max_y - min_y)
        if area < best_area - _EPSILON:
            best_area = area
            rect = np.array(
                [
                    [min_x, min_y],
                    [min_x, max_y],
                    [max_x, max_y],
                    [max_x, min_y],
                ]
            )
            best_corners = rect @ rotation_matrix

    if best_corners is None:
        raise RuntimeError("Échec du calcul du quadrilatère minimal.")

    ordered = _order_points(best_corners)
    return [tuple(pt) for pt in ordered]


def _as_point_array(points: Iterable[Point]) -> np.ndarray:
    if isinstance(points, np.ndarray):
        arr = points
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr.astype(float)

    point_list = list(points)
    arr = np.asarray(point_list, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr.astype(float)


def _order_points(points: np.ndarray) -> List[Point]:
    if points.shape != (4, 2):
        raise ValueError("Quatre points sont requis pour l'ordonnancement.")

    s = points.sum(axis=1)
    diff = points[:, 0] - points[:, 1]

    ordered = np.zeros((4, 2), dtype=float)
    ordered[0] = points[np.argmin(s)]  # top-left
    ordered[2] = points[np.argmax(s)]  # bottom-right
    ordered[1] = points[np.argmin(diff)]  # bottom-left
    ordered[3] = points[np.argmax(diff)]  # top-right

    return [tuple(pt) for pt in ordered]

def _validate_line(line: Line) -> None:
    """Vérifie qu'un segment est utilisable (non nul et non vertical)."""
    if math.isclose(_segment_length(line), 0.0, abs_tol=_EPSILON):
        raise ValueError("Chaque ligne doit être définie par deux points distincts.")

    angle = _normalized_angle_deg(line)
    if _VERTICAL_MIN_DEGREES - _EPSILON <= angle <= _VERTICAL_MAX_DEGREES + _EPSILON:
        raise ValueError(
            "Les segments compris entre 75° et 105° sont considérés comme verticaux et ne sont pas supportés."
        )


def _extend_line(line: Line, factor: float) -> Line:
    """Retourne une version prolongée d'un segment autour de son milieu."""
    if factor <= 0:
        raise ValueError("Le facteur d'extension doit être strictement positif.")

    (x1, y1), (x2, y2) = line
    length = _segment_length(line)
    if math.isclose(length, 0.0, abs_tol=_EPSILON):  # sécurité supplémentaire
        raise ValueError("Impossible d'étendre un segment de longueur nulle.")

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    unit_dx = (x2 - x1) / length
    unit_dy = (y2 - y1) / length
    half_extended_length = (length * factor) / 2.0

    p_start = (mid_x - unit_dx * half_extended_length, mid_y - unit_dy * half_extended_length)
    p_end = (mid_x + unit_dx * half_extended_length, mid_y + unit_dy * half_extended_length)
    return p_start, p_end


def _segment_length(line: Line) -> float:
    """Calcule la longueur euclidienne d'un segment."""
    (x1, y1), (x2, y2) = line
    return math.hypot(x2 - x1, y2 - y1)


def _normalized_angle_deg(line: Line) -> float:
    """Retourne l'angle du segment par rapport à l'axe des abscisses dans [0, 180)."""
    (x1, y1), (x2, y2) = line
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    return (math.degrees(angle_rad) + 180.0) % 180.0


def _line_parameters(line: Line) -> Tuple[float, float]:
    """Retourne la pente et l'ordonnée à l'origine d'une droite."""
    (x1, y1), (x2, y2) = line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept
