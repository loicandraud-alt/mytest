"""Outils pour construire un quadrilatère délimité par deux lignes inclinées."""

from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]
Line = Tuple[Point, Point]

_EPSILON = 1e-9
_VERTICAL_MIN_DEGREES = 75.0
_VERTICAL_MAX_DEGREES = 105.0


def _segment_length(line: Line) -> float:
    """Calcule la longueur euclidienne d'un segment."""
    (x1, y1), (x2, y2) = line
    return math.hypot(x2 - x1, y2 - y1)


def _normalized_angle_deg(line: Line) -> float:
    """Retourne l'angle du segment par rapport à l'axe des abscisses dans [0, 180)."""
    (x1, y1), (x2, y2) = line
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    return (math.degrees(angle_rad) + 180.0) % 180.0


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


def _line_parameters(line: Line) -> Tuple[float, float]:
    """Retourne la pente et l'ordonnée à l'origine d'une droite."""
    (x1, y1), (x2, y2) = line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def quadrilateral_from_lines(line1: Line, line2: Line, extension_factor: float = 100.0) -> List[Point]:
    """Construit le quadrilatère délimité par deux lignes inclinées et deux droites verticales.

    Args:
        line1: Premier segment (non vertical) défini par deux points (x, y).
        line2: Second segment (non vertical) défini par deux points (x, y).
        extension_factor: Facteur d'allongement appliqué à chaque segment.

    Returns:
        Les quatre sommets du quadrilatère dans l'ordre suivant :
        haut-gauche, bas-gauche, bas-droite, haut-droite.

    Raises:
        ValueError: si une ligne est verticale, dégénérée ou si le facteur est invalide.
    """
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


__all__ = ["quadrilateral_from_lines"]