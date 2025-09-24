
import cv2
import numpy as np
import math
import itertools
from typing import Optional

def detect_largest_hollow_parallelepiped(contour, image_shape=None, min_area=0.0):
    """Détecte la plus grande cavité parallélépipédique d'un contour.

    L'algorithme procède en trois étapes :

    1. Calcule l'enveloppe convexe du ``contour`` pour obtenir une forme
       extérieure sans concavités.
    2. Soustrait le contour original de cette enveloppe afin d'obtenir les
       zones creuses laissées par les concavités (forme ``A`` décrite dans la
       demande).
    3. Recherche, parmi ces zones, le plus grand parallélépipède (ici modélisé
       comme un rectangle potentiellement orienté) via ``cv2.minAreaRect`` et
       retourne ce candidat.

    Args:
        contour (np.ndarray): contour OpenCV (``(N, 1, 2)``) dont on veut
            analyser les cavités.
        image_shape (Tuple[int, int] | None): dimensions ``(hauteur, largeur)``
            de l'image de référence. Si ``None``, un masque minimal sera créé à
            partir de la boîte englobante de l'enveloppe convexe.
        min_area (float): surface minimale (en pixels) exigée pour le
            parallélépipède retourné.

    Returns:
        dict | None: ``None`` si aucune cavité éligible n'est trouvée. Sinon un
        dictionnaire contenant :

            - ``"box"`` : les quatre sommets du parallélépipède détecté
              (``np.ndarray`` de forme ``(4, 2)``).
            - ``"area"`` : l'aire du parallélépipède.
            - ``"rect"`` : la représentation ``(centre, (largeur, hauteur),
              angle)`` produite par ``cv2.minAreaRect``.
            - ``"size"`` et ``"angle"`` : redondance pratique des dimensions
              et de l'orientation du parallélépipède.
            - ``"diff_contour"`` : le contour de la zone creuse à laquelle il
              appartient.
            - ``"all_candidates"`` : liste de tous les parallélépipèdes
              détectés dans la forme ``A`` (le plus grand y figure également).
    """

    if contour is None or len(contour) < 3:
        return None

    contour = np.asarray(contour, dtype=np.int32)


    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)

    if image_shape is None:
        x, y, w, h = cv2.boundingRect(hull)
        padding = 2
        offset = np.array([[x - padding, y - padding]], dtype=np.int32)
        mask_shape = (h + 2 * padding, w + 2 * padding)

        def _shift(cnt):
            return (cnt - offset).astype(np.int32)

        contour_mask_points = _shift(contour)
        hull_mask_points = _shift(hull)
    else:
        mask_shape = tuple(int(v) for v in image_shape[:2])
        offset = np.zeros((1, 2), dtype=np.int32)
        contour_mask_points = contour.astype(np.int32)
        hull_mask_points = hull.astype(np.int32)

    if mask_shape[0] <= 0 or mask_shape[1] <= 0:
        return None

    hull_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull_mask_points], -1, 255, thickness=cv2.FILLED)

    contour_mask = np.zeros_like(hull_mask)
    cv2.drawContours(contour_mask, [contour_mask_points], -1, 255, thickness=cv2.FILLED)

    diff_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(contour_mask))

    diff_contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = float(min_area)
    best_candidate = None
    best_area = min_area
    candidates = []
    quad_rects = []
    for diff_cnt in diff_contours:
        if len(diff_cnt) < 3:
            continue

        quad_local = _largest_inscribed_quadrilateral(diff_cnt, diff_mask)
        if quad_local is None:
            continue

        area = _polygon_area(quad_local)
        if area < min_area:
            continue

        quad_global = quad_local + offset.astype(np.float32)

        vertical_height = float(np.max(quad_global[:, 1]) - np.min(quad_global[:, 1]))
        print(f"Hauteur verticale maximale du quadrilatère : {vertical_height:.2f}")

        quad_rect = cv2.minAreaRect(quad_local.reshape(-1, 1, 2).astype(np.float32))
        quad_rects.append(quad_rect)
        (cx, cy), (w, h), angle = quad_rect

        offset_vec = offset.reshape(1, 1, 2)
        diff_cnt_global = diff_cnt + offset_vec.astype(diff_cnt.dtype)
        diff_cnt_global = diff_cnt_global.astype(np.int32)

        candidate = {
            "box": quad_global,
            "area": area,
            "rect": (
                (float(cx) + float(offset[0, 0]), float(cy) + float(offset[0, 1])),
                (float(w), float(h)),
                float(angle),
            ),
            "size": (float(w), float(h)),
            "angle": float(angle),
            "diff_contour": diff_cnt_global,
        }
        candidates.append(candidate)

        if area > best_area:
            best_candidate = candidate
            best_area = area

    if best_candidate is None:
        return None

    enriched_candidate = dict(best_candidate)
    enriched_candidate["all_candidates"] = candidates
    return enriched_candidate, quad_rects

def _largest_inscribed_quadrilateral(diff_cnt: np.ndarray, diff_mask: np.ndarray) -> Optional[np.ndarray]:
        """Cherche le quadrilatère de plus grande aire contenu dans ``diff_cnt``."""

        if diff_cnt is None or diff_mask is None or diff_mask.size == 0:
            return None

        points = diff_cnt.reshape(-1, 2).astype(np.float32)
        if len(points) < 4:
            return None

        # Simplifie les points tout en conservant la géométrie globale.
        perimeter = cv2.arcLength(points.reshape(-1, 1, 2), True)
        epsilon = max(1.0, 0.01 * perimeter)
        approx = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, True).reshape(-1, 2)
        if len(approx) >= 4:
            candidate_points = approx
        else:
            candidate_points = points

        hull = cv2.convexHull(candidate_points.reshape(-1, 1, 2)).reshape(-1, 2)
        if len(hull) == 4 and _polygon_inside_mask(hull, diff_mask):
            return hull.astype(np.float32)

        if len(hull) < 4:
            return None

        # Réduit le nombre de points pour limiter la combinatoire.
        if len(candidate_points) > 24:
            indices = np.linspace(0, len(candidate_points) - 1, num=24, dtype=int)
            candidate_points = candidate_points[indices]

        best_area = 0.0
        best_quad = None
        for combo in itertools.combinations(range(len(candidate_points)), 4):
            pts = candidate_points[list(combo)].astype(np.float32)
            quad = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(quad) != 4:
                continue

            if not _polygon_inside_mask(quad, diff_mask):
                continue

            area = _polygon_area(quad)
            if area > best_area:
                best_area = area
                best_quad = quad

        if best_quad is None:
            return None

        return best_quad.astype(np.float32)

def _polygon_area(points: np.ndarray) -> float:
        """Calcule l'aire signée d'un polygone."""

        if points is None or len(points) < 3:
            return 0.0

        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _polygon_inside_mask(polygon: np.ndarray, mask: np.ndarray) -> bool:
        """Vérifie que le quadrilatère ``polygon`` est inclus dans ``mask``."""

        if mask is None or mask.size == 0:
            return False

        poly = np.round(polygon).astype(np.int32)
        fill = np.zeros_like(mask)
        cv2.fillPoly(fill, [poly.reshape(-1, 1, 2)], 255)
        overlap = cv2.bitwise_and(fill, mask)
        return cv2.countNonZero(cv2.subtract(fill, overlap)) == 0
