import cv2
import numpy as np
import math
from typing import List, Optional

def contour_area(contour):
    """Calcule et retourne l'aire (valeur positive) d'un contour."""
    if contour is None or len(contour) == 0:
        return 0.0

    area = cv2.contourArea(contour)
    return float(abs(area))

def _extract_xy(vertex):
    """Convertit un sommet de contour OpenCV en tuple ``(x, y)`` flottant."""

    flat = np.asarray(vertex, dtype=float).reshape(-1)
    if flat.size < 2:
        raise ValueError("Vertex does not contain two coordinates")
    return float(flat[0]), float(flat[1])

def checkContoursIndide(contours):
    """Log l'appartenance d'un contour à un autre dans la même liste.

    Pour chaque contour, on tente de récupérer un point représentatif (le
    centroïde si possible) et on vérifie si ce point est situé à l'intérieur
    d'un ou plusieurs autres contours de la liste. Les informations sont
    affichées via ``print`` et la structure de résultat est retournée pour un
    usage éventuel ultérieur.
    """

    inclusion_map = []

    for idx, contour in enumerate(contours):
        if contour is None or len(contour) == 0:
            inclusion_map.append((idx, []))
            print(f"Contour {idx} est vide ou non défini.")
            continue

        parents = []
        for other_idx, other in enumerate(contours):
            if other_idx == idx or other is None or len(other) == 0:
                continue

            # pointPolygonTest retourne > 0 si point à l'intérieur, 0 sur le bord
            # et < 0 si à l'extérieur. On exige que l'intégralité du contour soit
            # incluse : chaque point doit être à l'intérieur ou sur le bord.
            if all(
                cv2.pointPolygonTest(other, _extract_xy(pt), False) >= 0
                for pt in contour
            ):
                parents.append(other_idx)

        if parents:
            print(f"Contour {idx} est entouré par les contours {parents}.")
        else:
            print(f"Contour {idx} n'est entouré par aucun autre contour.")

        inclusion_map.append((idx, parents))

    return inclusion_map


def dilate_contour(cnt, image_shape, dilation_px=4):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    new_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return new_cnts[0]


def contour_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
    return tuple(contour[0][0])

def build_contour_mask(target_contour, all_contours, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [target_contour], -1, 255, thickness=cv2.FILLED)

    exclusion = np.zeros_like(mask)
    for other in all_contours:
        if other is None or len(other) == 0 or other is target_contour:
            continue

        centroid = contour_centroid(other)
        if cv2.pointPolygonTest(target_contour, centroid, False) >= 0:
            cv2.drawContours(exclusion, [other], -1, 255, thickness=cv2.FILLED)

    if cv2.countNonZero(exclusion) == 0:
        return mask

    return cv2.bitwise_and(mask, cv2.bitwise_not(exclusion))

def findPointsFromContour(cnt):
    """Retourne les points caractéristiques et l'approximation polygonale.

    Calcule les deux segments les plus longs non verticaux d'un contour et
    fournit également le polygone approché obtenu via ``cv2.approxPolyDP``.

    Returns:
        Tuple[(Tuple[np.ndarray, ...] | None), np.ndarray]:
            - Les quatre points définissant les deux segments horizontaux
              détectés (ou ``None`` si aucun segment valide n'est trouvé).
            - Le polygone approché « approx » (tableau de points OpenCV).
    """
    epsilon = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx = cv2.convexHull(approx)
    #approx = cnt

    if len(approx) < 2:
        return None, approx

    hor_segmentsbylength = []
    hor_segmentsbyheight = []
    hor_segmentsbyheightdesc = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        height = min(pt2[1],pt1[1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_deg = - angle_deg % 180
        # Vertical walls stay vertical despite perspspective
        if angle_deg < 80 or angle_deg > 100:
            if abs(dx) >20 :
                hor_segmentsbylength.append((length, height, dx, dy, pt1, pt2))
                hor_segmentsbyheight.append((length, height, dx, dy, pt1, pt2))
                hor_segmentsbyheightdesc.append((length, height, dx, dy, pt1, pt2))

    hor_segmentsbylength.sort(key=lambda seg: seg[0], reverse=True)
    hor_segmentsbyheight.sort(key=lambda seg: seg[1], reverse=True)
    hor_segmentsbyheightdesc.sort(key=lambda seg: seg[1], reverse=False)

    outpt11 = outpt12 = None
    outpt11_ = outpt12_ = None
    outpt21 = outpt22 = None
    if (hor_segmentsbylength):
        #longest
        outpt11 = hor_segmentsbylength[0][4]
        outpt12 = hor_segmentsbylength[0][5]

        #highest
        outpt11_ = hor_segmentsbyheightdesc[0][4]
        outpt12_ = hor_segmentsbyheightdesc[0][5]
        #lowest
        outpt21 = hor_segmentsbyheight[0][4]
        outpt22 = hor_segmentsbyheight[0][5]

    if any(pt is None for pt in (outpt11_, outpt12_, outpt21, outpt22)):
        return None, approx
    print(f"Line 1: {outpt11_, outpt12_}")
    print(f"Line 2: {outpt21, outpt22}")

    return (outpt11_, outpt12_, outpt21, outpt22), approx


def _contour_segment(contour: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Return the sub-contour from ``start_idx`` to ``end_idx`` inclusive."""

    if contour is None or len(contour) == 0:
        return np.empty((0, 1, 2), dtype=np.int32)

    contour = np.asarray(contour, dtype=np.int32)
    n = len(contour)
    if n == 0:
        return np.empty((0, 1, 2), dtype=np.int32)

    segment_points = []
    idx = start_idx % n
    end_idx = end_idx % n

    while True:
        segment_points.append(contour[idx][0])
        if idx == end_idx:
            break
        idx = (idx + 1) % n
        if idx == start_idx:
            break

    if len(segment_points) < 3:
        return np.empty((0, 1, 2), dtype=np.int32)

    return np.asarray(segment_points, dtype=np.int32).reshape((-1, 1, 2))


def find_concavities(
    contour: np.ndarray,
    min_depth: float = 5.0,
    min_area: float = 25.0,
) -> List[np.ndarray]:
    """Identify the concave regions of ``contour`` as individual polygons."""

    if contour is None or len(contour) < 4:
        return []

    contour = np.asarray(contour, dtype=np.int32)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return []

    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return []

    concavities = []
    depth_threshold = max(0.0, float(min_depth)) * 256.0

    for start_idx, end_idx, far_idx, depth in defects[:, 0]:
        if depth < depth_threshold:
            continue

        segment = _contour_segment(contour, int(start_idx), int(end_idx))
        if segment.size == 0:
            continue

        farthest_point = contour[int(far_idx) % len(contour)][0]
        if not any(np.array_equal(pt[0], farthest_point) for pt in segment):
            segment = np.vstack((segment, np.array([[farthest_point]], dtype=np.int32)))

        area = abs(cv2.contourArea(segment))
        if area < float(min_area):
            continue

        concavities.append(segment)

    return concavities


def _resample_concavity_vertices(
    concavity: np.ndarray,
    max_vertices: int = 24,
    step: float = 8.0,
) -> np.ndarray:
    """Simplify and densify ``concavity`` to obtain candidate vertices."""

    if concavity is None:
        return np.empty((0, 2), dtype=np.float32)

    contour = np.asarray(concavity, dtype=np.float32).reshape((-1, 1, 2))
    if len(contour) == 0:
        return np.empty((0, 2), dtype=np.float32)

    perimeter = max(cv2.arcLength(contour, True), 1.0)
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 4:
        approx = contour

    points = approx.reshape((-1, 2)).astype(np.float32)
    densified: List[np.ndarray] = []
    num_points = len(points)
    if num_points == 0:
        return np.empty((0, 2), dtype=np.float32)

    for idx in range(num_points):
        p1 = points[idx]
        p2 = points[(idx + 1) % num_points]
        densified.append(p1)
        dist = float(np.linalg.norm(p2 - p1))
        if dist <= step:
            continue
        subdivisions = int(dist // step)
        for sub_idx in range(1, subdivisions + 1):
            t = sub_idx / (subdivisions + 1)
            intermediate = (1.0 - t) * p1 + t * p2
            densified.append(intermediate.astype(np.float32))

    if len(densified) > max_vertices:
        indices = np.linspace(0, len(densified) - 1, num=max_vertices, dtype=int)
        densified = [densified[i] for i in indices]

    unique_points: List[np.ndarray] = []
    for pt in densified:
        if not unique_points:
            unique_points.append(pt)
            continue
        if np.linalg.norm(pt - unique_points[-1]) > 1e-3:
            unique_points.append(pt)

    return np.asarray(unique_points, dtype=np.float32)


def _polygon_area(points: np.ndarray) -> float:
    area = 0.0
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    for idx in range(len(pts)):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _is_convex_quad(quad: np.ndarray) -> bool:
    pts = np.asarray(quad, dtype=np.float32)
    if len(pts) != 4:
        return False

    cross_sign = 0.0
    for idx in range(4):
        a = pts[idx]
        b = pts[(idx + 1) % 4]
        c = pts[(idx + 2) % 4]
        ab = b - a
        bc = c - b
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        if abs(cross) < 1e-6:
            continue
        current_sign = math.copysign(1.0, cross)
        if cross_sign == 0.0:
            cross_sign = current_sign
        elif current_sign != cross_sign:
            return False
    return _polygon_area(pts) > 1e-3


def _points_inside_contour(points: List[np.ndarray], contour: np.ndarray) -> bool:
    contour_for_test = np.asarray(contour, dtype=np.float32).reshape((-1, 1, 2))
    for pt in points:
        if cv2.pointPolygonTest(
            contour_for_test, (float(pt[0]), float(pt[1])), False
        ) < -1e-3:
            return False
    return True


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_properly_intersect(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float = 1e-5
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (
        ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
        and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))
    ):
        return True

    return False


def _quad_within_concavity(quad: np.ndarray, concavity: np.ndarray) -> bool:
    pts = [np.asarray(p, dtype=np.float32) for p in quad]
    if not _points_inside_contour(pts, concavity):
        return False

    concavity_points = np.asarray(concavity, dtype=np.float32).reshape((-1, 2))
    num_concavity_pts = len(concavity_points)
    if num_concavity_pts >= 2:
        for idx in range(4):
            p1 = pts[idx]
            p2 = pts[(idx + 1) % 4]
            for jdx in range(num_concavity_pts):
                q1 = concavity_points[jdx]
                q2 = concavity_points[(jdx + 1) % num_concavity_pts]
                if _segments_properly_intersect(p1, p2, q1, q2):
                    return False

    for idx in range(4):
        edge_start = pts[idx]
        edge_end = pts[(idx + 1) % 4]
        for t in (0.25, 0.5, 0.75):
            sample = (1.0 - t) * edge_start + t * edge_end
            if cv2.pointPolygonTest(
                concavity_points.reshape((-1, 1, 2)),
                (float(sample[0]), float(sample[1])),
                False,
            ) < -1e-3:
                return False

    edges_midpoints = []
    for idx in range(4):
        midpoint = 0.5 * (pts[idx] + pts[(idx + 1) % 4])
        edges_midpoints.append(midpoint)

    centroid = sum(pts) / 4.0
    triangle_centroids = [
        (pts[0] + pts[1] + pts[2]) / 3.0,
        (pts[0] + pts[2] + pts[3]) / 3.0,
    ]

    return _points_inside_contour(edges_midpoints + [centroid] + triangle_centroids, concavity)


def largest_quadrilateral_in_concavity(
    concavity: np.ndarray,
    max_vertices: int = 24,
    sampling_step: float = 8.0,
) -> Optional[np.ndarray]:
    """Approximate the largest quadrilateral fitting inside ``concavity``.

    The quadrilateral vertices are chosen among sampled points lying on the
    concavity boundary to guarantee the resulting polygon stays fully inside
    the concave region.
    """

    if concavity is None or len(concavity) < 4:
        return None

    candidates = _resample_concavity_vertices(concavity, max_vertices, sampling_step)
    if len(candidates) < 4:
        return None

    best_quad: Optional[np.ndarray] = None
    best_area = 0.0

    num_candidates = len(candidates)
    for i in range(num_candidates - 3):
        for j in range(i + 1, num_candidates - 2):
            for k in range(j + 1, num_candidates - 1):
                for l in range(k + 1, num_candidates):
                    quad = np.array(
                        [
                            candidates[i],
                            candidates[j],
                            candidates[k],
                            candidates[l],
                        ],
                        dtype=np.float32,
                    )

                    if not _is_convex_quad(quad):
                        continue
                    if not _quad_within_concavity(quad, concavity):
                        continue

                    area = _polygon_area(quad)
                    if area > best_area:
                        best_area = area
                        best_quad = quad.copy()

    return best_quad


def findPointsFromContour2(cnt):
    """Retourne les lignes dominantes supérieure et inférieure d'un contour.

    Cette variante regroupe les segments horizontaux non verticaux détectés en
    deux familles : ceux situés dans la partie supérieure du contour et ceux
    dans la partie inférieure. Pour chaque famille, une ligne « dominante » est
    calculée en faisant la moyenne pondérée (par la longueur de chaque segment)
    des extrémités correspondantes.

    Returns:
        Tuple[(Tuple[Tuple[int, int], ...] | None), np.ndarray]:
            - Les quatre points définissant les lignes dominante supérieure puis
              inférieure (ou ``None`` si une des deux lignes ne peut être
              déterminée).
            - Le polygone approché « approx » (tableau de points OpenCV).
    """

    epsilon = 0.002 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 2:
        return None, approx

    segments = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        angle_deg = math.degrees(math.atan2(dy, dx))
        angle_deg = -angle_deg % 180
        if angle_deg < 80 or angle_deg > 100:
            if abs(dx) > 20:
                avg_height = (pt1[1] + pt2[1]) / 2.0
                segments.append((length, avg_height, pt1, pt2))

    if not segments:
        return None, approx

    ys = [pt[0][1] for pt in approx]
    top_y = min(ys)
    bottom_y = max(ys)
    mid_y = (top_y + bottom_y) / 2.0
    upper_limit = (top_y + mid_y) / 2.0
    lower_limit = (bottom_y + mid_y) / 2.0

    def dominant_line(filter_fn):
        filtered = [seg for seg in segments if filter_fn(seg)]
        total_length = sum(seg[0] for seg in filtered)
        if total_length <= 0:
            return None

        left_sum = np.zeros(2, dtype=float)
        right_sum = np.zeros(2, dtype=float)
        for length, _, pt1, pt2 in filtered:
            if pt1[0] < pt2[0] or (pt1[0] == pt2[0] and pt1[1] <= pt2[1]):
                left_pt, right_pt = pt1, pt2
            else:
                left_pt, right_pt = pt2, pt1
            left_sum += length * np.array(left_pt, dtype=float)
            right_sum += length * np.array(right_pt, dtype=float)

        left_pt = tuple(int(round(coord)) for coord in (left_sum / total_length))
        right_pt = tuple(int(round(coord)) for coord in (right_sum / total_length))
        return left_pt, right_pt

    top_line = dominant_line(lambda seg: seg[1] <= upper_limit)
    bottom_line = dominant_line(lambda seg: seg[1] >= lower_limit)

    if not top_line or not bottom_line:
        return None, approx

    top_pt1, top_pt2 = top_line
    bottom_pt1, bottom_pt2 = bottom_line

    print(f"Dominant top line: {(top_pt1, top_pt2)}")
    print(f"Dominant bottom line: {(bottom_pt1, bottom_pt2)}")

def floodfill_extract_contours(image_gray):
    """
    Utilise floodFill pour détecter chaque zone noire connectée,
    évite les pixels déjà remplis, et retourne les vrais contours.
    """
    thresholdToBeAZone = 0.002;
    contour_area_threshold = thresholdToBeAZone * float(image_gray.shape[0] * image_gray.shape[1])

    img = image_gray.copy()
    h, w = img.shape
    contours = []

    # floodFill a besoin d'un masque 2 pixels plus grand
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for y in range(h):
        for x in range(w):
            if img[y, x] == 0:  # pixel noir non traité (car floodFill va peindre en 255)
                # Reset du masque pour chaque floodFill
                flood_mask = np.zeros((h + 2, w + 2), np.uint8)

                # Appliquer floodFill : il modifie img en place
                cv2.floodFill(img, flood_mask, seedPoint=(x, y), newVal=255)

                # Extraire la zone peinte (flood_mask = 1 où rempli)
                filled_area = (flood_mask[1:-1, 1:-1] == 1).astype(np.uint8) * 255
                surface_pixels = cv2.countNonZero(filled_area)
                if surface_pixels > contour_area_threshold:
                    if (not filled_area_touche_haut(filled_area) and not filled_area_touche_bas(filled_area)):
                        cnts, _ = cv2.findContours(filled_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                        contours.extend(cnts)

    return contours

def filled_area_touche_haut(filled_area):
    return np.any(filled_area[0] == 255)


def filled_area_touche_bas(filled_area):
    return np.any(filled_area[-1] == 255)