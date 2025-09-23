import cv2
import numpy as np
import math

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