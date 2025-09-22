import cv2
import numpy as np
import math

def contour_area(contour):
    """Calcule et retourne l'aire (valeur positive) d'un contour."""
    if contour is None or len(contour) == 0:
        return 0.0

    area = cv2.contourArea(contour)
    return float(abs(area))



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

        # Calcul d'un point de test : centroïde si disponible, sinon premier point
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            point = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        else:
            point = tuple(contour[0][0])

        parents = []
        for other_idx, other in enumerate(contours):
            if other_idx == idx or other is None or len(other) == 0:
                continue

            # pointPolygonTest retourne > 0 si point à l'intérieur, 0 sur le bord
            # et < 0 si à l'extérieur. On accepte l'intérieur ou sur le bord.
            if cv2.pointPolygonTest(other, point, False) >= 0:
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
    """
    Calcule les deux segments les plus longs non verticaux d'un contour.

    Returns:
        Tuple des quatre points constituant les deux segments ou ``None`` si
        aucun couple valide n'est trouvé.
    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 2:
        return None

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
            if abs(dx) >50 :
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
        return None
    print(f"Line 1: {outpt11_, outpt12_}")
    print(f"Line 2: {outpt21, outpt22}")

    return outpt11_, outpt12_, outpt21, outpt22

