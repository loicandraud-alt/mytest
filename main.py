import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque

from PIL import Image, ImageFilter

from webercolor.linequedrilareral import quadrilateral_from_lines
from typing import List, Tuple
from webercolor.linequedrilareral import quadrilateral_from_lines
from wallquad import estimate_wall_quadrilateral
from perpspectiveoverlay import project_texture

def maxlength(approx):
    max_len = 0.0
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]  # boucle fermée
        length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if length > max_len:
            max_len = length
    return max_len


def filled_area_touche_haut(filled_area):
    return np.any(filled_area[0] == 255)


def filled_area_touche_bas(filled_area):
    return np.any(filled_area[-1] == 255)


def floodfill_extract_contours(image_gray):
    """
    Utilise floodFill pour détecter chaque zone noire connectée,
    évite les pixels déjà remplis, et retourne les vrais contours.
    """
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
                if surface_pixels > 5000:
                    if (not filled_area_touche_haut(filled_area) and not filled_area_touche_bas(filled_area)):
                        cnts, _ = cv2.findContours(filled_area, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                        contours.extend(cnts)

    return contours

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

def findlines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    lines_img = np.zeros_like(img)

    # Dessiner les lignes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rouge
    return lines_img


def dilate_contour(cnt, image_shape, dilation_px=4):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    new_cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return new_cnts[0]


def mergeimages(image1, image2):
    gray_result = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, texture_mask = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
    texture_mask_inv = cv2.bitwise_not(texture_mask)
    background_cleaned = cv2.bitwise_and(image1, image1, mask=texture_mask_inv)
    textured_cleaned = cv2.bitwise_and(image2, image2, mask=texture_mask)
    return cv2.bitwise_or(background_cleaned, textured_cleaned)

def mergeimages(image1, image2):
        gray_result = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        _, texture_mask = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
        texture_mask_inv = cv2.bitwise_not(texture_mask)
        background_cleaned = cv2.bitwise_and(image1, image1, mask=texture_mask_inv)
        textured_cleaned = cv2.bitwise_and(image2, image2, mask=texture_mask)
        return cv2.bitwise_or(background_cleaned, textured_cleaned)

def tile_texture(texture: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Repeat ``texture`` to cover a rectangle of ``target_width`` × ``target_height``."""

        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target dimensions must be strictly positive to tile a texture.")

        tex_height, tex_width = texture.shape[:2]
        if tex_width == 0 or tex_height == 0:
            raise ValueError("Texture must have non-zero dimensions.")

        repeat_x = max(1, math.ceil(target_width / tex_width))
        repeat_y = max(1, math.ceil(target_height / tex_height))
        tiled = np.tile(texture, (repeat_y, repeat_x, 1))
        return tiled[:target_height, :target_width]

def boostimagegray(img):
        alpha = 1
        beta = 0
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.5)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge((h, s, v))
        color_boosted = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(color_boosted, cv2.COLOR_BGR2GRAY)


def findAngle(cnt):
    """
    Calcule l'angle du segment le plus long d'un contour après approximation.
    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    max_len = 0
    angle_deg = 0
    if len(approx) < 2:
        return 0

    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length
            angle_deg = math.degrees(math.atan2(dy, dx))
    angle_deg = - angle_deg % 180
    #Vertical walls stay vertical despite perspspective
    if angle_deg >75 and angle_deg <105:
        angle_deg = angle_deg - 90
    return angle_deg

def findAngle2(cnt):
    """
    Calcule l'angle du segment le plus long d'un contour après approximation.
    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) < 2:
        return 0

    segments = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        segments.append((length, dx, dy))

    segments.sort(key=lambda seg: seg[0], reverse=True)

    for length, dx, dy in segments:
        # To improve, if vertical lnes
        if abs(dx) < 50:
            continue
        angle_deg = math.degrees(math.atan2(dy, dx)) % 180
        return angle_deg

    return 0

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

    segments = []
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        segments.append((length, dx, dy, pt1, pt2))

    segments.sort(key=lambda seg: seg[0], reverse=True)

    outpt11 = outpt12 = outpt21 = outpt22 = None

    for length, dx, dy, pt1, pt2 in segments:
        # To improve, if vertical lnes
        if abs(dx) < 50 : #and abs(dy > 30):
            continue
        if outpt11 is None:
            outpt11 = pt1
            outpt12 = pt2
        elif outpt21 is None:
            outpt21 = pt1
            outpt22 = pt2
            break
    if any(pt is None for pt in (outpt11, outpt12, outpt21, outpt22)):
        return None
    return outpt11, outpt12, outpt21, outpt22

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

def drawFile(path, image, edges, dilatation, mode):
    kernel = np.ones((dilatation, dilatation), np.uint8)
    myedgesdilatated = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(path + "_result_edges.jpg", myedgesdilatated)
    color_zones = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    toto = floodfill_extract_contours(myedgesdilatated)
    textured_image = image.copy()
    background_with_quads = image.copy()
    background_with_wallquad_quads = image.copy()

    # Charger et préparer les textures
    textures_files = {
        "brique": cv2.resize(cv2.imread("../webercolor/textures/brique.jpg"), (0, 0), fx=0.5, fy=0.5),
        "brique2": cv2.resize(cv2.imread("../webercolor/textures/brique2.jpg"), (0, 0), fx=0.2, fy=0.2),
        "brique3": cv2.resize(cv2.imread("../webercolor/textures/brique3.jpg"), (0, 0), fx=0.2, fy=0.2),
        "bois1": cv2.resize(cv2.imread("../webercolor/textures/bois1.jpg"), (0, 0), fx=0.5, fy=0.5),
        "bois2": cv2.resize(cv2.imread("../webercolor/textures/bois2.jpg"), (0, 0), fx=0.2, fy=0.2),
        "pierre": cv2.resize(cv2.imread("../webercolor/textures/pierre.jpg"), (0, 0), fx=0.1, fy=0.1),
        "enduit1": cv2.resize(cv2.imread("../webercolor/textures/enduit1.jpg"), (0, 0), fx=0.05, fy=0.05),
    }
    textures = list(textures_files.values())
    checkContoursIndide(toto)
    done = 0

    for cnt in toto:
        if done != 0:
            continue
        done = done + 1
        #1. Obtenir l'angle et prendre son opposé pour la correction␊
        angle = findAngle2(cnt)
        points = findPointsFromContour(cnt)
        quadrilateral_points = None
        if points is not None:
            pt11, pt12, pt21, pt22 = points
            try:
                line1 = ((float(pt11[0]), float(pt11[1])), (float(pt12[0]), float(pt12[1])))
                line2 = ((float(pt21[0]), float(pt21[1])), (float(pt22[0]), float(pt22[1])))
                quadrilateral_points = quadrilateral_from_lines(line1, line2)
            except ValueError:
                quadrilateral_points = None
            if quadrilateral_points:
                quad_array = np.array([[int(round(x)), int(round(y))] for x, y in quadrilateral_points],          dtype=np.int32)
                cv2.polylines(background_with_quads, [quad_array], isClosed=True, color=(0, 0, 255), thickness=3)
            wall_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(wall_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            try:
                wall_quad = estimate_wall_quadrilateral(wall_mask, image=image)
            except ValueError:
                wall_quad = None
            except Exception as exc:
                print(f"Erreur lors de l'estimation du quadrilatère avec wallquad: {exc}")
                wall_quad = None

            if wall_quad is not None:
                wall_quad_int = wall_quad.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(
                    background_with_wallquad_quads,
                    [wall_quad_int],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=3,
                )

            # 2. Choisir une texture (on ne la tourne pas ici)
            chosen_texture = random.choice(textures)
            dilatatedcnt = dilate_contour(cnt, image.shape, 3)
            x, y, w, h = cv2.boundingRect(dilatatedcnt)
            if quadrilateral_points:
                overlay_points = [
                    quadrilateral_points[0],  # top-left
                    quadrilateral_points[3],  # top-right
                    quadrilateral_points[2],  # bottom-right
                    quadrilateral_points[1],  # bottom-left
                ]
                try:
                    tiled_texture = tile_texture(chosen_texture, max(w, 1), max(h, 1))
                    textured_image = project_texture(
                        textured_image,
                        tiled_texture,
                        overlay_points,
                        clip_mask=wall_mask,
                    )
                    continue
                except Exception as exc:
                    print(f"Projection perspective échouée, utilisation du remplissage classique: {exc}")

            # chosen_texture = textures[2]
            # 3. Préparer la zone de destination

            # 4. Créer la transformation inverse pour le pavage rotatif
            # La rotation est centrée sur le centre de la zone à remplir (w/2, h/2)
            # On utilise l'angle positif car c'est une map inverse (destination -> source)
            M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1)

            # 5. Appliquer la transformation
            # warpAffine va remplir la zone (w,h) en utilisant la texture 'chosen_texture'
            # et en la répétant grâce à BORDER_WRAP pour un pavage parfait.
            textured_roi = cv2.warpAffine(
                chosen_texture,
                M_inv,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )

            # 6. Appliquer le masque pour ne garder que la forme du contour
            mask = np.zeros((h, w), dtype=np.uint8)
            # On déplace le contour pour qu'il soit à l'origine (0,0) du masque
            cv2.drawContours(mask, [dilatatedcnt - (x, y)], -1, 255, thickness=cv2.FILLED)

            textured_region = cv2.bitwise_and(textured_roi, textured_roi, mask=mask)

            # 7. Fusionner avec l'image finale
            roi_bg = textured_image[y:y + h, x:x + w]
            roi_bg_masked = cv2.bitwise_and(roi_bg, roi_bg, mask=cv2.bitwise_not(mask))
            textured_image[y:y + h, x:x + w] = cv2.add(roi_bg_masked, textured_region)

    cv2.imwrite(path + "_combined.jpg", textured_image)
    cv2.imwrite(path + "_background_quadrilaterals.jpg", background_with_quads)
    cv2.imwrite(path + "_background_wallquad_quadrilaterals.jpg", background_with_wallquad_quads)

    # Dessin des zones colorées pour visualisation
    for cnt in toto:
        random_color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.drawContours(color_zones, [cnt], -1, random_color, thickness=cv2.FILLED)

    return color_zones, myedgesdilatated


# --- SCRIPT PRINCIPAL ---
path = 'building7.jpg'
img = cv2.imread(path)
if img is None:
    print(f"Erreur: Impossible de charger l'image depuis {path}")
else:
    gray = boostimagegray(img)
    edges = cv2.Canny(gray, 1, 150)

    color_zones1, myedges1 = drawFile(path, img, edges, 4, cv2.RETR_CCOMP)
    cv2.imwrite(path + "_result_CCOMP_color_zones.jpg", color_zones1)