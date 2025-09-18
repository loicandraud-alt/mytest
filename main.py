import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque

from PIL import Image, ImageFilter

from webercolor.linequedrilareral import quadrilateral_from_lines


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
        segments.append((length, dx, dy, pt1, pt2))

    segments.sort(key=lambda seg: seg[0], reverse=True)

    angle_deg1 = -10
    angle_deg2 = -10


    for length, dx, dy, pt1, pt2 in segments:
        # To improve, if vertical lnes
        if abs(dx) < 50:
            continue
        if angle_deg1 < 0:
            angle_deg1 = math.degrees(math.atan2(dy, dx)) % 180
            outpt11 =  pt1
            outpt12 = pt2
        else:
            if angle_deg2 < 0:
                angle_deg2 = math.degrees(math.atan2(dy, dx)) % 180
                outpt21 = pt1
                outpt22 = pt2


    return outpt11, outpt12, outpt21, outpt22

def drawFile(path, image, edges, dilatation, mode):
    kernel = np.ones((dilatation, dilatation), np.uint8)
    myedgesdilatated = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(path + "_result_edges.jpg", myedgesdilatated)
    color_zones = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    toto = floodfill_extract_contours(myedgesdilatated)
    final_result = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

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

    for cnt in toto:
        # ---- NOUVELLE LOGIQUE DE TEXTURAGE ----

        # 1. Obtenir l'angle et prendre son opposé pour la correction
        angle = findAngle2(cnt)
        pt11, pt12, pt21, pt22 = findPointsFromContour(cnt)
        quadrilateral_from_lines()
        # 2. Choisir une texture (on ne la tourne pas ici)
        chosen_texture = random.choice(textures)
        #chosen_texture = textures[2]
        # 3. Préparer la zone de destination
        dilatatedcnt = dilate_contour(cnt, img.shape, 3)
        x, y, w, h = cv2.boundingRect(dilatatedcnt)

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
        roi_bg = final_result[y:y + h, x:x + w]
        roi_bg_masked = cv2.bitwise_and(roi_bg, roi_bg, mask=cv2.bitwise_not(mask))
        final_result[y:y + h, x:x + w] = cv2.add(roi_bg_masked, textured_region)

    final_composite = mergeimages(img, final_result)
    cv2.imwrite(path + "_combined.jpg", final_composite)

    # Dessin des zones colorées pour visualisation
    for cnt in toto:
        random_color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.drawContours(color_zones, [cnt], -1, random_color, thickness=cv2.FILLED)

    return color_zones, myedgesdilatated


# --- SCRIPT PRINCIPAL ---
path = 'building5.jpg'
img = cv2.imread(path)
if img is None:
    print(f"Erreur: Impossible de charger l'image depuis {path}")
else:
    gray = boostimagegray(img)
    edges = cv2.Canny(gray, 1, 150)

    color_zones1, myedges1 = drawFile(path, img, edges, 4, cv2.RETR_CCOMP)
    cv2.imwrite(path + "_result_CCOMP_color_zones.jpg", color_zones1)