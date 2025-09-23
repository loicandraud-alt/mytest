import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque

from PIL import Image, ImageFilter

from typing import List, Tuple
from perpspectiveoverlay import project_texture
from webercolor.contourUtils import checkContoursIndide, contour_area, build_contour_mask, dilate_contour, \
    findPointsFromContour, findPointsFromContour2
from webercolor.imageUtils import floodfill_extract_contours, boostimagegray
from webercolor.quadri import quadrilateral_from_lines, quadrilateral_from_lines2


def maxlength(approx):
    max_len = 0.0
    for i in range(len(approx)):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % len(approx)][0]  # boucle fermée
        length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if length > max_len:
            max_len = length
    return max_len






def findlines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    lines_img = np.zeros_like(img)

    # Dessiner les lignes
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rouge
    return lines_img







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








def drawFile(path, image, edges, dilatation, mode):
    kernel = np.ones((dilatation, dilatation), np.uint8)
    myedgesdilatated = cv2.dilate(edges, kernel, iterations=1)
    cv2.imwrite(path + "2_result_edges.jpg", myedgesdilatated)
    color_zones = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    toto = floodfill_extract_contours(myedgesdilatated)
    textured_image = image.copy()
    background_with_quads = image.copy()
    points_overlay = image.copy()
    approx_overlay = image.copy()

    # Charger et préparer les textures
    textures_files = {
        "brique": cv2.resize(cv2.imread("../webercolor/textures/brique.jpg"), (0, 0), fx=0.5, fy=0.5),
        #"brique2": cv2.resize(cv2.imread("../webercolor/textures/brique2.jpg"), (0, 0), fx=0.2, fy=0.2),
        "brique3": cv2.resize(cv2.imread("../webercolor/textures/brique3.jpg"), (0, 0), fx=0.2, fy=0.2),
        #"bois1": cv2.resize(cv2.imread("../webercolor/textures/bois1.jpg"), (0, 0), fx=0.5, fy=0.5),
        #"bois2": cv2.resize(cv2.imread("../webercolor/textures/bois2.jpg"), (0, 0), fx=0.2, fy=0.2),
        #"pierre": cv2.resize(cv2.imread("../webercolor/textures/pierre.jpg"), (0, 0), fx=0.1, fy=0.1),
        #"enduit1": cv2.resize(cv2.imread("../webercolor/textures/enduit1.jpg"), (0, 0), fx=0.05, fy=0.05),
    }
    textures = list(textures_files.values())
    checkContoursIndide(toto)
    done = 0

    for cnt in toto:
        if done > 100:
            continue
        done = done + 1

        area = contour_area(cnt)
        if (area < 10000):
            print(f"Contour bypassed because too small, {area}")
            continue

        #1. Obtenir l'angle et prendre son opposé pour la correction␊
        #angle = findAngle2(cnt)
        points, approx = findPointsFromContour(cnt)
        quadrilateral_points = None
        if approx is not None and len(approx) >= 2:
            cv2.polylines(approx_overlay, [approx], isClosed=True, color=(0, 255, 0), thickness=3)
        if points is not None:
            pt11, pt12, pt21, pt22 = points
            for pt in (pt11, pt12, pt21, pt22):
                cv2.circle(points_overlay, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), thickness=-1)
            try:
                line1 = ((float(pt11[0]), float(pt11[1])), (float(pt12[0]), float(pt12[1])))
                line2 = ((float(pt21[0]), float(pt21[1])), (float(pt22[0]), float(pt22[1])))
                quadrilateral_points = quadrilateral_from_lines2(line1, line2)
                contour_x = cnt[:, 0, 0]
                min_x = float(np.min(contour_x))
                max_x = float(np.max(contour_x))
                quadrilateral_points = quadrilateral_from_lines2(
                    line1,
                    line2,
                    x_bounds=(min_x, max_x),
                )
            except ValueError:
                quadrilateral_points = None
            if quadrilateral_points:
                quad_array = np.array([[int(round(x)), int(round(y))] for x, y in quadrilateral_points],          dtype=np.int32)
                cv2.polylines(background_with_quads, [quad_array], isClosed=True, color=(0, 0, 255), thickness=3)
            wall_mask = build_contour_mask(cnt, toto, image.shape)


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
                        clip_mask = wall_mask,
                    )
                    continue
                except Exception as exc:
                    print(f"Projection perspective échouée, utilisation du remplissage classique: {exc}")
            # 3. Préparer la zone de destination

            # 4. Créer la transformation inverse pour le pavage rotatif
            # La rotation est centrée sur le centre de la zone à remplir (w/2, h/2)
            # On utilise l'angle positif car c'est une map inverse (destination -> source)
            M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1)

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

    cv2.imwrite(path + "6_combined.jpg", textured_image)
    cv2.imwrite(path + "5_background_quadrilaterals.jpg", background_with_quads)
    cv2.imwrite(path + "4_points_debug.jpg", points_overlay)
    cv2.imwrite(path + "3_approx_debug.jpg", approx_overlay)

    # Dessin des zones colorées pour visualisation
    for cnt in toto:
        random_color = tuple(np.random.randint(0, 256, size=3).tolist())
        cv2.drawContours(color_zones, [cnt], -1, random_color, thickness=cv2.FILLED)

    return color_zones, myedgesdilatated


# --- SCRIPT PRINCIPAL ---
#mypath = 'building10.jpg'
paths = ['building10.jpg','building9.jpg','building7.jpg','building5.jpg','building4.jpg','building2.jpg']
#img = cv2.imread(mypath)
#if img is None:
#    print(f"Erreur: Impossible de charger l'image depuis {mypath}")
#else:
#    gray = boostimagegray(img)
 #   edges = cv2.Canny(gray, 1, 150)#

#    color_zones1, myedges1 = drawFile(mypath, img, edges, 4, cv2.RETR_CCOMP)
#    cv2.imwrite(mypath + "_result_CCOMP_color_zones.jpg", color_zones1)

for path in paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image depuis {path}")
    else:
        gray = boostimagegray(img)
        edges = cv2.Canny(gray, 1, 150)

        color_zones1, myedges1 = drawFile(path, img, edges, 4, cv2.RETR_CCOMP)
        cv2.imwrite(path + "_result_1CCOMP_color_zones.jpg", color_zones1)