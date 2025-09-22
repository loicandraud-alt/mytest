
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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

def filled_area_touche_haut(filled_area):
    return np.any(filled_area[0] == 255)


def filled_area_touche_bas(filled_area):
    return np.any(filled_area[-1] == 255)

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
