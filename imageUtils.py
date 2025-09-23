
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math



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
