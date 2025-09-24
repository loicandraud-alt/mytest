
import cv2
import numpy as np
import math

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

    for diff_cnt in diff_contours:
        if len(diff_cnt) < 3:
            continue

        rect = cv2.minAreaRect(diff_cnt)
        (cx, cy), (w, h), angle = rect
        area = float(w) * float(h)
        if area < min_area:
            continue

        box = cv2.boxPoints(rect)
        box = np.asarray(box, dtype=np.float32)
        box += offset.astype(np.float32)

        offset_vec = offset.reshape(1, 1, 2)
        diff_cnt_global = diff_cnt + offset_vec.astype(diff_cnt.dtype)
        diff_cnt_global = diff_cnt_global.astype(np.int32)

        candidate = {
            "box": box,
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
    return enriched_candidate
