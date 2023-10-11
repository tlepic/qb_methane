import cv2
import numpy as np

def rotate_image(image, angle):
    """
    Effectue une rotation d'image d'un angle donné.

    Args:
        image (numpy.ndarray): L'image à faire pivoter.
        angle (float): Angle de rotation en degrés.

    Returns:
        numpy.ndarray: Image pivotée.
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def translate_image(image, tx, ty):
    """
    Effectue une translation d'une partie de l'image.

    Args:
        image (numpy.ndarray): L'image à traduire.
        tx (int): Décalage horizontal.
        ty (int): Décalage vertical.

    Returns:
        numpy.ndarray: Image avec la translation appliquée.
    """
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image