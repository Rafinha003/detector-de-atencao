import cv2
import numpy as np
import math


MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nariz
    (0.0, -63.6, -12.5),   # Queixo
    (-43.3, 32.7, -26.0),  # Olho esquerdo
    (43.3, 32.7, -26.0),   # Olho direito
    (-28.9, -28.9, -24.1), # Boca esquerda
    (28.9, -28.9, -24.1)   # Boca direita
])

# Índices MediaPipe
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def get_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in LANDMARK_IDS
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    pitch, yaw, roll = angles
    return pitch, yaw
