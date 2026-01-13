import numpy as np


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x * image_w, lm.y * image_h]))

    A = euclidean(points[1], points[5])
    B = euclidean(points[2], points[4])
    C = euclidean(points[0], points[3])

    ear = (A + B) / (2.0 * C)
    return ear
