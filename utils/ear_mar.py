"""
EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) calculations.

Why geometry-based?
- Works with glasses (landmarks still tracked on face mesh)
- Works in low light (MediaPipe is robust; CLAHE further helps)
- No pixel color analysis → no dependence on lighting
- Works with partial occlusion

EAR Formula (Soukupova & Cech, 2016):
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    where p1..p6 are 6 eye landmarks (top, bottom, corners)

    When eye is OPEN  → EAR ≈ 0.25–0.35
    When eye is CLOSED → EAR < 0.20

MAR Formula (similar principle for mouth):
    MAR = vertical opening / horizontal width
    When mouth CLOSED → MAR ≈ 0.1–0.3
    When YAWNING      → MAR > 0.6
"""

import numpy as np


def euclidean(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def landmark_to_point(lm, w, h):
    """Convert normalized MediaPipe landmark to pixel coordinates."""
    return (int(lm.x * w), int(lm.y * h))


def compute_EAR(landmarks, eye_indices, frame_w, frame_h):
    """
    Compute Eye Aspect Ratio for a given eye.

    eye_indices: list of 6 landmark indices in order:
        [left_corner, top_inner, top_outer, right_corner, bot_outer, bot_inner]
        For MediaPipe Face Mesh:
          LEFT_EYE  = [362, 385, 387, 263, 373, 380]
          RIGHT_EYE = [33,  160, 158, 133, 153, 144]

    Returns:
        float: EAR value (higher = more open)
    """
    pts = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in eye_indices]

    # Horizontal distance: corner to corner
    horizontal = euclidean(pts[0], pts[3])

    # Two vertical distances (top-bottom pairs)
    vert1 = euclidean(pts[1], pts[5])
    vert2 = euclidean(pts[2], pts[4])

    if horizontal < 1e-6:
        return 0.0

    ear = (vert1 + vert2) / (2.0 * horizontal)
    return ear


def compute_MAR(landmarks, mouth_indices, frame_w, frame_h):
    """
    Compute Mouth Aspect Ratio.

    mouth_indices: list of 8 landmark indices:
        [left_corner, top_left, top_center, top_right,
         right_corner, bot_right, bot_center, bot_left]
        For MediaPipe: [61, 39, 0, 269, 291, 405, 17, 181]

    Returns:
        float: MAR value (higher = more open / yawning)
    """
    pts = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in mouth_indices]

    # Horizontal: left corner to right corner
    horizontal = euclidean(pts[0], pts[4])

    # Three vertical measurements for robustness
    vert1 = euclidean(pts[1], pts[7])   # inner top-left  ↔ inner bot-left
    vert2 = euclidean(pts[2], pts[6])   # top center      ↔ bottom center
    vert3 = euclidean(pts[3], pts[5])   # inner top-right ↔ inner bot-right

    if horizontal < 1e-6:
        return 0.0

    mar = (vert1 + vert2 + vert3) / (3.0 * horizontal)
    return mar


def compute_EAR_with_debug(landmarks, eye_indices, frame_w, frame_h):
    """
    Same as compute_EAR but also returns the landmark points
    for visualization / debugging purposes.
    """
    pts = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in eye_indices]
    horizontal = euclidean(pts[0], pts[3])
    vert1 = euclidean(pts[1], pts[5])
    vert2 = euclidean(pts[2], pts[4])
    ear = (vert1 + vert2) / (2.0 * horizontal) if horizontal > 1e-6 else 0.0
    return ear, pts


def compute_MAR_with_debug(landmarks, mouth_indices, frame_w, frame_h):
    """
    Same as compute_MAR but returns landmark points for debugging.
    """
    pts = [landmark_to_point(landmarks[i], frame_w, frame_h) for i in mouth_indices]
    horizontal = euclidean(pts[0], pts[4])
    vert1 = euclidean(pts[1], pts[7])
    vert2 = euclidean(pts[2], pts[6])
    vert3 = euclidean(pts[3], pts[5])
    mar = (vert1 + vert2 + vert3) / (3.0 * horizontal) if horizontal > 1e-6 else 0.0
    return mar, pts
