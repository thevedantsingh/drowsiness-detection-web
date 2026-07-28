"""
Driver Face Selector

When multiple faces are detected (e.g., driver + passenger),
we need to reliably select the DRIVER's face only.

Strategy:
1. Compute bounding box area for each detected face.
2. The driver in a dashcam/interior camera is typically:
   - The LARGEST face (closest to camera)
   - AND the most centred or left-biased face (RHD: right-hand drive)
   
   We use a combined score:
       score = area_weight * normalized_area
             + position_weight * horizontal_position_score

   This is configurable for LHD (left-hand drive) vs RHD (right-hand drive) cars.
"""

import numpy as np


def get_face_bbox(face_landmarks, frame_w, frame_h):
    """Get bounding box of a face from its landmarks."""
    xs = [lm.x * frame_w for lm in face_landmarks.landmark]
    ys = [lm.y * frame_h for lm in face_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    area = (x_max - x_min) * (y_max - y_min)
    cx = (x_min + x_max) / 2   # horizontal center of face
    cy = (y_min + y_max) / 2
    return x_min, y_min, x_max, y_max, area, cx, cy


def select_driver_face(
    multi_face_landmarks,
    frame_shape,
    drive_side: str = "right",   # "right" = RHD (India, UK) | "left" = LHD (USA, Europe)
    area_weight: float = 0.6,
    position_weight: float = 0.4
):
    """
    Select the driver's face from multiple detected faces.

    Args:
        multi_face_landmarks : list of face landmark objects from MediaPipe
        frame_shape          : (height, width, channels) of the frame
        drive_side           : "right" for right-hand drive, "left" for left-hand drive
        area_weight          : weight for face size (0–1)
        position_weight      : weight for horizontal position (0–1)

    Returns:
        The face landmarks object belonging to the driver.
    """
    if len(multi_face_landmarks) == 1:
        return multi_face_landmarks[0]

    frame_h, frame_w = frame_shape[:2]
    scores = []

    for face in multi_face_landmarks:
        _, _, _, _, area, cx, _ = get_face_bbox(face, frame_w, frame_h)

        # Normalize area to [0, 1] range relative to full frame
        normalized_area = area / (frame_w * frame_h)

        # Position score: for RHD cars the driver is on the RIGHT side of the frame
        # (camera looks at driver from front/dashboard perspective)
        # For LHD cars the driver is on the LEFT side
        normalized_cx = cx / frame_w   # 0 = leftmost, 1 = rightmost

        if drive_side == "right":
            # Driver is on the right → reward higher cx
            position_score = normalized_cx
        else:
            # Driver is on the left → reward lower cx
            position_score = 1.0 - normalized_cx

        score = area_weight * normalized_area + position_weight * position_score
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return multi_face_landmarks[best_idx]


def get_all_face_boxes(multi_face_landmarks, frame_shape):
    """
    Return bounding boxes for all detected faces (for visualization).
    Returns list of (x_min, y_min, x_max, y_max) tuples in pixel coords.
    """
    frame_h, frame_w = frame_shape[:2]
    boxes = []
    for face in multi_face_landmarks:
        x_min, y_min, x_max, y_max, _, _, _ = get_face_bbox(face, frame_w, frame_h)
        boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return boxes
