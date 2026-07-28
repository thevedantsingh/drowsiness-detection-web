"""
Head Pose Estimation using MediaPipe Face Mesh landmarks.

Uses solvePnP with a 3D facial model to estimate:
  - Pitch  : Up/Down tilt  → positive = chin down (nodding off)
  - Yaw    : Left/Right    → positive = looking right
  - Roll   : Tilt sideways → positive = tilting right

No external 3D model file needed — uses standard facial geometry constants.
"""

import numpy as np
import cv2


# 3D facial reference points (world coordinates in mm, approx. average face)
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip          (landmark 1)
    [0.0,   -330.0, -65.0  ],   # Chin              (landmark 152)
    [-225.0, 170.0, -135.0 ],   # Left eye corner   (landmark 33)
    [225.0,  170.0, -135.0 ],   # Right eye corner  (landmark 263)
    [-150.0,-150.0, -125.0 ],   # Left mouth corner (landmark 61)
    [150.0, -150.0, -125.0 ],   # Right mouth corner(landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(landmarks, frame_w, frame_h):
    """
    Estimate head orientation from MediaPipe face landmarks.

    Args:
        landmarks : face_mesh.landmark list
        frame_w   : frame width  in pixels
        frame_h   : frame height in pixels

    Returns:
        pitch (float) : forward nod angle in degrees (+ = head down)
        yaw   (float) : left/right angle in degrees
        roll  (float) : tilt angle in degrees
    """
    image_points = np.array([
        [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
        for i in FACE_LANDMARK_IDS
    ], dtype=np.float64)

    # Camera intrinsics (estimated — focal length ≈ frame width)
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))   # Assuming no lens distortion

    success, rotation_vec, translation_vec = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector → rotation matrix → Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat([rotation_mat, translation_vec])

    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])
    roll  = float(euler_angles[2])

    return pitch, yaw, roll


def draw_pose_axes(frame, landmarks, frame_w, frame_h, length=100):
    """
    Draw 3D orientation axes on the nose tip for visual debugging.
    Red=X(right), Green=Y(up), Blue=Z(forward)
    """
    image_points = np.array([
        [landmarks[i].x * frame_w, landmarks[i].y * frame_h]
        for i in FACE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return

    axis_3d = np.float32([
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])

    nose_2d = tuple(image_points[0].astype(int))
    axis_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    cv2.line(frame, nose_2d, tuple(axis_pts[0].ravel().astype(int)), (0,   0, 255), 2)
    cv2.line(frame, nose_2d, tuple(axis_pts[1].ravel().astype(int)), (0, 255,   0), 2)
    cv2.line(frame, nose_2d, tuple(axis_pts[2].ravel().astype(int)), (255, 0,   0), 2)
