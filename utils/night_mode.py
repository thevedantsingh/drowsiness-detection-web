"""
Night Mode / Low-Light Enhancement

Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to
dramatically improve face visibility in dark conditions.

Why CLAHE instead of simple brightness boost?
- Simple brightness boost clips highlights and crushes shadows.
- CLAHE enhances contrast locally → preserves details in both dark and bright regions.
- MediaPipe can then detect landmarks much more reliably on the enhanced frame.
"""

import cv2
import numpy as np


# Pre-build CLAHE object once (reuse across frames for efficiency)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_frame(frame: np.ndarray, strength: str = "auto") -> np.ndarray:
    """
    Enhance a low-light BGR frame using CLAHE on the luminance channel.

    Args:
        frame    : BGR image (numpy array)
        strength : "auto"   → detect brightness and apply if needed
                   "always" → always apply CLAHE
                   "off"    → return frame unchanged (for testing)

    Returns:
        Enhanced BGR image.
    """
    if strength == "off":
        return frame

    if strength == "auto":
        # Only enhance if average brightness is low
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 100:   # Already bright enough
            return frame

    # Convert BGR → LAB (L=luminance, A/B=color channels)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE only to the L (luminance) channel
    l_enhanced = _clahe.apply(l)

    # Merge back and convert to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def denoise_frame(frame: np.ndarray, strength: int = 5) -> np.ndarray:
    """
    Optional denoising pass for very grainy/dark footage.
    Slower than CLAHE — use only if needed.

    Args:
        frame    : BGR image
        strength : filter strength (3–10, higher = more blur)
    """
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def get_brightness(frame: np.ndarray) -> float:
    """Return mean brightness of frame (0–255)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
