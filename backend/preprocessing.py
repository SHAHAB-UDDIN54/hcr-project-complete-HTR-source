"""Safe image preparation for handwriting OCR.

The recognizer is deliberately fed the original/normalized color image rather
than a contour-derived collection of 28x28 character crops. OpenCV is used
only for validation, gentle illumination normalization, border cleanup and
optional debug images.
"""
from __future__ import annotations

import io
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import DEBUG_DIR, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    h, w = image.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        raise ValueError("Image is too small. Please upload a larger image.")
    if max(h, w) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def _trim_uniform_border(image: np.ndarray) -> np.ndarray:
    """Remove only very large uniform edge bands; never crop ordinary ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    border = max(3, min(h, w) // 100)
    sample = np.concatenate([
        gray[:border, :].ravel(), gray[-border:, :].ravel(),
        gray[:, :border].ravel(), gray[:, -border:].ravel(),
    ])
    median = float(np.median(sample))
    # Detect camera bars / solid borders by comparing edge rows/columns to the
    # interior. Only remove if the band is unusually uniform.
    def uniform(v):
        return float(np.std(v)) < 8.0 and abs(float(np.mean(v)) - median) < 18

    top = bottom = left = right = 0
    for i in range(min(h // 4, 250)):
        if uniform(gray[i, :]): top = i + 1
        else: break
    for i in range(min(h // 4, 250)):
        if uniform(gray[h - 1 - i, :]): bottom = i + 1
        else: break
    for i in range(min(w // 4, 250)):
        if uniform(gray[:, i]): left = i + 1
        else: break
    for i in range(min(w // 4, 250)):
        if uniform(gray[:, w - 1 - i]): right = i + 1
        else: break
    if top + bottom >= h - 20 or left + right >= w - 20:
        return image
    return image[top:h-bottom if bottom else h, left:w-right if right else w]


def normalize_for_detection(image: np.ndarray, debug: bool = False) -> np.ndarray:
    image = _trim_uniform_border(image)
    # CLAHE is applied only to luminance, preserving the original color cues.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, "normalized.png"), normalized)
    return normalized


def prepare_image(image_bytes: bytes, debug: bool = False) -> np.ndarray:
    image = decode_image(image_bytes)
    return normalize_for_detection(image, debug=debug)
