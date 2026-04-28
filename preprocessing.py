"""
preprocessing.py
----------------
Image preprocessing pipeline for receipt OCR.
Handles: noise removal, blur, skew correction, contrast/lighting.
"""

import cv2
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate
import imutils
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        steps = [
            ("resize",    self._resize_if_small),
            ("grayscale", self._to_grayscale),
            ("denoise",   self._remove_noise),
            ("deskew",    self._deskew),
            ("contrast",  self._enhance_contrast),
            ("binarize",  self._binarize),
            ("dilation",  self._dilate_text),
        ]

        img = image.copy()
        for name, fn in steps:
            try:
                img = fn(img)
                logger.debug("Step '%s' completed — shape: %s", name, img.shape)
            except Exception as exc:
                logger.warning("Step '%s' failed (%s), continuing.", name, exc)

        return img

    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    def _resize_if_small(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        img = cv2.medianBlur(img, 3)
        img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        angle = self._compute_skew_angle(img)
        if abs(angle) < 0.5:
            return img
        logger.debug("Correcting skew: %.2f degrees", angle)
        return imutils.rotate_bound(img, -angle)

    def _compute_skew_angle(self, img: np.ndarray) -> float:
        try:
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            best_score, best_angle = -1, 0
            for angle in np.arange(-10, 10, 0.5):
                rotated = ndimage_rotate(binary, angle, reshape=False, order=0)
                histogram = np.sum(rotated, axis=1)
                score = np.var(histogram)
                if score > best_score:
                    best_score = score
                    best_angle = angle
            return best_angle
        except Exception as exc:
            logger.warning("Skew detection failed: %s", exc)
            return 0.0

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10
        )

    def _dilate_text(self, img: np.ndarray) -> np.ndarray:
        kernel = np.ones((1, 1), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)