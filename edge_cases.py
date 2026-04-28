"""
edge_cases.py
-------------
Edge case detection and handling for the OCR pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EdgeCaseType(Enum):
    BLANK_IMAGE          = auto()
    LOW_RESOLUTION       = auto()
    EXTREME_BLUR         = auto()
    VERY_DARK_IMAGE      = auto()
    VERY_BRIGHT_IMAGE    = auto()
    PARTIAL_RECEIPT      = auto()
    NO_TEXT_DETECTED     = auto()
    MINIMAL_TEXT         = auto()
    UNUSUAL_ASPECT_RATIO = auto()


@dataclass
class EdgeCaseReport:
    detected: List[EdgeCaseType] = field(default_factory=list)
    warnings: List[str]          = field(default_factory=list)
    is_processable: bool         = True
    suggested_actions: List[str] = field(default_factory=list)

    def add(self, case: EdgeCaseType, warning: str, action: str = ""):
        self.detected.append(case)
        self.warnings.append(warning)
        if action:
            self.suggested_actions.append(action)

    def to_dict(self) -> dict:
        return {
            "detected_cases":    [c.name for c in self.detected],
            "warnings":          self.warnings,
            "is_processable":    self.is_processable,
            "suggested_actions": self.suggested_actions,
        }


class EdgeCaseHandler:
    MIN_WIDTH        = 100
    MIN_HEIGHT       = 100
    BLUR_THRESHOLD   = 30.0
    DARK_THRESHOLD   = 40
    BRIGHT_THRESHOLD = 230
    MIN_TEXT_LINES   = 3
    UNUSUAL_AR_MIN   = 0.2
    UNUSUAL_AR_MAX   = 5.0

    def analyse_image(self, image: np.ndarray) -> EdgeCaseReport:
        report = EdgeCaseReport()
        if image is None or image.size == 0:
            report.add(EdgeCaseType.BLANK_IMAGE, "Image is empty or failed to load.", "Verify the file path.")
            report.is_processable = False
            return report

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_pixel = float(np.mean(gray))
        std_pixel  = float(np.std(gray))

        if std_pixel < 5:
            report.add(EdgeCaseType.BLANK_IMAGE, f"Image appears blank (std={std_pixel:.1f}).", "Check the file.")
            report.is_processable = False
            return report

        if w < self.MIN_WIDTH or h < self.MIN_HEIGHT:
            report.add(EdgeCaseType.LOW_RESOLUTION, f"Image too small ({w}x{h}px).", "Capture at higher resolution.")
            report.is_processable = False

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.BLUR_THRESHOLD:
            report.add(EdgeCaseType.EXTREME_BLUR, f"Very blurry (var={laplacian_var:.1f}).", "Re-capture with better focus.")

        if mean_pixel < self.DARK_THRESHOLD:
            report.add(EdgeCaseType.VERY_DARK_IMAGE, f"Very dark (mean={mean_pixel:.0f}).", "Increase brightness.")
        elif mean_pixel > self.BRIGHT_THRESHOLD:
            report.add(EdgeCaseType.VERY_BRIGHT_IMAGE, f"Overexposed (mean={mean_pixel:.0f}).", "Reduce exposure.")

        ar = w / max(h, 1)
        if ar < self.UNUSUAL_AR_MIN or ar > self.UNUSUAL_AR_MAX:
            report.add(EdgeCaseType.UNUSUAL_ASPECT_RATIO, f"Unusual aspect ratio (AR={ar:.2f}).", "Crop or rotate image.")

        return report

    def analyse_ocr_output(self, ocr_result, report: EdgeCaseReport) -> EdgeCaseReport:
        full_text  = ocr_result.full_text.strip()
        word_count = len(full_text.split())
        line_count = len([l for l in full_text.splitlines() if l.strip()])

        if word_count == 0:
            report.add(EdgeCaseType.NO_TEXT_DETECTED, "No text extracted.", "Try a different OCR engine.")
            report.is_processable = False
        elif line_count < self.MIN_TEXT_LINES:
            report.add(EdgeCaseType.MINIMAL_TEXT, f"Only {line_count} line(s) detected.", "Ensure full receipt is visible.")

        return report

    def try_recover(self, image: np.ndarray, report: EdgeCaseReport) -> np.ndarray:
        if not report.is_processable:
            return image
        recovered = image.copy()
        if EdgeCaseType.VERY_DARK_IMAGE in report.detected:
            recovered = cv2.convertScaleAbs(recovered, alpha=1.5, beta=40)
        if EdgeCaseType.VERY_BRIGHT_IMAGE in report.detected:
            recovered = cv2.convertScaleAbs(recovered, alpha=0.7, beta=-30)
        if EdgeCaseType.EXTREME_BLUR in report.detected:
            gaussian  = cv2.GaussianBlur(recovered, (9, 9), 10.0)
            recovered = cv2.addWeighted(recovered, 1.5, gaussian, -0.5, 0)
        return recovered