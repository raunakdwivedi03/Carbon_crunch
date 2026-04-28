"""
ocr_engine.py
-------------
Text detection and recognition layer.
Supports Tesseract OCR and EasyOCR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRWord:
    text: str
    confidence: float
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def is_empty(self) -> bool:
        return self.text.strip() == ""


@dataclass
class OCRLine:
    words: List[OCRWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words if not w.is_empty)

    @property
    def avg_confidence(self) -> float:
        valid = [w.confidence for w in self.words if not w.is_empty]
        return float(np.mean(valid)) if valid else 0.0


@dataclass
class OCRResult:
    words: List[OCRWord]
    lines: List[OCRLine]
    full_text: str
    engine: str
    avg_confidence: float

    @classmethod
    def empty(cls, engine: str = "unknown") -> "OCRResult":
        return cls(words=[], lines=[], full_text="", engine=engine, avg_confidence=0.0)


class TesseractEngine:
    def __init__(self, lang: str = "eng", psm: int = 6):
        self.lang = lang
        self.psm = psm
        self._check_installed()

    def _check_installed(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._pytesseract = pytesseract
        except Exception as exc:
            raise RuntimeError("Tesseract is not installed or not on PATH.") from exc

    def run(self, image: np.ndarray) -> OCRResult:
        config = f"--oem 3 --psm {self.psm}"
        data = self._pytesseract.image_to_data(
            image, lang=self.lang, config=config,
            output_type=self._pytesseract.Output.DICT,
        )

        words: List[OCRWord] = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            raw_conf = data["conf"][i]
            conf = max(0.0, float(raw_conf)) / 100.0
            words.append(OCRWord(
                text=text, confidence=conf,
                left=data["left"][i], top=data["top"][i],
                width=data["width"][i], height=data["height"][i],
            ))

        lines = self._group_into_lines(words)
        full_text = self._pytesseract.image_to_string(image, lang=self.lang, config=config)
        avg_conf = float(np.mean([w.confidence for w in words])) if words else 0.0

        return OCRResult(words=words, lines=lines, full_text=full_text,
                         engine="tesseract", avg_confidence=avg_conf)

    def _group_into_lines(self, words: List[OCRWord], tolerance: int = 8) -> List[OCRLine]:
        if not words:
            return []
        buckets: dict = {}
        for w in words:
            cy = w.top + w.height // 2
            matched = None
            for key in buckets:
                if abs(key - cy) <= tolerance:
                    matched = key
                    break
            if matched is None:
                buckets[cy] = [w]
            else:
                buckets[matched].append(w)
        lines = []
        for key in sorted(buckets):
            row = sorted(buckets[key], key=lambda w: w.left)
            lines.append(OCRLine(words=row))
        return lines


class EasyOCREngine:
    def __init__(self, lang: List[str] = None, gpu: bool = False):
        self.lang = lang or ["en"]
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            logger.info("Initialising EasyOCR reader...")
            self._reader = easyocr.Reader(self.lang, gpu=self.gpu)
        return self._reader

    def run(self, image: np.ndarray) -> OCRResult:
        reader = self._get_reader()
        raw = reader.readtext(image, detail=1, paragraph=False)

        words: List[OCRWord] = []
        line_texts: List[str] = []

        for (bbox, text, conf) in raw:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            left, top = int(min(xs)), int(min(ys))
            width  = int(max(xs) - min(xs))
            height = int(max(ys) - min(ys))
            words.append(OCRWord(text=text, confidence=float(conf),
                                 left=left, top=top, width=width, height=height))
            line_texts.append(text)

        lines = [OCRLine(words=[w]) for w in words]
        full_text = "\n".join(line_texts)
        avg_conf = float(np.mean([w.confidence for w in words])) if words else 0.0

        return OCRResult(words=words, lines=lines, full_text=full_text,
                         engine="easyocr", avg_confidence=avg_conf)


class OCREngine:
    ENGINES = {"tesseract": TesseractEngine, "easyocr": EasyOCREngine}

    def __init__(self, engine_name: str = "tesseract", **engine_kwargs):
        self.engine_name = engine_name
        self._engine = self._init_engine(engine_name, engine_kwargs)

    def _init_engine(self, name: str, kwargs: dict):
        cls = self.ENGINES.get(name)
        if cls is None:
            raise ValueError(f"Unknown engine '{name}'.")
        try:
            return cls(**kwargs)
        except Exception as exc:
            logger.warning("Could not initialise '%s': %s — trying fallback.", name, exc)
            fallback = "easyocr" if name == "tesseract" else "tesseract"
            return self.ENGINES[fallback]()

    def run(self, image: np.ndarray) -> OCRResult:
        if image is None or image.size == 0:
            return OCRResult.empty(self.engine_name)
        try:
            return self._engine.run(image)
        except Exception as exc:
            logger.exception("OCR failed: %s", exc)
            return OCRResult.empty(self.engine_name)