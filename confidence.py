"""
confidence.py
-------------
3-layer confidence scoring for extracted receipt fields.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DATE_VALIDATORS = [
    re.compile(r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$"),
    re.compile(r"^\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}$"),
    re.compile(r"^\d{1,2}\s+\w+\s+\d{2,4}$"),
    re.compile(r"^\w+\s+\d{1,2},?\s+\d{2,4}$"),
]

PRICE_VALIDATOR    = re.compile(r"^\d{1,8}(\.\d{1,2})?$")
STORE_NAME_INVALID = re.compile(r"^\d+$|^[^\w\s]+$")


@dataclass
class FieldScore:
    ocr_score:       float = 0.0
    pattern_score:   float = 0.0
    heuristic_score: float = 0.0
    final_score:     float = 0.0
    flagged:         bool  = False

    WEIGHTS = {"ocr": 0.35, "pattern": 0.40, "heuristic": 0.25}

    def compute_final(self) -> float:
        w = self.WEIGHTS
        self.final_score = (
            w["ocr"]       * self.ocr_score +
            w["pattern"]   * self.pattern_score +
            w["heuristic"] * self.heuristic_score
        )
        self.final_score = round(min(self.final_score, 1.0), 4)
        return self.final_score

    def to_dict(self) -> dict:
        return {
            "ocr_score":       round(self.ocr_score, 3),
            "pattern_score":   round(self.pattern_score, 3),
            "heuristic_score": round(self.heuristic_score, 3),
            "final_score":     round(self.final_score, 3),
            "flagged":         self.flagged,
        }


class ConfidenceScorer:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def score_receipt(self, extracted, ocr_result) -> Dict[str, FieldScore]:
        word_conf_map = self._build_word_conf_map(ocr_result)
        scores: Dict[str, FieldScore] = {}
        scores["store_name"]   = self._score_store(extracted.store_name.value, word_conf_map, ocr_result.full_text)
        scores["date"]         = self._score_date(extracted.date.value, word_conf_map)
        scores["total_amount"] = self._score_total(extracted.total_amount.value, word_conf_map, ocr_result.full_text)
        for name, fs in scores.items():
            fs.compute_final()
            fs.flagged = fs.final_score < self.threshold
        return scores

    def reliability_report(self, scores: Dict[str, FieldScore]) -> dict:
        flagged = [name for name, fs in scores.items() if fs.flagged]
        overall = float(np.mean([fs.final_score for fs in scores.values()])) if scores else 0.0
        return {
            "overall_reliability": round(overall, 3),
            "flagged_fields":      flagged,
            "reliable":            len(flagged) == 0,
            "field_scores":        {name: fs.to_dict() for name, fs in scores.items()},
        }

    def _score_store(self, value, word_conf, full_text) -> FieldScore:
        fs = FieldScore()
        if not value:
            return fs
        fs.ocr_score = self._avg_conf(value, word_conf)
        if STORE_NAME_INVALID.match(value):
            fs.pattern_score = 0.2
        elif len(value.split()) >= 1 and len(value) >= 3:
            fs.pattern_score = 0.8 if (value.isupper() or value.istitle()) else 0.6
        else:
            fs.pattern_score = 0.4
        top_lines = "\n".join(full_text.splitlines()[:4])
        fs.heuristic_score = 0.9 if value in top_lines else 0.5
        return fs

    def _score_date(self, value, word_conf) -> FieldScore:
        fs = FieldScore()
        if not value:
            return fs
        fs.ocr_score     = self._avg_conf(value, word_conf)
        matched          = any(p.match(value.strip()) for p in DATE_VALIDATORS)
        fs.pattern_score = 0.95 if matched else 0.3
        has_structure    = bool(re.search(r"\d+[\-\/\.]\d+", value))
        fs.heuristic_score = 0.85 if has_structure else 0.4
        return fs

    def _score_total(self, value, word_conf, full_text) -> FieldScore:
        fs = FieldScore()
        if not value:
            return fs
        fs.ocr_score = self._avg_conf(value, word_conf)
        clean = str(value).replace(",", "").strip()
        fs.pattern_score = 0.95 if PRICE_VALIDATOR.match(clean) else 0.2
        lines = full_text.lower().splitlines()
        near_keyword = any(
            "total" in line or "amount" in line or "payable" in line
            for line in lines if value in line or clean in line
        )
        fs.heuristic_score = 0.90 if near_keyword else 0.45
        return fs

    def _build_word_conf_map(self, ocr_result) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for word in ocr_result.words:
            if word.text:
                acc.setdefault(word.text, []).append(word.confidence)
        return {w: float(np.mean(confs)) for w, confs in acc.items()}

    def _avg_conf(self, text: str, word_conf: dict) -> float:
        words = text.split()
        confs = [word_conf.get(w, 0.65) for w in words]
        return float(np.mean(confs)) if confs else 0.65