"""
extractor.py
------------
Extracts structured receipt fields from raw OCR text.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{2,4})\b",
]

PRICE_PATTERN = re.compile(
    r"(?:USD|INR|Rs\.?|₹|\$|€|£|AED)?\s*([\d,]+\.?\d{0,2})",
    re.IGNORECASE
)

TOTAL_KEYWORDS = [
    "total", "grand total", "amount due", "balance due", "net total",
    "subtotal", "sub-total", "amount payable", "net amount", "total amount",
    "payable", "please pay", "amount", "bill total",
]

ITEM_LINE_PATTERN = re.compile(
    r"^(.+?)\s+(?:USD|INR|Rs\.?|₹|\$|€|£)?\s*([\d,]+\.\d{2})\s*$",
    re.IGNORECASE | re.MULTILINE
)

NOISE_PATTERNS = [
    re.compile(r"[-=*_]{3,}"),
    re.compile(r"\|\s*\|"),
]


@dataclass
class ConfidenceField:
    value: Optional[str]
    confidence: float
    flagged: bool = False

    def to_dict(self) -> dict:
        return {"value": self.value, "confidence": round(self.confidence, 3)}


@dataclass
class ReceiptItem:
    name: str
    price: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {"name": self.name, "price": self.price}


@dataclass
class ExtractedReceipt:
    store_name:   ConfidenceField
    date:         ConfidenceField
    items:        List[ReceiptItem]
    total_amount: ConfidenceField
    overall_confidence: float
    low_confidence_fields: List[str]

    def to_simple_dict(self) -> dict:
        return {
            "store_name":   self.store_name.value,
            "date":         self.date.value,
            "items":        [i.to_dict() for i in self.items],
            "total_amount": self.total_amount.value,
        }

    def to_confidence_dict(self) -> dict:
        return {
            "store_name":            self.store_name.to_dict(),
            "date":                  self.date.to_dict(),
            "items":                 [i.to_dict() for i in self.items],
            "total_amount":          self.total_amount.to_dict(),
            "overall_confidence":    round(self.overall_confidence, 3),
            "low_confidence_fields": self.low_confidence_fields,
        }


class ReceiptExtractor:
    LOW_CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._date_regexes = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]

    def extract(self, full_text: str, word_confidences: dict = None) -> ExtractedReceipt:
        word_confidences = word_confidences or {}
        lines = self._clean_lines(full_text)

        store = self._extract_store_name(lines, word_confidences)
        date  = self._extract_date(lines, word_confidences)
        items = self._extract_items(lines)
        total = self._extract_total(lines, word_confidences)

        low_fields = []
        for name, fld in [("store_name", store), ("date", date), ("total_amount", total)]:
            if fld.confidence < self.LOW_CONFIDENCE_THRESHOLD:
                fld.flagged = True
                low_fields.append(name)

        field_confs = [store.confidence, date.confidence, total.confidence]
        item_confs  = [i.confidence for i in items] if items else [0.5]
        overall     = float(np.mean(field_confs + item_confs))

        return ExtractedReceipt(
            store_name=store, date=date, items=items,
            total_amount=total, overall_confidence=overall,
            low_confidence_fields=low_fields,
        )

    def _extract_store_name(self, lines, word_conf) -> ConfidenceField:
        candidates = []
        for i, line in enumerate(lines[:6]):
            line = line.strip()
            if len(line) < 3 or PRICE_PATTERN.search(line):
                continue
            pos_score  = 1.0 - (i / 10)
            case_score = 0.3 if (line.isupper() or line.istitle()) else 0.1
            len_score  = min(len(line) / 30, 0.3)
            ocr_score  = self._avg_word_conf(line, word_conf)
            score = pos_score + case_score + len_score + ocr_score * 0.3
            candidates.append((line, min(score, 1.0)))

        if not candidates:
            return ConfidenceField(value=None, confidence=0.0)

        candidates.sort(key=lambda x: x[1], reverse=True)
        value, score = candidates[0]
        return ConfidenceField(value=value, confidence=round(min(score, 0.99), 3))

    def _extract_date(self, lines, word_conf) -> ConfidenceField:
        full_text = "\n".join(lines)
        for pattern in self._date_regexes:
            m = pattern.search(full_text)
            if m:
                date_str   = m.group(1)
                ocr_score  = self._avg_word_conf(date_str, word_conf)
                confidence = 0.75 + ocr_score * 0.25
                return ConfidenceField(value=date_str, confidence=round(confidence, 3))
        return ConfidenceField(value=None, confidence=0.0)

    def _extract_total(self, lines, word_conf) -> ConfidenceField:
        candidates = []
        for line in lines:
            line_lower = line.lower()
            keyword_score = 0.0
            for kw in TOTAL_KEYWORDS:
                if kw in line_lower:
                    keyword_score = max(keyword_score, len(kw) / 15)
            if keyword_score == 0.0:
                continue
            price_m = PRICE_PATTERN.search(line)
            if not price_m:
                continue
            raw_price = price_m.group(1).replace(",", "")
            try:
                float(raw_price)
            except ValueError:
                continue
            ocr_score  = self._avg_word_conf(line, word_conf)
            confidence = 0.5 + keyword_score * 0.3 + ocr_score * 0.2
            candidates.append((raw_price, round(min(confidence, 0.99), 3)))

        if not candidates:
            all_prices = PRICE_PATTERN.findall("\n".join(lines))
            if all_prices:
                return ConfidenceField(value=all_prices[-1].replace(",", ""), confidence=0.35)
            return ConfidenceField(value=None, confidence=0.0)

        candidates.sort(key=lambda x: x[1], reverse=True)
        value, conf = candidates[0]
        return ConfidenceField(value=value, confidence=conf)

    def _extract_items(self, lines) -> List[ReceiptItem]:
        items = []
        in_item_section = False
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in ["item", "description", "qty", "price", "product"]):
                in_item_section = True
                continue
            if any(kw in line_lower for kw in TOTAL_KEYWORDS):
                break
            m = ITEM_LINE_PATTERN.match(line.strip())
            if m:
                name  = m.group(1).strip()
                price = m.group(2).strip()
                if name and price and len(name) > 1:
                    conf = 0.8 if in_item_section else 0.6
                    items.append(ReceiptItem(name=name, price=price, confidence=conf))
                    in_item_section = True
        return items

    def _clean_lines(self, text: str) -> List[str]:
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            for pat in NOISE_PATTERNS:
                line = pat.sub("", line)
            line = line.strip()
            if line:
                cleaned.append(line)
        return cleaned

    def _avg_word_conf(self, text: str, word_conf: dict) -> float:
        words = text.split()
        confs = [word_conf.get(w, 0.7) for w in words]
        return float(np.mean(confs)) if confs else 0.7