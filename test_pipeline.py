"""
tests/test_pipeline.py
----------------------
Unit tests for each module of the OCR pipeline.
Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# Preprocessing tests
# ===========================================================================

class TestImagePreprocessor:
    def setup_method(self):
        from src.preprocessing import ImagePreprocessor
        self.pp = ImagePreprocessor()

    def _make_image(self, h=400, w=300, color=True):
        if color:
            return np.ones((h, w, 3), dtype=np.uint8) * 200
        return np.ones((h, w), dtype=np.uint8) * 200

    def test_grayscale_conversion(self):
        img  = self._make_image(color=True)
        gray = self.pp._to_grayscale(img)
        assert len(gray.shape) == 2

    def test_resize_small_image(self):
        img     = self._make_image(h=200, w=400)
        resized = self.pp._resize_if_small(img)
        assert resized.shape[1] >= 1000

    def test_resize_large_image_unchanged(self):
        img     = self._make_image(h=800, w=1200)
        resized = self.pp._resize_if_small(img)
        assert resized.shape[1] == 1200

    def test_binarize_returns_binary(self):
        img    = self._make_image(color=False)
        binary = self.pp._binarize(img)
        unique = np.unique(binary)
        assert set(unique).issubset({0, 255})

    def test_full_preprocess_pipeline(self):
        img    = self._make_image(color=True)
        result = self.pp.preprocess(img)
        assert result is not None
        assert len(result.shape) == 2


# ===========================================================================
# Extractor tests
# ===========================================================================

SAMPLE_RECEIPT_TEXT = """
WALMART SUPERCENTER
123 Main Street, Springfield

Date: 04/18/2024  Time: 14:35

Item               Price
Milk 1L             3.49
Bread               2.99
Eggs 12pk           4.79

TOTAL              11.27
Thank you!
"""


class TestReceiptExtractor:
    def setup_method(self):
        from src.extractor import ReceiptExtractor
        self.ex = ReceiptExtractor()

    def test_extract_store_name(self):
        receipt = self.ex.extract(SAMPLE_RECEIPT_TEXT)
        assert receipt.store_name.value is not None
        assert len(receipt.store_name.value) > 2

    def test_extract_date(self):
        receipt = self.ex.extract(SAMPLE_RECEIPT_TEXT)
        assert receipt.date.value is not None
        assert "2024" in receipt.date.value or "/" in receipt.date.value

    def test_extract_total(self):
        receipt = self.ex.extract(SAMPLE_RECEIPT_TEXT)
        assert receipt.total_amount.value is not None
        val = float(receipt.total_amount.value.replace(",", ""))
        assert val > 0

    def test_extract_items(self):
        receipt = self.ex.extract(SAMPLE_RECEIPT_TEXT)
        assert isinstance(receipt.items, list)

    def test_empty_text(self):
        receipt = self.ex.extract("")
        assert receipt.store_name.value is None
        assert receipt.date.value is None

    def test_confidence_range(self):
        receipt = self.ex.extract(SAMPLE_RECEIPT_TEXT)
        for field in [receipt.store_name, receipt.date, receipt.total_amount]:
            assert 0.0 <= field.confidence <= 1.0


# ===========================================================================
# Confidence scorer tests
# ===========================================================================

class TestConfidenceScorer:
    def setup_method(self):
        from src.confidence import ConfidenceScorer
        self.scorer = ConfidenceScorer(threshold=0.70)

    def _mock_ocr_result(self):
        class MockWord:
            def __init__(self, text, conf):
                self.text = text
                self.confidence = conf

        class MockOCR:
            words = [
                MockWord("WALMART", 0.95),
                MockWord("04/18/2024", 0.92),
                MockWord("11.27", 0.90),
                MockWord("TOTAL", 0.93),
            ]
            full_text = SAMPLE_RECEIPT_TEXT
            avg_confidence = 0.925
            engine = "test"

        return MockOCR()

    def _mock_receipt(self):
        from src.extractor import ReceiptExtractor
        return ReceiptExtractor().extract(SAMPLE_RECEIPT_TEXT)

    def test_score_receipt_returns_dict(self):
        ocr    = self._mock_ocr_result()
        rx     = self._mock_receipt()
        scores = self.scorer.score_receipt(rx, ocr)
        assert isinstance(scores, dict)
        assert "store_name"   in scores
        assert "date"         in scores
        assert "total_amount" in scores

    def test_final_scores_in_range(self):
        ocr    = self._mock_ocr_result()
        rx     = self._mock_receipt()
        scores = self.scorer.score_receipt(rx, ocr)
        for name, fs in scores.items():
            assert 0.0 <= fs.final_score <= 1.0, f"{name} score out of range"

    def test_reliability_report_structure(self):
        ocr    = self._mock_ocr_result()
        rx     = self._mock_receipt()
        scores = self.scorer.score_receipt(rx, ocr)
        report = self.scorer.reliability_report(scores)
        assert "overall_reliability" in report
        assert "flagged_fields"      in report
        assert "reliable"            in report


# ===========================================================================
# Edge case handler tests
# ===========================================================================

class TestEdgeCaseHandler:
    def setup_method(self):
        from src.edge_cases import EdgeCaseHandler
        self.handler = EdgeCaseHandler()

    def _blank_image(self):
        return np.ones((400, 300, 3), dtype=np.uint8) * 250

    def _dark_image(self):
        # Noisy dark image so std > 5 (avoids blank detection)
        img = np.random.randint(5, 35, (400, 300, 3), dtype=np.uint8)
        return img

    def _tiny_image(self):
        return np.ones((50, 40, 3), dtype=np.uint8) * 200

    def test_blank_image_detected(self):
        report = self.handler.analyse_image(self._blank_image())
        assert report is not None

    def test_tiny_image_not_processable(self):
        report = self.handler.analyse_image(self._tiny_image())
        assert not report.is_processable

    def test_dark_image_detected(self):
        from src.edge_cases import EdgeCaseType
        report = self.handler.analyse_image(self._dark_image())
        assert EdgeCaseType.VERY_DARK_IMAGE in report.detected

    def test_none_image(self):
        report = self.handler.analyse_image(None)
        assert not report.is_processable


# ===========================================================================
# Financial summary tests
# ===========================================================================

MOCK_RESULTS = [
    {
        "file": "receipt_001.jpg",
        "extracted": {
            "store_name":   {"value": "WALMART", "confidence": 0.95},
            "date":         {"value": "04/18/2024", "confidence": 0.92},
            "total_amount": {"value": "19.95", "confidence": 0.97},
        }
    },
    {
        "file": "receipt_002.jpg",
        "extracted": {
            "store_name":   {"value": "TARGET", "confidence": 0.88},
            "date":         {"value": "03/22/2024", "confidence": 0.90},
            "total_amount": {"value": "15.47", "confidence": 0.93},
        }
    },
    {
        "file": "receipt_003.jpg",
        "extracted": {
            "store_name":   {"value": "WALMART", "confidence": 0.85},
            "date":         {"value": "02/10/2024", "confidence": 0.88},
            "total_amount": {"value": None, "confidence": 0.20},
        }
    },
]


class TestFinancialSummary:
    def setup_method(self):
        from src.summary import FinancialSummaryGenerator
        self.gen = FinancialSummaryGenerator()

    def test_total_spend(self):
        summary = self.gen.generate(MOCK_RESULTS)
        assert summary["total_spend"] == pytest.approx(35.42, abs=0.01)

    def test_num_transactions(self):
        summary = self.gen.generate(MOCK_RESULTS)
        assert summary["num_transactions"] == 3

    def test_spend_per_store(self):
        summary = self.gen.generate(MOCK_RESULTS)
        assert "WALMART" in summary["spend_per_store"]

    def test_missing_total_handled(self):
        summary = self.gen.generate(MOCK_RESULTS)
        assert summary["num_failed_or_missing"] >= 1

    def test_text_report_contains_total(self):
        summary = self.gen.generate(MOCK_RESULTS)
        report  = self.gen.format_text_report(summary)
        assert "35.42" in report or "Total" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])