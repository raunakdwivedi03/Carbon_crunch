"""
AI-OCR Receipt Extraction Pipeline
====================================
Carbon Crunch Shortlisting Assignment
"""

from .pipeline import Pipeline
from .preprocessing import ImagePreprocessor
from .ocr_engine import OCREngine, OCRResult
from .extractor import ReceiptExtractor, ExtractedReceipt
from .confidence import ConfidenceScorer
from .edge_cases import EdgeCaseHandler
from .summary import FinancialSummaryGenerator

__all__ = [
    "Pipeline",
    "ImagePreprocessor",
    "OCREngine",
    "OCRResult",
    "ReceiptExtractor",
    "ExtractedReceipt",
    "ConfidenceScorer",
    "EdgeCaseHandler",
    "FinancialSummaryGenerator",
]