"""
pipeline.py
-----------
Main OCR pipeline orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Union

import cv2
from tqdm import tqdm

from .preprocessing import ImagePreprocessor
from .ocr_engine    import OCREngine
from .extractor     import ReceiptExtractor
from .confidence    import ConfidenceScorer
from .edge_cases    import EdgeCaseHandler
from .summary       import FinancialSummaryGenerator

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class Pipeline:
    def __init__(
        self,
        engine:               str   = "tesseract",
        output_dir:           Union[str, Path] = "outputs",
        confidence_threshold: float = 0.70,
    ):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine   = OCREngine(engine_name=engine)
        self.extractor    = ReceiptExtractor()
        self.scorer       = ConfidenceScorer(threshold=confidence_threshold)
        self.edge_handler = EdgeCaseHandler()
        self.summariser   = FinancialSummaryGenerator()
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path: Union[str, Path]) -> dict:
        image_path = Path(image_path)
        result: dict = {"file": str(image_path), "status": "ok"}

        try:
            raw_image = self.preprocessor.load_image(str(image_path))
        except FileNotFoundError as exc:
            return {**result, "status": "error", "error": str(exc)}

        edge_report = self.edge_handler.analyse_image(raw_image)
        result["edge_cases"] = edge_report.to_dict()

        if not edge_report.is_processable:
            return {**result, "status": "unprocessable"}

        recovered    = self.edge_handler.try_recover(raw_image, edge_report)

        try:
            preprocessed = self.preprocessor.preprocess(recovered)
        except Exception as exc:
            logger.exception("Preprocessing failed: %s", exc)
            preprocessed = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY)

        ocr_result = self.ocr_engine.run(preprocessed)
        result["raw_text"]          = ocr_result.full_text
        result["ocr_engine"]        = ocr_result.engine
        result["ocr_avg_confidence"] = round(ocr_result.avg_confidence, 3)

        edge_report = self.edge_handler.analyse_ocr_output(ocr_result, edge_report)
        result["edge_cases"] = edge_report.to_dict()

        if not edge_report.is_processable:
            return {**result, "status": "no_text"}

        word_conf_map = {w.text: w.confidence for w in ocr_result.words}
        receipt       = self.extractor.extract(ocr_result.full_text, word_conf_map)
        scores        = self.scorer.score_receipt(receipt, ocr_result)
        reliability   = self.scorer.reliability_report(scores)

        result["extracted"]         = receipt.to_confidence_dict()
        result["confidence_scores"] = reliability
        result["simple_extraction"] = receipt.to_simple_dict()

        return result

    def process_folder(self, folder: Union[str, Path]) -> List[dict]:
        folder = Path(folder)
        images = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS])

        if not images:
            logger.warning("No supported images found in: %s", folder)
            return []

        logger.info("Processing %d image(s) from: %s", len(images), folder)
        results = []

        for img_path in tqdm(images, desc="Processing receipts", unit="img"):
            result = self.process_image(img_path)
            results.append(result)
            self._save_receipt_json(result, img_path.stem)

        return results

    def generate_summary(self, results: List[dict]) -> dict:
        summary = self.summariser.generate(results)
        report  = self.summariser.format_text_report(summary)

        summary_path = self.output_dir / "financial_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        report_path = self.output_dir / "financial_summary.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(report)
        return summary

    def _save_receipt_json(self, result: dict, stem: str):
        out_path = self.output_dir / f"{stem}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error("Could not save JSON for %s: %s", stem, exc)