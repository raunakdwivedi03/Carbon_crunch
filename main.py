#!/usr/bin/env python3
"""
main.py
-------
Command-line interface for the AI-OCR Receipt Extraction Pipeline.

Examples
--------
# Process all receipts in a folder (auto-detect engine):
    python main.py --input receipts/ --output outputs/

# Process a single image:
    python main.py --input receipts/receipt_001.jpg --output outputs/

# Use EasyOCR instead of Tesseract:
    python main.py --input receipts/ --engine easyocr

# Set custom confidence threshold:
    python main.py --input receipts/ --threshold 0.65

# Verbose logging:
    python main.py --input receipts/ -v
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import Pipeline, SUPPORTED_EXTENSIONS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr-pipeline",
        description="AI-OCR Receipt Extraction Pipeline (Carbon Crunch Assignment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to a receipt image or a folder of receipt images."
    )
    p.add_argument(
        "--output", "-o", default="outputs",
        help="Directory for JSON outputs and summary (default: outputs/)."
    )
    p.add_argument(
        "--engine", "-e", choices=["tesseract", "easyocr"], default="tesseract",
        help="OCR engine to use (default: tesseract)."
    )
    p.add_argument(
        "--threshold", "-t", type=float, default=0.70,
        help="Confidence threshold for flagging low-confidence fields (default: 0.70)."
    )
    p.add_argument(
        "--no-summary", action="store_true",
        help="Skip generating the financial summary."
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging."
    )
    return p


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("main")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    # Initialise pipeline
    pipeline = Pipeline(
        engine=args.engine,
        output_dir=args.output,
        confidence_threshold=args.threshold,
    )

    # Process
    if input_path.is_dir():
        results = pipeline.process_folder(input_path)
    elif input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        result  = pipeline.process_image(input_path)
        results = [result]
        # Print single-image result nicely
        print("\n" + "=" * 60)
        print("  EXTRACTION RESULT")
        print("=" * 60)
        print(json.dumps(result.get("simple_extraction", {}), indent=2))
        print("\nConfidence Scores:")
        conf = result.get("confidence_scores", {})
        print(f"  Overall reliability : {conf.get('overall_reliability', 'N/A')}")
        print(f"  Flagged fields      : {conf.get('flagged_fields', [])}")
        print("=" * 60)
    else:
        logger.error("Unsupported file type: %s", input_path.suffix)
        logger.info("Supported types: %s", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(1)

    if not results:
        logger.warning("No images were processed.")
        sys.exit(0)

    # Summary
    if not args.no_summary:
        pipeline.generate_summary(results)

    logger.info("Done. Outputs written to: %s", args.output)


if __name__ == "__main__":
    main()