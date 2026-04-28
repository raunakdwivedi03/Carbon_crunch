#!/usr/bin/env python3
"""
demo.py
-------
Generates synthetic receipt images and runs the full pipeline on them.
Use this to verify the pipeline works without the actual dataset.

Usage:  python demo.py
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("demo")


# ---------------------------------------------------------------------------
# Synthetic receipt generator
# ---------------------------------------------------------------------------

SAMPLE_RECEIPTS = [
    {
        "store":  "WALMART SUPERCENTER",
        "date":   "04/18/2024",
        "items":  [
            ("Organic Milk 1L",    "3.49"),
            ("Bread Whole Wheat",  "2.99"),
            ("Eggs 12-Pack",       "4.79"),
            ("Orange Juice 2L",    "5.49"),
            ("Butter Unsalted",    "3.29"),
        ],
        "total":  "19.95",
    },
    {
        "store":  "TARGET STORE #452",
        "date":   "03/22/2024",
        "items":  [
            ("Shampoo 500ml",      "6.99"),
            ("Toothpaste Twin Pk", "4.49"),
            ("Soap Bar 3-Pack",    "3.99"),
        ],
        "total":  "15.47",
    },
    {
        "store":  "WHOLE FOODS MARKET",
        "date":   "02/10/2024",
        "items":  [
            ("Avocado x3",         "4.50"),
            ("Almond Butter 16oz", "9.99"),
            ("Greek Yogurt 32oz",  "6.79"),
            ("Kombucha 16oz",      "3.99"),
        ],
        "total":  "25.27",
    },
]


def draw_receipt(receipt_data: dict, width: int = 600) -> np.ndarray:
    """
    Render a synthetic receipt as a white image with black text.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    line_height = 30
    padding     = 40
    total_lines = 6 + len(receipt_data["items"]) * 1 + 4
    height      = total_lines * line_height + padding * 2

    img = np.ones((height, width), dtype=np.uint8) * 255   # White background

    def put(text, y, scale=0.6, thickness=1, bold=False):
        f = font_bold if bold else font
        x = padding
        cv2.putText(img, text, (x, y), f, scale, 0, thickness, cv2.LINE_AA)

    y = padding + 30
    put(receipt_data["store"], y, scale=0.8, thickness=2, bold=True); y += 35
    put("-" * 55, y, scale=0.5); y += 20
    put(f"Date: {receipt_data['date']}", y); y += 30
    put("-" * 55, y, scale=0.5); y += 20
    put("Item                          Price", y, bold=True); y += 25
    put("-" * 55, y, scale=0.5); y += 20

    for name, price in receipt_data["items"]:
        line = f"{name:<30} {price:>8}"
        put(line, y); y += line_height

    put("-" * 55, y, scale=0.5); y += 20
    put(f"TOTAL                         {receipt_data['total']:>8}", y, scale=0.75, thickness=2, bold=True)
    y += 35
    put("Thank you for shopping!", y, scale=0.5)

    # Convert to BGR for saving
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def add_noise(img: np.ndarray, level: str = "low") -> np.ndarray:
    """Add realistic-ish noise to a synthetic receipt."""
    result = img.copy()
    if level == "low":
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif level == "high":
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # Add slight blur
        result = cv2.GaussianBlur(result, (3, 3), 0)
    return result


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    receipts_dir = Path("sample_receipts")
    receipts_dir.mkdir(exist_ok=True)

    logger.info("Generating %d synthetic receipt images...", len(SAMPLE_RECEIPTS))
    image_paths = []

    for i, receipt_data in enumerate(SAMPLE_RECEIPTS):
        img = draw_receipt(receipt_data)

        # Vary noise level
        noise_level = ["low", "low", "high"][i]
        img = add_noise(img, noise_level)

        # Optionally add slight skew to the third receipt
        if i == 2:
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle=2.5, scale=1.0)
            img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

        out_path = receipts_dir / f"receipt_{i+1:03d}.jpg"
        cv2.imwrite(str(out_path), img)
        image_paths.append(out_path)
        logger.info("  Created: %s", out_path)

    # Run pipeline
    logger.info("\nRunning OCR pipeline on synthetic receipts...")
    from src.pipeline import Pipeline

    pipeline = Pipeline(engine="tesseract", output_dir="outputs", confidence_threshold=0.70)
    results  = pipeline.process_folder(receipts_dir)

    # Print extraction results
    print("\n" + "=" * 60)
    print("  PER-RECEIPT EXTRACTION RESULTS")
    print("=" * 60)
    for r in results:
        fname = Path(r["file"]).name
        status = r.get("status", "?")
        print(f"\n[ {fname} ]  status={status}")
        simple = r.get("simple_extraction", {})
        print(f"  Store   : {simple.get('store_name', 'N/A')}")
        print(f"  Date    : {simple.get('date', 'N/A')}")
        print(f"  Total   : {simple.get('total_amount', 'N/A')}")
        items = simple.get("items", [])
        if items:
            print(f"  Items   : {len(items)} line item(s) detected")
        conf = r.get("confidence_scores", {})
        print(f"  Reliability : {conf.get('overall_reliability', 'N/A')}")
        flagged = conf.get("flagged_fields", [])
        if flagged:
            print(f"  Low-conf  : {', '.join(flagged)}")

    # Summary
    summary = pipeline.generate_summary(results)
    print("\nJSON summary saved --> outputs/financial_summary.json")
    print("Text report saved  --> outputs/financial_summary.txt")


if __name__ == "__main__":
    main()