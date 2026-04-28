"""
summary.py
----------
Financial summary generator across all receipts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional

logger = logging.getLogger(__name__)


def _parse_amount(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        clean = str(value).replace(",", "").replace("$", "").replace("₹", "").replace("£", "").strip()
        return float(clean)
    except (ValueError, TypeError):
        return None


class FinancialSummaryGenerator:
    def generate(self, receipts: List[dict]) -> dict:
        total_spend = 0.0
        num_parsed  = 0
        num_failed  = 0
        spend_per_store: dict = defaultdict(float)
        dates: List[str] = []

        for r in receipts:
            extracted = r.get("extracted", {})
            if not extracted:
                num_failed += 1
                continue

            amount_field = extracted.get("total_amount", {})
            amount_val   = amount_field.get("value") if isinstance(amount_field, dict) else amount_field
            amount       = _parse_amount(amount_val)

            store_field  = extracted.get("store_name", {})
            store_name   = store_field.get("value") if isinstance(store_field, dict) else store_field
            store_name   = store_name or "Unknown Store"

            date_field = extracted.get("date", {})
            date_val   = date_field.get("value") if isinstance(date_field, dict) else date_field
            if date_val:
                dates.append(date_val)

            if amount is not None:
                total_spend += amount
                spend_per_store[store_name] += amount
                num_parsed += 1
            else:
                num_failed += 1

        num_transactions = len(receipts)
        avg_transaction  = (total_spend / num_parsed) if num_parsed else 0.0

        spend_per_store_sorted = dict(
            sorted(spend_per_store.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "total_spend":               round(total_spend, 2),
            "num_transactions":          num_transactions,
            "num_successfully_parsed":   num_parsed,
            "num_failed_or_missing":     num_failed,
            "average_transaction_value": round(avg_transaction, 2),
            "spend_per_store":           {k: round(v, 2) for k, v in spend_per_store_sorted.items()},
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest":   max(dates) if dates else None,
            },
            "currency_note": "All amounts normalised to numeric values (currency symbols stripped).",
        }

    def format_text_report(self, summary: dict) -> str:
        lines = [
            "=" * 50,
            "        EXPENSE SUMMARY REPORT",
            "=" * 50,
            f"Total Transactions : {summary['num_transactions']}",
            f"Successfully Parsed: {summary['num_successfully_parsed']}",
            f"Failed / Missing   : {summary['num_failed_or_missing']}",
            "",
            f"Total Spend        : {summary['total_spend']:.2f}",
            f"Avg Transaction    : {summary['average_transaction_value']:.2f}",
            "",
        ]
        date_range = summary.get("date_range", {})
        if date_range.get("earliest"):
            lines.append(f"Date Range : {date_range['earliest']}  to  {date_range['latest']}")
            lines.append("")
        lines.append("Spend Per Store")
        lines.append("-" * 50)
        store_data = summary.get("spend_per_store", {})
        if store_data:
            for store, amount in store_data.items():
                lines.append(f"  {store:<30} {amount:>10.2f}")
        else:
            lines.append("  (no store data available)")
        lines.append("=" * 50)
        return "\n".join(lines)