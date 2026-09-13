"""
usage_tracker.py — Records LLM/VLM API usage and generates usage_report.md.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import USAGE_REPORT

# Cost per 1M tokens (approximate, Gemini Flash pricing)
COST_PER_M_INPUT = 0.075
COST_PER_M_OUTPUT = 0.30


@dataclass
class UsageRecord:
    request_id: str
    model: str
    call_type: str       # "message_resolution", "image_extraction", "explanation"
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = field(init=False)

    def __post_init__(self):
        self.cost_usd = (
            self.input_tokens / 1_000_000 * COST_PER_M_INPUT
            + self.output_tokens / 1_000_000 * COST_PER_M_OUTPUT
        )


class UsageTracker:
    def __init__(self):
        self.records: list[UsageRecord] = []
        self._start_time: Optional[float] = None

    def start_timer(self):
        self._start_time = time.time()

    def record(
        self,
        request_id: str,
        model: str,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: Optional[float] = None,
    ):
        if latency_ms is None and self._start_time is not None:
            latency_ms = (time.time() - self._start_time) * 1000
            self._start_time = None
        self.records.append(UsageRecord(
            request_id=request_id,
            model=model,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms or 0.0,
        ))

    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def total_calls(self) -> int:
        return len(self.records)

    def write_report(self):
        """Generate evaluation/usage_report.md from actual run data."""
        USAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)

        n = max(self.total_calls(), 1)
        n_requests = 250  # full dataset

        by_model: dict[str, dict] = {}
        for r in self.records:
            if r.model not in by_model:
                by_model[r.model] = {"calls": 0, "in": 0, "out": 0, "cost": 0.0}
            by_model[r.model]["calls"] += 1
            by_model[r.model]["in"] += r.input_tokens
            by_model[r.model]["out"] += r.output_tokens
            by_model[r.model]["cost"] += r.cost_usd

        lines = [
            "# Token Usage Report",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total model calls | {self.total_calls()} |",
            f"| Total input tokens | {self.total_input_tokens():,} |",
            f"| Total output tokens | {self.total_output_tokens():,} |",
            f"| Total tokens | {self.total_input_tokens() + self.total_output_tokens():,} |",
            f"| Avg input tokens/request | {self.total_input_tokens() // n_requests:,} |",
            f"| Avg output tokens/request | {self.total_output_tokens() // n_requests:,} |",
            f"| Estimated total cost (USD) | ${self.total_cost():.4f} |",
            f"| Avg cost/request (USD) | ${self.total_cost() / n_requests:.4f} |",
            "",
            "## Per-Model Breakdown",
            "",
            "| Model | Calls | Input Tokens | Output Tokens | Cost (USD) |",
            "|-------|-------|--------------|---------------|------------|",
        ]
        for model, stats in by_model.items():
            lines.append(
                f"| {model} | {stats['calls']} | {stats['in']:,} | "
                f"{stats['out']:,} | ${stats['cost']:.4f} |"
            )

        lines += [
            "",
            "## Call Type Breakdown",
            "",
            "| Call Type | Count |",
            "|-----------|-------|",
        ]
        by_type: dict[str, int] = {}
        for r in self.records:
            by_type[r.call_type] = by_type.get(r.call_type, 0) + 1
        for ct, count in sorted(by_type.items()):
            lines.append(f"| {ct} | {count} |")

        lines += [
            "",
            "---",
            "*Report generated automatically from the final full-dataset run.*",
        ]

        with open(USAGE_REPORT, "w") as f:
            f.write("\n".join(lines))

        print(f"[usage_tracker] Report written to {USAGE_REPORT}")
