"""
main.py — Buy or Wait? Financial Decision Agent
Entry point: python3 code/main.py

Pipeline for each request:
  1. Load all data (once)
  2. Extract image amounts via VLM (once, cached)
  3. Resolve messages via LLM (once per user, cached)
  4. Build canonical financial state (recurrence inference)
  5. Run 90-day simulator → max safe amount + earliest safe date
  6. Generate candidate payment plans
  7. Apply safety gate → official ranking → best plan
  8. Generate explanation
  9. Write output.csv row
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# Ensure code/ is on path when run as python3 code/main.py
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATASET_DIR,
    FORECAST_DAYS,
    OUTPUT_CSV,
    REQUESTS_CSV,
    SAMPLE_REQUESTS_CSV,
)
from data_loader import DataStore, PaymentOption, RequestBundle
from image_extractor import apply_image_amounts, load_image_amounts
from message_resolver import get_or_resolve
from financial_state import build_financial_state
from cashflow import simulate, ScheduledPayment
from affordability import earliest_safe_date, max_safe_amount
from strategy import (
    PaymentPlan,
    derive_status,
    format_spending_changes,
    select_best_plan,
)
from explainer import generate_explanation
from usage_tracker import UsageTracker

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def format_payment_plan(payments: list[tuple[date, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(
        f"{d}:{round(a, 2)}" for d, a in sorted(payments, key=lambda x: x[0])
    )


def process_request(
    bundle: RequestBundle,
    image_amounts: dict[str, float],
    tracker: UsageTracker,
) -> dict:
    """Process one request. Returns a dict of output column values."""

    rid = bundle.request_id
    uid = bundle.user_id
    request_date = bundle.request_date
    requested_amount = bundle.requested_amount
    deadline = bundle.desired_completion_date
    allows_partial = bundle.allows_partial_payment
    profile = bundle.profile

    if profile is None:
        print(f"  [WARN] No profile for {uid}, skipping {rid}")
        return _empty_row(rid, requested_amount)

    # ── Step 1: Resolve messages ──────────────────────────────────────────────
    msg_context = get_or_resolve(uid, bundle.messages, usage_tracker=tracker)

    # ── Step 2: Apply image amounts + build financial state ───────────────────
    all_events = bundle.events[:]  # copy so we don't mutate the store
    apply_image_amounts(all_events, image_amounts)

    state = build_financial_state(
        profile=profile,
        events=all_events,
        request_date=request_date,
        image_amounts=image_amounts,
        msg_context=msg_context,
    )

    # ── Step 3: Max safe amount (binary search) ───────────────────────────────
    safe_amount = max_safe_amount(state, request_date, requested_amount)

    # ── Step 4: Earliest safe date for full payment ───────────────────────────
    earliest = earliest_safe_date(state, request_date, requested_amount, deadline)
    # Also find without deadline constraint for the output field
    earliest_uncapped = earliest_safe_date(state, request_date, requested_amount, None)

    # ── Step 5: Select best payment plan ─────────────────────────────────────
    best_plan = select_best_plan(
        state=state,
        requested_amount=requested_amount,
        amount_safe_today=safe_amount,
        earliest_date=earliest,
        request_date=request_date,
        deadline=deadline,
        profile=profile,
        payment_options=bundle.payment_options,
        allows_partial=allows_partial,
    )

    # ── Step 6: Derive status ─────────────────────────────────────────────────
    status = derive_status(best_plan, requested_amount, safe_amount, request_date)

    # ── Step 7: Format outputs ────────────────────────────────────────────────
    spending_changes_str = format_spending_changes(best_plan.spending_cuts)
    payment_plan_str = format_payment_plan(best_plan.payments)

    earliest_str = str(earliest_uncapped) if earliest_uncapped else ""

    # ── Step 8: Generate explanation ──────────────────────────────────────────
    explanation = generate_explanation(
        balance=state.balance,
        currency=profile.home_currency,
        min_balance=state.min_balance,
        requested_amount=requested_amount,
        safe_amount=safe_amount,
        best_plan=best_plan,
        earliest_date=earliest_uncapped,
        spending_changes_str=spending_changes_str,
        usage_tracker=tracker,
        request_id=rid,
    )

    return {
        "request_id": rid,
        "amount_safe_to_pay": round(safe_amount, 2),
        "affordability_status": status,
        "recommended_payment_method": best_plan.method,
        "payment_plan": payment_plan_str,
        "earliest_date_for_full_payment": earliest_str,
        "spending_changes_needed": spending_changes_str,
        "decision_explanation": explanation,
    }


def _empty_row(request_id: str, requested_amount: float) -> dict:
    return {
        "request_id": request_id,
        "amount_safe_to_pay": 0.0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": "Insufficient data to evaluate this request.",
    }


def run(requests_csv=None, output_path=None, verbose=True):
    """Main pipeline. Runs all requests and writes output.csv."""

    output_path = output_path or OUTPUT_CSV
    tracker = UsageTracker()

    print("=" * 60)
    print("Buy or Wait? — Financial Decision Agent")
    print("=" * 60)

    # Load all data once
    store = DataStore(requests_csv)

    # Extract image amounts once (cached)
    print("\nExtracting image amounts...")
    image_amounts = load_image_amounts(store.images)

    # Process all requests
    print(f"\nProcessing {len(store.requests_df)} requests...\n")
    rows = []
    errors = []

    for i, bundle in enumerate(store.iter_bundles(), 1):
        rid = bundle.request_id
        if verbose:
            print(f"[{i:3d}/{len(store.requests_df)}] {rid} ({bundle.user_id}) "
                  f"— {bundle.request_type} {bundle.requested_amount} {bundle.profile.home_currency if bundle.profile else '?'}")
        try:
            row = process_request(bundle, image_amounts, tracker)
            rows.append(row)
            if verbose:
                print(f"         -> {row['affordability_status']} | "
                      f"{row['recommended_payment_method']} | "
                      f"safe={row['amount_safe_to_pay']}")
        except Exception as e:
            print(f"  [ERROR] {rid}: {e}")
            errors.append(rid)
            rows.append(_empty_row(rid, bundle.requested_amount))

    # Write output.csv
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Output written: {output_path} ({len(rows)} rows)")
    if errors:
        print(f"Errors: {len(errors)} requests failed — {errors[:5]}")

    # Write usage report
    tracker.write_report()
    print(f"Total LLM calls: {tracker.total_calls()} | "
          f"Est. cost: ${tracker.total_cost():.4f}")
    print("=" * 60)

    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Buy or Wait? Financial Agent")
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on sample_requests.csv instead of full dataset"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: repo root output.csv)"
    )
    args = parser.parse_args()

    requests_file = SAMPLE_REQUESTS_CSV if args.sample else REQUESTS_CSV
    output_file = args.output or (OUTPUT_CSV if not args.sample else
                                   Path(OUTPUT_CSV).parent / "output_sample.csv")

    run(requests_csv=requests_file, output_path=output_file)
