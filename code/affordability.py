"""
affordability.py — Two key calculations:

1. max_safe_amount: largest X such that paying X today keeps balance >= min_balance
   for all 90 days (binary search, confirmed income only)

2. earliest_safe_date: first date when paying the full requested_amount is safe
   (day-by-day scan, confirmed income only)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from config import BINARY_SEARCH_ITERATIONS, FORECAST_DAYS
from cashflow import ScheduledPayment, simulate
from financial_state import CanonicalFinancialState


def max_safe_amount(
    state: CanonicalFinancialState,
    request_date: date,
    requested_amount: float,
) -> float:
    """
    Binary search: largest amount safe to pay on request_date.
    Returns value in [0, requested_amount], rounded to 2 decimal places.
    """
    # Quick check: even paying 0 is unsafe (shouldn't happen, but guard)
    baseline = simulate(state, [], [], request_date)
    if not baseline.is_safe:
        return 0.0

    # Quick check: can we pay the full amount?
    full_result = simulate(
        state,
        [ScheduledPayment(request_date, requested_amount)],
        [],
        request_date,
    )
    if full_result.is_safe:
        return round(requested_amount, 2)

    # Binary search
    lo, hi = 0.0, requested_amount
    for _ in range(BINARY_SEARCH_ITERATIONS):
        if hi - lo < 0.01:
            break
        mid = (lo + hi) / 2.0
        result = simulate(
            state,
            [ScheduledPayment(request_date, mid)],
            [],
            request_date,
        )
        if result.is_safe:
            lo = mid
        else:
            hi = mid

    return round(lo, 2)


def earliest_safe_date(
    state: CanonicalFinancialState,
    request_date: date,
    requested_amount: float,
    deadline: Optional[date] = None,
) -> Optional[date]:
    """
    Scan day-by-day: first date when paying requested_amount in full is safe.
    Returns None if no date found within 90-day forecast.
    Optionally capped at deadline.
    """
    end = request_date + timedelta(days=FORECAST_DAYS)
    if deadline:
        end = min(end, deadline + timedelta(days=1))  # inclusive deadline

    for offset in range((end - request_date).days + 1):
        test_date = request_date + timedelta(days=offset)
        result = simulate(
            state,
            [ScheduledPayment(test_date, requested_amount)],
            [],
            request_date,
        )
        if result.is_safe:
            return test_date

    return None


def max_safe_amount_with_cuts(
    state: CanonicalFinancialState,
    request_date: date,
    requested_amount: float,
    cuts: list,
) -> float:
    """
    Binary search for max safe amount when spending cuts are applied.
    """
    baseline = simulate(state, [], cuts, request_date)
    if not baseline.is_safe:
        return 0.0

    full_result = simulate(
        state,
        [ScheduledPayment(request_date, requested_amount)],
        cuts,
        request_date,
    )
    if full_result.is_safe:
        return round(requested_amount, 2)

    lo, hi = 0.0, requested_amount
    for _ in range(BINARY_SEARCH_ITERATIONS):
        if hi - lo < 0.01:
            break
        mid = (lo + hi) / 2.0
        result = simulate(
            state,
            [ScheduledPayment(request_date, mid)],
            cuts,
            request_date,
        )
        if result.is_safe:
            lo = mid
        else:
            hi = mid

    return round(lo, 2)
