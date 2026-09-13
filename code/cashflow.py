"""
cashflow.py — 90-day deterministic cash-flow simulator.

Core mathematical engine. The only place where financial safety is computed.

B_t = B_{t-1} + ConfirmedIncome_t - Expenses_t - ExtraPayments_t

Safety: min_t(B_t) >= minimum_balance_to_keep
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from config import FORECAST_DAYS
from financial_state import CanonicalFinancialState, CashFlowItem


@dataclass
class ScheduledPayment:
    """A payment from the strategy under test."""
    payment_date: date
    amount: float


@dataclass
class SpendingCut:
    """A reduction or stop of a flexible recurring expense."""
    event_id: str
    original_amount: float
    new_amount: float          # 0.0 = stop completely
    cut_type: str              # "stop" or "reduce_to"
    category: str
    flexibility: str


@dataclass
class SimulationResult:
    daily_balances: list[float]
    min_balance_reached: float
    safety_margin: float          # min_balance_reached - minimum_balance_to_keep
    is_safe: bool
    days_simulated: int


def simulate(
    state: CanonicalFinancialState,
    extra_payments: list[ScheduledPayment],
    spending_cuts: list[SpendingCut],
    start_date: date,
    days: int = FORECAST_DAYS,
) -> SimulationResult:
    """
    Simulate cash flow for `days` days starting from start_date.

    Rules:
    - start_date is always the request_date (never datetime.now())
    - Only confirmed income (is_confirmed=True) is added to balance
    - Pending credits are excluded
    - Cancelled/failed/unrealized events are already excluded from state
    - Extra payments are strategy payments under test
    - Spending cuts reduce or eliminate recurring flexible expenses
    """
    min_balance = state.min_balance
    balance = state.balance

    # Build spending cut lookup: event_id prefix → new_amount
    # Note: inferred event IDs are prefixed with "inferred_{original_id}_N"
    cut_lookup: dict[str, float] = {}
    for cut in spending_cuts:
        cut_lookup[cut.event_id] = cut.new_amount
        # Also cover inferred occurrences of this event
        cut_lookup[f"inferred_{cut.event_id}"] = cut.new_amount

    # Build daily index for cash flow items
    # expense_by_date: date → list of (amount, event_id)
    # income_by_date: date → list of amount
    expense_by_date: dict[date, list[tuple[float, str]]] = {}
    income_by_date: dict[date, list[float]] = {}

    for item in state.cash_flow_items:
        if item.event_date < start_date or item.event_date >= start_date + timedelta(days=days):
            continue
        if item.is_income:
            if item.is_confirmed:
                income_by_date.setdefault(item.event_date, []).append(item.amount)
        else:
            expense_by_date.setdefault(item.event_date, []).append(
                (item.amount, item.event_id)
            )

    # Index extra payments
    payment_by_date: dict[date, float] = {}
    for pmt in extra_payments:
        payment_by_date[pmt.payment_date] = (
            payment_by_date.get(pmt.payment_date, 0.0) + pmt.amount
        )

    daily_balances = []
    min_bal_reached = balance

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)

        # Add confirmed income
        for amt in income_by_date.get(current_date, []):
            balance += amt

        # Subtract expenses (applying spending cuts)
        for (amt, eid) in expense_by_date.get(current_date, []):
            # Check if cut applies (exact match or inferred prefix match)
            effective_amt = amt
            matched_cut = None
            if eid in cut_lookup:
                matched_cut = cut_lookup[eid]
            else:
                # Check for inferred event (format: "inferred_{original_id}_{N}")
                for cut_key, cut_val in cut_lookup.items():
                    if eid.startswith(f"inferred_{cut_key}_"):
                        matched_cut = cut_val
                        break

            if matched_cut is not None:
                effective_amt = matched_cut  # 0.0 = stopped

            balance -= effective_amt

        # Subtract strategy payments
        balance -= payment_by_date.get(current_date, 0.0)

        daily_balances.append(round(balance, 2))
        if balance < min_bal_reached:
            min_bal_reached = balance

    safety_margin = min_bal_reached - min_balance

    return SimulationResult(
        daily_balances=daily_balances,
        min_balance_reached=round(min_bal_reached, 2),
        safety_margin=round(safety_margin, 2),
        is_safe=safety_margin >= 0,
        days_simulated=days,
    )


def simulate_no_purchase(
    state: CanonicalFinancialState,
    start_date: date,
    days: int = FORECAST_DAYS,
) -> SimulationResult:
    """Simulate without any purchase — baseline for understanding future balance."""
    return simulate(state, [], [], start_date, days)
