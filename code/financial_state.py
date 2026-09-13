"""
financial_state.py — Builds the canonical financial state for a user.

Key responsibilities:
1. Apply image-extracted amounts to events with blank amounts
2. Apply message amendments (salary changes, cancellations, amount corrections)
3. Infer recurring income and expense schedules from historical patterns
4. Produce a clean list of dated cash-flow events for the 90-day simulator

Critical rule: only CONFIRMED income participates in the safety simulation.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from config import (
    AMOUNT_CV_THRESHOLD,
    MIN_RECURRENCE_COUNT,
    SALARY_DAY_VARIANCE,
)
from data_loader import FinancialEvent, UserProfile
from message_resolver import ResolvedMessageContext

# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class CashFlowItem:
    """A single dated cash-flow event for the simulator."""
    event_date: date
    amount: float              # Always positive; direction encoded in is_income
    is_income: bool
    category: str
    event_id: str
    flexibility: str           # fixed, stoppable, reducible, reducible_or_stoppable
    minimum_allowed_amount: Optional[float]
    is_confirmed: bool         # False = probable/uncertain (never used in safety calc)
    is_essential: bool         # Essential if category is protected by user
    description: str


@dataclass
class CanonicalFinancialState:
    balance: float
    min_balance: float
    home_currency: str
    cash_flow_items: list[CashFlowItem]
    # Pending debits (already in items but tracked separately for clarity)
    pending_debits: list[CashFlowItem]
    # Flexible expense events the user can cut
    flexible_items: list[CashFlowItem]


# ── Statuses to EXCLUDE from cash-flow simulation ────────────────────────────
# Per problem statement: ignore pending credits, failed, cancelled, unrealized, investment_*
EXCLUDE_STATUSES = {"cancelled", "failed", "unrealized"}
EXCLUDE_EVENT_TYPES = {"investment_purchase", "investment_valuation", "investment_sale"}
EXCLUDE_DIRECTIONS = {"non_cash"}

# Statuses to include in future cash flow
INCLUDE_FUTURE_STATUSES = {"pending", "scheduled"}


def _is_flexible(ev: FinancialEvent) -> bool:
    return ev.flexibility in ("stoppable", "reducible", "reducible_or_stoppable")


def _is_expense_direction(ev: FinancialEvent) -> bool:
    return ev.direction == "debit"


def _is_income_direction(ev: FinancialEvent) -> bool:
    return ev.direction == "credit"


def _median_amount(amounts: list[float]) -> float:
    return statistics.median(amounts) if amounts else 0.0


def _coeff_variation(amounts: list[float]) -> float:
    if len(amounts) < 2:
        return 0.0
    mean = statistics.mean(amounts)
    if mean == 0:
        return 0.0
    return statistics.stdev(amounts) / mean


# ── Recurrence inference ──────────────────────────────────────────────────────

def _infer_next_date(last_date: date, day_of_month: int) -> date:
    """Return the next month's occurrence of day_of_month after last_date."""
    import calendar
    year, month = last_date.year, last_date.month
    month += 1
    if month > 12:
        month = 1
        year += 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day_of_month, max_day))


def _advance_date_by_days(last_date: date, interval_days: int) -> date:
    return last_date + timedelta(days=interval_days)


def _infer_recurring_expenses(
    settled_events: list[FinancialEvent],
    request_date: date,
    forecast_end: date,
    profile: UserProfile,
) -> list[CashFlowItem]:
    """
    Infer recurring expense schedule from historical settled expenses.

    Handles two patterns:
    - Monthly expenses: rent, utilities, subscriptions (25-40 day interval)
    - Weekly/frequent expenses: groceries, transport (5-20 day interval)
      For weekly expenses we project the TOTAL MONTHLY SPEND as one monthly item.
    """
    projected: list[CashFlowItem] = []
    protected_cats = set(profile.expense_categories_to_protect)

    # Group expense events by (category, event_type, flexibility)
    groups: dict[tuple, list[FinancialEvent]] = defaultdict(list)
    for ev in settled_events:
        if (
            ev.direction == "debit"
            and ev.status == "settled"
            and ev.event_type not in EXCLUDE_EVENT_TYPES
            and ev.amount_home is not None
        ):
            key = (ev.category, ev.flexibility, ev.event_type)
            groups[key].append(ev)

    for (category, flexibility, event_type), events in groups.items():
        # Sort by date
        events.sort(key=lambda e: e.event_date)
        if len(events) < MIN_RECURRENCE_COUNT:
            continue

        # Compute intervals between consecutive events
        intervals = []
        for i in range(1, len(events)):
            delta = (events[i].event_date - events[i - 1].event_date).days
            intervals.append(delta)

        if not intervals:
            continue

        median_interval = _median_amount(intervals)

        is_monthly = 20 <= median_interval <= 45
        is_weekly = 3 <= median_interval < 20

        if not (is_monthly or is_weekly):
            continue

        # Check amount stability
        amounts = [ev.amount_home for ev in events if ev.amount_home is not None]
        if not amounts:
            continue

        is_essential = category in protected_cats
        last_ev = events[-1]

        if is_monthly:
            # Project each monthly occurrence individually (rent, utilities, subscriptions)
            cv = _coeff_variation(amounts)
            if cv > AMOUNT_CV_THRESHOLD and flexibility == "fixed":
                continue
            median_amt = _median_amount(amounts[-3:])
            if median_amt <= 0:
                continue

            recent = events[-3:]
            days_of_month = [e.event_date.day for e in recent]
            modal_day = max(1, min(28, round(statistics.mean(days_of_month))))

            next_d = _infer_next_date(last_ev.event_date, modal_day)
            while next_d < request_date:
                next_d = _infer_next_date(next_d, modal_day)

            occurrence_count = 0
            while next_d <= forecast_end and occurrence_count < 4:
                projected.append(CashFlowItem(
                    event_date=next_d,
                    amount=round(median_amt, 2),
                    is_income=False,
                    category=category,
                    event_id=last_ev.event_id,
                    flexibility=flexibility,
                    minimum_allowed_amount=last_ev.minimum_allowed_amount,
                    is_confirmed=True,
                    is_essential=is_essential,
                    description=last_ev.description,
                ))
                next_d = _infer_next_date(next_d, modal_day)
                occurrence_count += 1

        elif is_weekly:
            # Weekly/frequent expenses (groceries, transport)
            # Project individual occurrences at the median interval
            median_amt = _median_amount(amounts[-6:])  # last 6 occurrences
            if median_amt <= 0:
                continue

            # Only include if relatively stable (low CV)
            cv = _coeff_variation(amounts[-6:] if len(amounts) >= 6 else amounts)
            if cv > 0.5:
                continue  # Too variable, skip

            freq_days = int(round(median_interval))

            next_d = last_ev.event_date + timedelta(days=freq_days)
            while next_d < request_date:
                next_d += timedelta(days=freq_days)

            occurrence_count = 0
            while next_d <= forecast_end and occurrence_count < 15:
                projected.append(CashFlowItem(
                    event_date=next_d,
                    amount=round(median_amt, 2),
                    is_income=False,
                    category=category,
                    event_id=last_ev.event_id,
                    flexibility=flexibility,
                    minimum_allowed_amount=last_ev.minimum_allowed_amount,
                    is_confirmed=True,
                    is_essential=is_essential,
                    description=last_ev.description,
                ))
                next_d += timedelta(days=freq_days)
                occurrence_count += 1

    return projected


def _infer_recurring_income(
    settled_income: list[FinancialEvent],
    scheduled_income: list[FinancialEvent],
    request_date: date,
    forecast_end: date,
    msg_context: Optional[ResolvedMessageContext],
) -> list[CashFlowItem]:
    """
    Infer future salary/income from historical patterns + scheduled events.
    Only produces CONFIRMED income items.
    """
    projected: list[CashFlowItem] = []

    # First: use scheduled income events from dataset (highest priority)
    for ev in scheduled_income:
        if ev.event_date > request_date and ev.amount_home is not None:
            amount = ev.amount_home
            # Apply message salary override if available
            if msg_context and msg_context.confirmed_next_salary is not None:
                amount = msg_context.confirmed_next_salary
            projected.append(CashFlowItem(
                event_date=ev.event_date,
                amount=round(amount, 2),
                is_income=True,
                category=ev.category,
                event_id=ev.event_id,
                flexibility="fixed",
                minimum_allowed_amount=None,
                is_confirmed=True,
                is_essential=False,
                description=ev.description,
            ))

    # Check if employment has ended
    if msg_context and msg_context.employment_ended:
        return projected

    # Filter settled income to recurring base salary / payroll events
    # Exclude variable/unconfirmed streams like commission, bonus, refund, arrears
    non_salary_keywords = {"commission", "bonus", "refund", "arrears", "promotion", "sales commission"}
    salary_events = []
    for e in settled_income:
        if (
            e.event_type == "income"
            and e.status == "settled"
            and e.direction == "credit"
            and e.amount_home is not None
        ):
            desc_lower = e.description.lower()
            if any(kw in desc_lower for kw in non_salary_keywords):
                continue
            salary_events.append(e)

    # Check if latest settled event was a final payroll
    if salary_events:
        salary_events.sort(key=lambda e: e.event_date)
        last_desc = salary_events[-1].description.lower()
        if "final employer payroll" in last_desc and (not msg_context or msg_context.confirmed_next_salary is None):
            return projected

    # If no settled salary events and we already have scheduled events, continue from scheduled
    if not salary_events and projected:
        seed = max(projected, key=lambda x: x.event_date)
        modal_day = seed.event_date.day
        scheduled_dates = {item.event_date for item in projected}
        next_d = _infer_next_date(seed.event_date, modal_day)
        while next_d < request_date:
            next_d = _infer_next_date(next_d, modal_day)
        count = 0
        while next_d <= forecast_end and count < 4:
            if next_d not in scheduled_dates:
                projected.append(CashFlowItem(
                    event_date=next_d,
                    amount=seed.amount,
                    is_income=True,
                    category="salary",
                    event_id=f"inferred_salary_from_scheduled_{count}",
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                    is_confirmed=True,
                    is_essential=False,
                    description="Inferred salary payment",
                ))
            next_d = _infer_next_date(next_d, modal_day)
            count += 1
        return projected

    if not salary_events and not (msg_context and msg_context.confirmed_next_salary is not None):
        return projected

    if salary_events:
        salary_events.sort(key=lambda e: e.event_date)
        last_salary = salary_events[-1]

        # Check for monthly pattern
        intervals = []
        for i in range(1, len(salary_events)):
            delta = (salary_events[i].event_date - salary_events[i - 1].event_date).days
            intervals.append(delta)

        if not intervals:
            modal_day = last_salary.event_date.day
        else:
            recent = salary_events[-3:]
            days_of_month = [e.event_date.day for e in recent]
            modal_day = round(statistics.mean(days_of_month))

        amounts = [last_salary.amount_home]
        amounts += [e.amount_home for e in salary_events[-3:] if e.amount_home is not None]
        median_amt = _median_amount(amounts)
        last_event_id = last_salary.event_id
        seed_date = last_salary.event_date
    else:
        modal_day = 15
        median_amt = msg_context.confirmed_next_salary
        last_event_id = "msg_override"
        seed_date = request_date

    # Apply message salary override
    if msg_context and msg_context.confirmed_next_salary is not None:
        median_amt = msg_context.confirmed_next_salary
        if msg_context.confirmed_next_salary_date:
            try:
                from datetime import datetime
                parsed_d = datetime.strptime(msg_context.confirmed_next_salary_date, "%Y-%m-%d").date()
                modal_day = parsed_d.day
                seed_date = parsed_d
            except ValueError:
                pass

    # Build set of already-projected dates (from scheduled events)
    scheduled_dates = {item.event_date for item in projected}

    if projected:
        seed_date = max(item.event_date for item in projected)

    # Continue projecting monthly salary from the seed date
    if seed_date >= request_date and seed_date not in scheduled_dates:
        next_d = seed_date
    else:
        next_d = _infer_next_date(seed_date, modal_day)

    # Advance past request_date
    while next_d < request_date:
        next_d = _infer_next_date(next_d, modal_day)

    count = 0
    while next_d <= forecast_end and count < 4:
        if next_d not in scheduled_dates:
            projected.append(CashFlowItem(
                event_date=next_d,
                amount=round(median_amt, 2),
                is_income=True,
                category="salary",
                event_id=f"inferred_salary_{last_event_id}_{count}",
                flexibility="fixed",
                minimum_allowed_amount=None,
                is_confirmed=True,
                is_essential=False,
                description="Inferred salary payment",
            ))
        next_d = _infer_next_date(next_d, modal_day)
        count += 1

    return projected


# ── Main builder ──────────────────────────────────────────────────────────────

def build_financial_state(
    profile: UserProfile,
    events: list[FinancialEvent],
    request_date: date,
    image_amounts: dict[str, float],
    msg_context: Optional[ResolvedMessageContext],
) -> CanonicalFinancialState:
    """
    Build the canonical financial state for a user on a given request_date.
    """
    forecast_end = request_date + timedelta(days=90)
    protected_cats = set(profile.expense_categories_to_protect)

    # ── Step 1: Apply image-extracted amounts ─────────────────────────────────
    for ev in events:
        if ev.amount is None and ev.event_id in image_amounts:
            ev.amount = image_amounts[ev.event_id]
            ev.amount_home = image_amounts[ev.event_id]  # Assumed in home currency

    # ── Step 2: Apply message amendments ─────────────────────────────────────
    cancelled_ids: set[str] = set()
    if msg_context:
        cancelled_ids = msg_context.cancelled_event_ids
        for eid, new_amt in msg_context.amended_amounts.items():
            for ev in events:
                if ev.event_id == eid:
                    ev.amount = new_amt
                    ev.amount_home = new_amt

    # ── Step 3: Classify events ───────────────────────────────────────────────
    settled: list[FinancialEvent] = []
    pending_future: list[FinancialEvent] = []
    scheduled_income: list[FinancialEvent] = []

    for ev in events:
        # Skip cancelled, failed, unrealized, non-cash
        if ev.status in EXCLUDE_STATUSES:
            continue
        if ev.event_id in cancelled_ids:
            continue
        if ev.event_type in EXCLUDE_EVENT_TYPES:
            continue
        if ev.direction in EXCLUDE_DIRECTIONS:
            continue
        if ev.amount_home is None:
            continue  # Can't use events with unknown amounts

        if ev.status == "settled":
            settled.append(ev)
        elif ev.status in INCLUDE_FUTURE_STATUSES:
            if ev.direction == "credit" and ev.event_type == "income":
                scheduled_income.append(ev)
            else:
                pending_future.append(ev)

    settled_expenses = [e for e in settled if e.direction == "debit"]
    settled_income_events = [e for e in settled if e.direction == "credit"]

    # ── Step 4: Build pending debit items (already in dataset, future dates) ─
    cash_flow_items: list[CashFlowItem] = []
    pending_debits: list[CashFlowItem] = []

    for ev in pending_future:
        if ev.direction != "debit":
            # Skip pending credits per problem statement
            continue
        use_date = ev.settlement_date or ev.event_date
        if use_date < request_date:
            use_date = request_date  # Treat overdue pending as due today
        item = CashFlowItem(
            event_date=use_date,
            amount=round(ev.amount_home, 2),
            is_income=False,
            category=ev.category,
            event_id=ev.event_id,
            flexibility=ev.flexibility,
            minimum_allowed_amount=ev.minimum_allowed_amount,
            is_confirmed=True,
            is_essential=(ev.category in protected_cats),
            description=ev.description,
        )
        cash_flow_items.append(item)
        pending_debits.append(item)

    # ── Step 5: Infer recurring expenses ─────────────────────────────────────
    inferred_expenses = _infer_recurring_expenses(
        settled_expenses, request_date, forecast_end, profile
    )
    cash_flow_items.extend(inferred_expenses)

    # ── Step 6: Infer recurring income ───────────────────────────────────────
    inferred_income = _infer_recurring_income(
        settled_income_events, scheduled_income, request_date, forecast_end, msg_context
    )
    cash_flow_items.extend(inferred_income)

    # ── Step 7: Collect flexible items ───────────────────────────────────────
    flexible_items = [
        item for item in cash_flow_items
        if not item.is_income and _is_flexible_item(item, profile)
    ]

    return CanonicalFinancialState(
        balance=profile.current_available_balance,
        min_balance=profile.minimum_balance_to_keep,
        home_currency=profile.home_currency,
        cash_flow_items=cash_flow_items,
        pending_debits=pending_debits,
        flexible_items=flexible_items,
    )


def _is_flexible_item(item: CashFlowItem, profile: UserProfile) -> bool:
    """Check if this item is one the user is willing to reduce or stop."""
    if item.flexibility not in ("stoppable", "reducible", "reducible_or_stoppable"):
        return False
    cat = item.category
    return (
        cat in profile.expense_categories_user_is_willing_to_reduce
        or cat in profile.expense_categories_user_is_willing_to_stop
    )
