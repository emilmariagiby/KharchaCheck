"""
strategy.py — Generate all candidate payment plans and apply the official ranking.

Strategy generation:
    FULL          — pay requested_amount today
    PARTIAL       — pay amount_safe_to_pay today, remainder on earliest_safe_date
    INSTALLMENTS  — follow exact payment_option schedule from dataset
    WAIT          — pay requested_amount on earliest_safe_date
    CUT_FLEXIBLE  — pay full today after stopping/reducing flexible expenses
    NOT_RECOMMENDED — fallback

Official ranking (lexicographic, per problem spec):
    1. Completes by desired_completion_date
    2. No spending changes needed
    3. Minimize total amount paid
    4. Start payment earlier
    5. Fewer payments
    6. Lowest payment_option_id
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from cashflow import ScheduledPayment, SpendingCut, simulate
from data_loader import PaymentOption, UserProfile
from financial_state import CanonicalFinancialState, CashFlowItem


@dataclass
class PaymentPlan:
    method: str                       # full_payment, partial_payment, installments, wait, not_recommended
    payments: list[tuple[date, float]]  # (date, amount)
    spending_cuts: list[SpendingCut]
    is_safe: bool
    safety_margin: float
    total_amount_paid: float
    completes_by_deadline: bool
    payment_option_id: Optional[str] = None  # For installments


def _format_plan(payments: list[tuple[date, float]]) -> str:
    if not payments:
        return "none"
    return "|".join(f"{d}:{round(amt, 2)}" for d, amt in sorted(payments))


def _simulate_plan(
    state: CanonicalFinancialState,
    payments: list[tuple[date, float]],
    cuts: list[SpendingCut],
    request_date: date,
) -> tuple[bool, float]:
    """Return (is_safe, safety_margin)."""
    scheduled = [ScheduledPayment(d, a) for d, a in payments]
    result = simulate(state, scheduled, cuts, request_date)
    return result.is_safe, result.safety_margin


def _plan_completes_by(payments: list[tuple[date, float]], deadline: date) -> bool:
    if not payments:
        return False
    last_date = max(d for d, _ in payments)
    return last_date <= deadline


# ── Strategy generators ───────────────────────────────────────────────────────

def try_full_payment(
    state: CanonicalFinancialState,
    requested_amount: float,
    request_date: date,
    deadline: date,
    profile: UserProfile,
) -> Optional[PaymentPlan]:
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None
    payments = [(request_date, requested_amount)]
    is_safe, margin = _simulate_plan(state, payments, [], request_date)
    return PaymentPlan(
        method="full_payment",
        payments=payments,
        spending_cuts=[],
        is_safe=is_safe,
        safety_margin=margin,
        total_amount_paid=requested_amount,
        completes_by_deadline=_plan_completes_by(payments, deadline),
    )


def try_partial_payment(
    state: CanonicalFinancialState,
    requested_amount: float,
    amount_safe_today: float,
    earliest_date: Optional[date],
    request_date: date,
    deadline: date,
    profile: UserProfile,
    allows_partial: bool,
) -> Optional[PaymentPlan]:
    if "partial_payment" not in profile.payment_methods_user_will_consider:
        return None
    if not allows_partial:
        return None
    if amount_safe_today <= 0 or amount_safe_today >= requested_amount:
        return None
    if earliest_date is None:
        return None
    if earliest_date > deadline:
        return None

    remainder = round(requested_amount - amount_safe_today, 2)
    payments = [(request_date, amount_safe_today), (earliest_date, remainder)]
    is_safe, margin = _simulate_plan(state, payments, [], request_date)
    return PaymentPlan(
        method="partial_payment",
        payments=payments,
        spending_cuts=[],
        is_safe=is_safe,
        safety_margin=margin,
        total_amount_paid=requested_amount,
        completes_by_deadline=_plan_completes_by(payments, deadline),
    )


def try_installment(
    state: CanonicalFinancialState,
    option: PaymentOption,
    request_date: date,
    deadline: date,
    profile: UserProfile,
) -> Optional[PaymentPlan]:
    if "installments" not in profile.payment_methods_user_will_consider:
        return None

    # Check max installment months constraint
    if profile.max_installment_months is not None:
        if option.number_of_payments > profile.max_installment_months:
            return None

    payments = option.generate_schedule()
    if not payments:
        return None

    # Reject if first payment is before request_date
    if payments[0][0] < request_date:
        return None

    is_safe, margin = _simulate_plan(state, payments, [], request_date)
    return PaymentPlan(
        method="installments",
        payments=payments,
        spending_cuts=[],
        is_safe=is_safe,
        safety_margin=margin,
        total_amount_paid=option.total_payable_amount,
        completes_by_deadline=_plan_completes_by(payments, deadline),
        payment_option_id=option.payment_option_id,
    )


def try_wait(
    state: CanonicalFinancialState,
    requested_amount: float,
    earliest_date: Optional[date],
    request_date: date,
    deadline: date,
    profile: UserProfile,
) -> Optional[PaymentPlan]:
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None
    if earliest_date is None:
        return None
    if earliest_date <= request_date:
        # Already safe now — FULL is better
        return None
    if earliest_date > deadline:
        # Cannot wait past desired deadline
        return None

    payments = [(earliest_date, requested_amount)]
    is_safe, margin = _simulate_plan(state, payments, [], request_date)
    return PaymentPlan(
        method="wait",
        payments=payments,
        spending_cuts=[],
        is_safe=is_safe,
        safety_margin=margin,
        total_amount_paid=requested_amount,
        completes_by_deadline=_plan_completes_by(payments, deadline),
    )


def try_cut_flexible_spending(
    state: CanonicalFinancialState,
    requested_amount: float,
    request_date: date,
    deadline: date,
    profile: UserProfile,
) -> Optional[PaymentPlan]:
    """
    Greedy: add spending cuts one at a time (largest saving first) until
    full payment on request_date becomes safe.
    Caps at 3 cuts per problem spec.
    """
    if "full_payment" not in profile.payment_methods_user_will_consider:
        return None

    # Candidates: flexible items in categories user is willing to reduce/stop
    stop_cats = set(profile.expense_categories_user_is_willing_to_stop)
    reduce_cats = set(profile.expense_categories_user_is_willing_to_reduce)

    # Filter to only the ACTUAL recurring items (not inferred duplicates)
    seen_categories: dict[str, CashFlowItem] = {}
    for item in state.flexible_items:
        if item.category not in seen_categories:
            seen_categories[item.category] = item

    candidates: list[tuple[float, CashFlowItem, str]] = []  # (saving, item, type)
    for item in seen_categories.values():
        if item.category in stop_cats and item.flexibility in ("stoppable", "reducible_or_stoppable"):
            candidates.append((item.amount, item, "stop"))
        elif item.category in reduce_cats and item.flexibility in ("reducible", "reducible_or_stoppable"):
            # Reduce to minimum_allowed_amount or 50% whichever is higher
            min_amt = item.minimum_allowed_amount or round(item.amount * 0.5, 2)
            saving = item.amount - min_amt
            if saving > 0:
                candidates.append((saving, item, "reduce"))

    # Sort by saving descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    accumulated_cuts: list[SpendingCut] = []
    for saving, item, cut_type in candidates[:3]:  # max 3 cuts
        if cut_type == "stop":
            cut = SpendingCut(
                event_id=item.event_id.replace("inferred_", "").rsplit("_", 1)[0]
                    if item.event_id.startswith("inferred_") else item.event_id,
                original_amount=item.amount,
                new_amount=0.0,
                cut_type="stop",
                category=item.category,
                flexibility=item.flexibility,
            )
        else:
            min_amt = item.minimum_allowed_amount or round(item.amount * 0.5, 2)
            cut = SpendingCut(
                event_id=item.event_id.replace("inferred_", "").rsplit("_", 1)[0]
                    if item.event_id.startswith("inferred_") else item.event_id,
                original_amount=item.amount,
                new_amount=round(min_amt, 2),
                cut_type="reduce_to",
                category=item.category,
                flexibility=item.flexibility,
            )

        accumulated_cuts.append(cut)

        payments = [(request_date, requested_amount)]
        is_safe, margin = _simulate_plan(state, payments, accumulated_cuts, request_date)
        if is_safe:
            return PaymentPlan(
                method="full_payment",
                payments=payments,
                spending_cuts=accumulated_cuts[:],
                is_safe=True,
                safety_margin=margin,
                total_amount_paid=requested_amount,
                completes_by_deadline=True,
            )

    return None


# ── Official ranking ──────────────────────────────────────────────────────────

def _rank_key(plan: PaymentPlan) -> tuple:
    first_date = min(d for d, _ in plan.payments) if plan.payments else date.max
    return (
        0 if plan.completes_by_deadline else 1,    # 1. completes by desired_completion_date
        0 if not plan.spending_cuts else 1,        # 2. no spending changes needed first
        plan.total_amount_paid,                     # 3. minimize total amount paid (fees/interest)
        first_date,                                 # 4. start payment earlier (pay today vs wait)
        len(plan.payments),                         # 5. fewer payments
        plan.payment_option_id or "zzz",            # 6. lowest payment_option_id
    )


def select_best_plan(
    state: CanonicalFinancialState,
    requested_amount: float,
    amount_safe_today: float,
    earliest_date: Optional[date],
    request_date: date,
    deadline: date,
    profile: UserProfile,
    payment_options: list[PaymentOption],
    allows_partial: bool,
) -> PaymentPlan:
    """
    Generate all candidate plans, apply safety gate, return best by official ranking.
    """
    candidates: list[PaymentPlan] = []

    # 1. Full payment
    plan = try_full_payment(state, requested_amount, request_date, deadline, profile)
    if plan:
        candidates.append(plan)

    # 2. Partial payment
    plan = try_partial_payment(
        state, requested_amount, amount_safe_today, earliest_date,
        request_date, deadline, profile, allows_partial
    )
    if plan:
        candidates.append(plan)

    # 3. All installment options (only method=installments from payment_options)
    for option in payment_options:
        if option.payment_method == "installments":
            plan = try_installment(state, option, request_date, deadline, profile)
            if plan:
                candidates.append(plan)

    # 4. Wait
    plan = try_wait(state, requested_amount, earliest_date, request_date, deadline, profile)
    if plan:
        candidates.append(plan)

    # 5. Cut flexible spending (full payment with cuts)
    plan = try_cut_flexible_spending(state, requested_amount, request_date, deadline, profile)
    if plan:
        candidates.append(plan)

    # Safety gate: keep only safe plans
    safe_plans = [p for p in candidates if p.is_safe]

    if not safe_plans:
        return PaymentPlan(
            method="not_recommended",
            payments=[],
            spending_cuts=[],
            is_safe=False,
            safety_margin=0.0,
            total_amount_paid=0.0,
            completes_by_deadline=False,
        )

    return min(safe_plans, key=_rank_key)


def format_spending_changes(cuts: list[SpendingCut]) -> str:
    if not cuts:
        return "none"
    parts = []
    for cut in cuts:
        if cut.cut_type == "stop":
            parts.append(f"stop:{cut.event_id}")
        else:
            parts.append(f"reduce_to:{cut.event_id}:{round(cut.new_amount, 2)}")
    return "|".join(parts)


def derive_status(
    best_plan: PaymentPlan,
    requested_amount: float,
    amount_safe_today: float,
    request_date: date,
) -> str:
    """Derive affordability_status from the best plan."""
    if best_plan.method == "not_recommended":
        return "not_affordable"

    if best_plan.method == "full_payment" and not best_plan.spending_cuts:
        # Full payment today without cuts
        first_date = min(d for d, _ in best_plan.payments)
        if first_date == request_date and amount_safe_today >= requested_amount:
            return "affordable_now"

    if best_plan.method in ("full_payment", "partial_payment", "installments"):
        if best_plan.completes_by_deadline:
            return "affordable_with_plan"

    if best_plan.method == "wait":
        return "affordable_later"

    # Fallback
    if best_plan.completes_by_deadline:
        return "affordable_with_plan"
    return "affordable_later"
