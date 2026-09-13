"""
explainer.py — LLM-generated decision explanation (last step, non-blocking).

This is the ONLY place where the LLM influences the recommendation.
It receives the already-computed financial facts and turns them into a 
short, factual explanation. It does NOT make any financial decisions.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

try:
    from google import genai as _genai_new
    _USE_NEW_SDK = True
except ImportError:
    import google.generativeai as _genai_old
    _USE_NEW_SDK = False

from config import GOOGLE_API_KEY, LLM_MODEL
from strategy import PaymentPlan


_EXPLAIN_PROMPT = """You are writing a one-sentence financial decision explanation.

Facts:
- User balance: {balance} {currency}
- Minimum balance to keep: {min_balance} {currency}
- Requested amount: {requested_amount} {currency}
- Amount safe to pay today: {safe_amount} {currency}
- Recommended method: {method}
- Payment plan: {payment_plan}
- Earliest date for full payment: {earliest_date}
- Spending changes needed: {spending_changes}

Write exactly ONE short sentence (under 30 words) explaining the recommendation.
Start with the action (e.g. "Pay", "Use", "Wait"). Include the key financial fact.
Do not invent numbers not in the facts above. Return only the sentence, no formatting.
"""


def generate_explanation(
    balance: float,
    currency: str,
    min_balance: float,
    requested_amount: float,
    safe_amount: float,
    best_plan: PaymentPlan,
    earliest_date: Optional[date],
    spending_changes_str: str,
    usage_tracker=None,
    request_id: str = "",
) -> str:
    """Generate a short explanation using LLM. Falls back to rule-based if LLM unavailable."""

    if not GOOGLE_API_KEY:
        return _fallback_explanation(
            currency, requested_amount, safe_amount, best_plan, earliest_date, min_balance
        )

    try:
        payment_plan_str = (
            "|".join(f"{d}:{a}" for d, a in best_plan.payments)
            if best_plan.payments else "none"
        )
        prompt = _EXPLAIN_PROMPT.format(
            balance=round(balance, 2),
            currency=currency,
            min_balance=round(min_balance, 2),
            requested_amount=round(requested_amount, 2),
            safe_amount=round(safe_amount, 2),
            method=best_plan.method,
            payment_plan=payment_plan_str,
            earliest_date=str(earliest_date) if earliest_date else "not within 90 days",
            spending_changes=spending_changes_str,
        )

        if _USE_NEW_SDK:
            client = _genai_new.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            text = response.text.strip()
        else:
            _genai_old.configure(api_key=GOOGLE_API_KEY)
            model = _genai_old.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            text = response.text.strip()

        if usage_tracker:
            in_tok = len(prompt) // 4
            out_tok = len(text) // 4
            usage_tracker.record(
                request_id=request_id,
                model=LLM_MODEL,
                call_type="explanation",
                input_tokens=in_tok,
                output_tokens=out_tok,
            )

        # Clean any markdown
        text = re.sub(r"\*+", "", text).strip()
        return text if text else _fallback_explanation(
            currency, requested_amount, safe_amount, best_plan, earliest_date, min_balance
        )

    except Exception as e:
        print(f"  [explainer] LLM error ({request_id}): {e} — using fallback")
        return _fallback_explanation(
            currency, requested_amount, safe_amount, best_plan, earliest_date, min_balance
        )


def _fallback_explanation(
    currency: str,
    requested_amount: float,
    safe_amount: float,
    best_plan: PaymentPlan,
    earliest_date: Optional[date],
    min_balance: float,
) -> str:
    """Rule-based explanation when LLM is unavailable."""
    amt = round(requested_amount, 2)
    safe = round(safe_amount, 2)
    mb = round(min_balance, 2)

    if best_plan.method == "not_recommended":
        return (
            f"Not recommended: paying {currency} {amt} would breach the "
            f"{currency} {mb} minimum balance throughout the 90-day forecast."
        )
    if best_plan.method == "full_payment" and not best_plan.spending_cuts:
        return f"Pay {currency} {amt} today; this leaves at least {currency} {mb} available over the next 90 days."
    if best_plan.method == "full_payment" and best_plan.spending_cuts:
        return (
            f"Pay {currency} {amt} today after reducing flexible spending; "
            f"this keeps the balance above {currency} {mb}."
        )
    if best_plan.method == "partial_payment":
        return (
            f"Pay {currency} {safe} today and the remaining "
            f"{currency} {round(amt - safe, 2)} on {earliest_date}."
        )
    if best_plan.method == "installments":
        n = len(best_plan.payments)
        first_amt = round(best_plan.payments[0][1], 2) if best_plan.payments else 0
        return f"Use {n} installments of {currency} {first_amt}; this keeps the balance safe throughout."
    if best_plan.method == "wait":
        return (
            f"Paying {currency} {amt} today would breach the minimum; "
            f"wait until {earliest_date} when the full amount is safe."
        )
    return f"Recommended method: {best_plan.method} for {currency} {amt}."
