"""
data_loader.py — Loads and pre-processes all dataset CSV files.

Responsibilities:
- Load all 9 CSVs with correct dtypes
- Parse pipe-separated profile fields into Python lists
- Build exchange rate lookup with date interpolation
- Convert all event amounts to home_currency
- Return a per-request RequestBundle
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    FINANCIAL_EVENTS_CSV,
    FINANCIAL_PROFILES_CSV,
    EXCHANGE_RATES_CSV,
    IMAGES_CSV,
    MESSAGES_CSV,
    PAYMENT_OPTIONS_CSV,
    REQUESTS_CSV,
    SAMPLE_REQUESTS_CSV,
)

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: list[str]
    expense_categories_to_protect: list[str]
    expense_categories_user_is_willing_to_reduce: list[str]
    expense_categories_user_is_willing_to_stop: list[str]
    payment_methods_user_will_consider: list[str]
    max_installment_months: Optional[float]


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str           # expense, income, subscription, debt_payment, refund, investment_*
    description: str
    category: str
    direction: str            # debit, credit, non_cash
    amount: Optional[float]   # may be None (resolved from image)
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str               # settled, pending, scheduled, cancelled, failed, unrealized
    linked_event_id: Optional[str]
    flexibility: str          # fixed, stoppable, reducible, reducible_or_stoppable
    minimum_allowed_amount: Optional[float]
    # Filled in after currency conversion:
    amount_home: Optional[float] = None


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str       # full_payment, installments, partial_payment
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[float]   # None for single-payment options
    financing_fee: float
    total_payable_amount: float

    def generate_schedule(self) -> list[tuple[date, float]]:
        """Return list of (payment_date, amount) in chronological order."""
        from datetime import timedelta
        schedule = []
        for i in range(self.number_of_payments):
            if i == 0:
                d = self.first_payment_date
            else:
                freq = int(round(self.payment_frequency_days or 30))
                d = self.first_payment_date + timedelta(days=freq * i)
            schedule.append((d, round(self.payment_amount, 2)))
        return schedule


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]
    sent_at: datetime
    source_type: str          # employer, service_provider, bank, merchant, financial_service
    message_text: str


@dataclass
class ImageRecord:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str


@dataclass
class RequestBundle:
    """Everything needed to process one request."""
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str
    profile: UserProfile
    events: list[FinancialEvent]           # user's events, converted to home_currency
    payment_options: list[PaymentOption]   # options for THIS request
    messages: list[Message]                # user-level + request-level messages
    image_records: list[ImageRecord]       # images linked to this request's user


# ── Exchange rate helper ───────────────────────────────────────────────────────

class ExchangeRateTable:
    """Lookup: convert amount from one currency to another on a given date."""

    def __init__(self, rates_df: pd.DataFrame):
        # Build dict: (from_ccy, to_ccy) → sorted list of (rate_date, rate)
        self._table: dict[tuple[str, str], list[tuple[date, float]]] = {}
        for _, row in rates_df.iterrows():
            key = (row["from_currency"], row["to_currency"])
            d = _parse_date(row["rate_date"])
            self._table.setdefault(key, []).append((d, float(row["rate"])))
        for key in self._table:
            self._table[key].sort(key=lambda x: x[0])

    def convert(self, amount: float, from_ccy: str, to_ccy: str, on_date: date) -> float:
        if from_ccy == to_ccy:
            return amount
        rate = self._get_rate(from_ccy, to_ccy, on_date)
        if rate is not None:
            return amount * rate
        # Try via USD as bridge
        rate_to_usd = self._get_rate(to_ccy, from_ccy, on_date)
        if rate_to_usd:
            return amount / rate_to_usd
        # Try: from → USD → to
        r1 = self._get_rate("USD", from_ccy, on_date)
        r2 = self._get_rate("USD", to_ccy, on_date)
        if r1 and r2:
            return amount * (r2 / r1)
        # Fallback: no conversion (return as-is, log warning)
        return amount

    def _get_rate(self, from_ccy: str, to_ccy: str, on_date: date) -> Optional[float]:
        key = (from_ccy, to_ccy)
        if key not in self._table:
            return None
        entries = self._table[key]
        # Find nearest date (prefer closest on or before)
        best = None
        for d, r in entries:
            if d <= on_date:
                best = r
            else:
                break
        if best is None:
            # All dates are after on_date — use earliest available
            best = entries[0][1]
        return best


# ── Loaders ───────────────────────────────────────────────────────────────────

def _parse_date(s) -> date:
    if pd.isna(s):
        return None
    if isinstance(s, date):
        return s
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()


def _parse_pipe_list(s) -> list[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def load_profiles() -> dict[str, UserProfile]:
    df = pd.read_csv(FINANCIAL_PROFILES_CSV, dtype=str)
    profiles = {}
    for _, row in df.iterrows():
        uid = row["user_id"]
        profiles[uid] = UserProfile(
            user_id=uid,
            home_currency=row["home_currency"],
            current_available_balance=float(row["current_available_balance"]),
            minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
            financial_priorities=_parse_pipe_list(row["financial_priorities"]),
            expense_categories_to_protect=_parse_pipe_list(row["expense_categories_to_protect"]),
            expense_categories_user_is_willing_to_reduce=_parse_pipe_list(
                row["expense_categories_user_is_willing_to_reduce"]
            ),
            expense_categories_user_is_willing_to_stop=_parse_pipe_list(
                row["expense_categories_user_is_willing_to_stop"]
            ),
            payment_methods_user_will_consider=_parse_pipe_list(
                row["payment_methods_user_will_consider"]
            ),
            max_installment_months=(
                float(row["max_installment_months"])
                if not pd.isna(row["max_installment_months"])
                else None
            ),
        )
    return profiles


def load_events(rates: ExchangeRateTable, profiles: dict[str, UserProfile]) -> dict[str, list[FinancialEvent]]:
    """Return dict: user_id → list[FinancialEvent] with amounts in home_currency."""
    df = pd.read_csv(FINANCIAL_EVENTS_CSV, dtype=str)
    user_events: dict[str, list[FinancialEvent]] = {}

    for _, row in df.iterrows():
        uid = row["user_id"]
        raw_amount = row["amount"]
        amount = float(raw_amount) if not pd.isna(raw_amount) else None
        ev_date = _parse_date(row["event_date"])
        settlement = _parse_date(row["settlement_date"]) if not pd.isna(row["settlement_date"]) else None
        min_allowed = float(row["minimum_allowed_amount"]) if not pd.isna(row["minimum_allowed_amount"]) else None
        linked = row["linked_event_id"] if not pd.isna(row["linked_event_id"]) else None

        # Currency conversion
        event_ccy = str(row["currency"]).strip()
        home_ccy = profiles[uid].home_currency if uid in profiles else event_ccy
        amount_home: Optional[float] = None
        if amount is not None:
            ref_date = settlement or ev_date
            amount_home = rates.convert(amount, event_ccy, home_ccy, ref_date)

        ev = FinancialEvent(
            event_id=row["event_id"],
            user_id=uid,
            event_type=row["event_type"],
            description=str(row["description"]),
            category=str(row["category"]),
            direction=str(row["direction"]),
            amount=amount,
            currency=event_ccy,
            event_date=ev_date,
            settlement_date=settlement,
            status=str(row["status"]),
            linked_event_id=linked,
            flexibility=str(row["flexibility"]),
            minimum_allowed_amount=min_allowed,
            amount_home=amount_home,
        )
        user_events.setdefault(uid, []).append(ev)

    return user_events


def load_payment_options() -> dict[str, list[PaymentOption]]:
    """Return dict: request_id → list[PaymentOption]."""
    df = pd.read_csv(PAYMENT_OPTIONS_CSV, dtype=str)
    options: dict[str, list[PaymentOption]] = {}
    for _, row in df.iterrows():
        freq = float(row["payment_frequency_days"]) if not pd.isna(row["payment_frequency_days"]) else None
        opt = PaymentOption(
            payment_option_id=row["payment_option_id"],
            request_id=row["request_id"],
            payment_method=row["payment_method"],
            payment_amount=float(row["payment_amount"]),
            number_of_payments=int(row["number_of_payments"]),
            first_payment_date=_parse_date(row["first_payment_date"]),
            payment_frequency_days=freq,
            financing_fee=float(row["financing_fee"]),
            total_payable_amount=float(row["total_payable_amount"]),
        )
        options.setdefault(row["request_id"], []).append(opt)
    return options


def load_messages() -> dict[str, list[Message]]:
    """Return dict: user_id → list[Message] (sorted by sent_at)."""
    df = pd.read_csv(MESSAGES_CSV, dtype=str)
    msgs: dict[str, list[Message]] = {}
    for _, row in df.iterrows():
        uid = row["user_id"]
        request_id = row["request_id"] if not pd.isna(row["request_id"]) else None
        related_event_id = row["related_event_id"] if not pd.isna(row["related_event_id"]) else None
        sent_at_str = str(row["sent_at"]).replace("Z", "+00:00")
        try:
            sent_at = datetime.fromisoformat(sent_at_str)
        except ValueError:
            sent_at = datetime.now()
        m = Message(
            message_id=row["message_id"],
            user_id=uid,
            request_id=request_id,
            related_event_id=related_event_id,
            sent_at=sent_at,
            source_type=str(row["source_type"]),
            message_text=str(row["message_text"]),
        )
        msgs.setdefault(uid, []).append(m)
    # Sort each user's messages by sent_at
    for uid in msgs:
        msgs[uid].sort(key=lambda x: x.sent_at)
    return msgs


def load_images() -> dict[str, list[ImageRecord]]:
    """Return dict: user_id → list[ImageRecord]."""
    df = pd.read_csv(IMAGES_CSV, dtype=str)
    imgs: dict[str, list[ImageRecord]] = {}
    for _, row in df.iterrows():
        uid = row["user_id"]
        imgs.setdefault(uid, []).append(
            ImageRecord(
                image_id=row["image_id"],
                user_id=uid,
                request_id=row["request_id"],
                related_event_id=row["related_event_id"],
            )
        )
    return imgs


def load_exchange_rates() -> ExchangeRateTable:
    df = pd.read_csv(EXCHANGE_RATES_CSV, dtype=str)
    return ExchangeRateTable(df)


def load_requests(csv_path=None) -> pd.DataFrame:
    path = csv_path or REQUESTS_CSV
    df = pd.read_csv(path, dtype=str)
    df["requested_amount"] = df["requested_amount"].astype(float)
    df["allows_partial_payment"] = df["allows_partial_payment"].str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)
    return df


# ── Master loader ─────────────────────────────────────────────────────────────

class DataStore:
    """Loads all CSVs once; provides per-request bundles efficiently."""

    def __init__(self, requests_csv=None):
        print("Loading exchange rates...")
        self.rates = load_exchange_rates()
        print("Loading profiles...")
        self.profiles = load_profiles()
        print("Loading events...")
        self.events = load_events(self.rates, self.profiles)
        print("Loading payment options...")
        self.payment_options = load_payment_options()
        print("Loading messages...")
        self.messages = load_messages()
        print("Loading images...")
        self.images = load_images()
        print("Loading requests...")
        self.requests_df = load_requests(requests_csv)
        print(f"Data loaded: {len(self.requests_df)} requests, {sum(len(v) for v in self.events.values())} events")

    def get_bundle(self, row: pd.Series) -> RequestBundle:
        uid = row["user_id"]
        rid = row["request_id"]
        return RequestBundle(
            request_id=rid,
            user_id=uid,
            request_date=_parse_date(row["request_date"]),
            request_type=str(row["request_type"]),
            requested_amount=float(row["requested_amount"]),
            desired_completion_date=_parse_date(row["desired_completion_date"]),
            allows_partial_payment=bool(row["allows_partial_payment"]),
            request_text=str(row["request_text"]),
            profile=self.profiles.get(uid),
            events=self.events.get(uid, []),
            payment_options=self.payment_options.get(rid, []),
            messages=self.messages.get(uid, []),
            image_records=self.images.get(uid, []),
        )

    def iter_bundles(self):
        for _, row in self.requests_df.iterrows():
            yield self.get_bundle(row)
