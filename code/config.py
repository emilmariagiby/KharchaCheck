"""
config.py — Central configuration for the Buy or Wait financial agent.
"""
import os
from pathlib import Path

# ── Repo root (the directory containing this code/ folder) ──────────────────
REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"
CODE_DIR = REPO_ROOT / "code"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_CSV = REPO_ROOT / "output.csv"
USAGE_REPORT = REPO_ROOT / "evaluation" / "usage_report.md"
IMAGE_CACHE_FILE = CODE_DIR / ".image_cache.json"

# ── Dataset file paths ───────────────────────────────────────────────────────
REQUESTS_CSV = DATASET_DIR / "requests.csv"
SAMPLE_REQUESTS_CSV = DATASET_DIR / "sample_requests.csv"
FINANCIAL_PROFILES_CSV = DATASET_DIR / "financial_profiles.csv"
FINANCIAL_EVENTS_CSV = DATASET_DIR / "financial_events.csv"
EXCHANGE_RATES_CSV = DATASET_DIR / "exchange_rates.csv"
PAYMENT_OPTIONS_CSV = DATASET_DIR / "request_payment_options.csv"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
IMAGES_CSV = DATASET_DIR / "images.csv"

# ── Simulation parameters ─────────────────────────────────────────────────────
FORECAST_DAYS = 90

# ── LLM / VLM ────────────────────────────────────────────────────────────────
# Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or .env file
def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    # Check for .env file in REPO_ROOT or parent
    for env_path in [REPO_ROOT / ".env", REPO_ROOT.parent / ".env"]:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY=") or line.startswith("GEMINI_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

GOOGLE_API_KEY = _get_api_key()

# Model for text (message parsing + explanation generation)
LLM_MODEL = "gemini-2.0-flash"

# Model for vision (image amount extraction)
VLM_MODEL = "gemini-2.0-flash"

# ── Recurrence inference ──────────────────────────────────────────────────────
# Minimum occurrences in history to consider an expense recurring
MIN_RECURRENCE_COUNT = 2

# Max variance in day-of-month (±days) to consider salary "monthly on same day"
SALARY_DAY_VARIANCE = 3

# Max coefficient of variation to treat an amount as "stable"
AMOUNT_CV_THRESHOLD = 0.15

# ── Binary search precision ───────────────────────────────────────────────────
BINARY_SEARCH_ITERATIONS = 50

# ── Output formatting ─────────────────────────────────────────────────────────
AMOUNT_DECIMAL_PLACES = 2
