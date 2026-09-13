"""
image_extractor.py — VLM-based extraction of amounts from financial document images.

Runs once at startup for all 16 images, caches results to disk.
Returns: dict mapping event_id → extracted_amount (float)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _USE_NEW_SDK = True
except ImportError:
    import google.generativeai as genai
    _USE_NEW_SDK = False
from PIL import Image

from config import GOOGLE_API_KEY, IMAGE_CACHE_FILE, MEDIA_DIR, VLM_MODEL


_EXTRACTION_PROMPT = """You are extracting a single numeric amount from a financial document image.

The document is one of: payroll letter, utility bill, grocery receipt, bank statement, or similar.

Instructions:
1. Find the primary monetary amount (the main charge, salary, or total).
2. Return ONLY the numeric value, no currency symbols, no commas, no text.
3. If the document shows multiple amounts, return the most prominent or final total.
4. If you cannot find any amount, return: UNKNOWN

Examples of correct responses:
25256.0
1037.52
873000
UNKNOWN
"""


def _configure_genai():
    if GOOGLE_API_KEY and not _USE_NEW_SDK:
        genai.configure(api_key=GOOGLE_API_KEY)


def _extract_amount_from_image(image_path: Path, model_name: str) -> Optional[float]:
    """Call VLM to extract numeric amount from a PNG image."""
    try:
        img = Image.open(image_path)
        if _USE_NEW_SDK:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model=model_name,
                contents=[_EXTRACTION_PROMPT, img],
            )
            text = response.text.strip()
        else:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([_EXTRACTION_PROMPT, img])
            text = response.text.strip()
        if text.upper() == "UNKNOWN" or not text:
            return None
        # Clean any accidental formatting
        cleaned = re.sub(r"[^\d.\-]", "", text)
        if cleaned:
            return float(cleaned)
        return None
    except Exception as e:
        print(f"  [image_extractor] Error processing {image_path.name}: {e}")
        return None


def load_image_amounts(image_records_by_user: dict, force_refresh: bool = False) -> dict[str, float]:
    """
    Extract amounts from all images in images.csv.
    Returns dict: event_id → amount (float)
    
    Results are cached to IMAGE_CACHE_FILE to avoid repeated VLM calls.
    image_records_by_user: dict from data_loader (user_id → list[ImageRecord])
    """
    _configure_genai()

    # Load cache if it exists
    cache: dict[str, float] = {}
    if not force_refresh and IMAGE_CACHE_FILE.exists():
        with open(IMAGE_CACHE_FILE, "r") as f:
            cache = json.load(f)
        print(f"[image_extractor] Loaded {len(cache)} cached image amounts")
        return cache

    # Collect all ImageRecord objects
    all_records = []
    for records in image_records_by_user.values():
        all_records.extend(records)

    print(f"[image_extractor] Extracting amounts from {len(all_records)} images...")

    for rec in all_records:
        image_path = MEDIA_DIR / f"{rec.image_id}.png"
        if not image_path.exists():
            print(f"  [image_extractor] Image file not found: {image_path}")
            continue

        print(f"  Processing {rec.image_id} -> event {rec.related_event_id}...")
        amount = _extract_amount_from_image(image_path, VLM_MODEL)
        if amount is not None:
            cache[rec.related_event_id] = amount
            print(f"    Extracted: {amount}")
        else:
            print(f"    Could not extract amount from {rec.image_id}")

    # Save cache
    IMAGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"[image_extractor] Done. Extracted {len(cache)} amounts. Saved to cache.")
    return cache


def apply_image_amounts(events: list, image_amounts: dict[str, float]) -> list:
    """
    Fill in missing amounts for events that were resolved from images.
    Mutates events in-place (also updates amount_home using the same conversion).
    """
    updated = 0
    for ev in events:
        if ev.amount is None and ev.event_id in image_amounts:
            extracted = image_amounts[ev.event_id]
            ev.amount = extracted
            # amount_home is same if no conversion needed (image amounts assumed in home_currency)
            # For safety, treat extracted amount as already in the event's currency
            ev.amount_home = extracted  # Will be re-converted if needed in financial_state
            updated += 1
    if updated:
        print(f"  [image_extractor] Applied {updated} image-extracted amounts to events")
    return events
