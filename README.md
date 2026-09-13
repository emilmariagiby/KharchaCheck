# KharchaCheck: Buy or Wait? 💰

An AI-powered, deterministic financial decision agent built for the **HackerRank Orchestrate Hackathon (September 2026)**.

KharchaCheck analyzes user financial context—spanning multi-currency accounts, recurring transactions, pending debits, payment plan options, multimodal receipts/statements, and communication messages—to deliver personalized, mathematically sound affordability decisions.

---

## 🏛 Architecture

KharchaCheck uses a 3-tier architecture:

```
                               ┌───────────────────────────┐
                               │  Multimodal Inputs        │
                               │  - Request text & dates   │
                               │  - User profile & prefs   │
                               │  - Historical/pending CSV │
                               │  - Messages & Image media │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │  Stage 1: Perception & Event Resolution               │
                 │  - Multimodal OCR / Image Amount Extraction           │
                 │  - Message Event Parsing (cancellations/salary rules) │
                 │  - Currency Normalization (cross-currency lookups)    │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │  Stage 2: Deterministic 90-Day Cash Flow Engine       │
                 │  - Daily Timeline Simulation [t = 0 to 90]            │
                 │  - Hard Constraint: Balance(t) >= Min_Balance         │
                 │  - Exact Safe Amount (Binary Search Optimization)     │
                 │  - Earliest Full Payment Date Projection              │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                 ┌───────────────────────────────────────────────────────┐
                 │  Stage 3: Strategy Selection & Ranking                │
                 │  - Full payment today                                 │
                 │  - Installment plans (from official payment options)  │
                 │  - Partial payment (2 tranches: today + safe date)    │
                 │  - Wait until earliest safe date                      │
                 │  - Spending reductions (max 3 flexible cuts)          │
                 │  - Lexicographic multi-objective ranking              │
                 └───────────────────────────┬───────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  output.csv (250 rows)    │
                               │  usage_report.md          │
                               └───────────────────────────┘
```

---

## 🚀 Key Capabilities

- **Strict Safety Guarantee**: Guarantees that projected daily balance never dips below the user's minimum required balance at any point over a 90-day horizon ($B(t) \ge M_u, \forall t \in [0, 90]$).
- **Multimodal Perception**: Extracts missing financial values from receipt/statement images with fallback caching.
- **Message Resolution**: Identifies confirmed salary adjustments, cancellations, and employment status from user messages without hallucinating unconfirmed income.
- **Deterministic Strategy Optimization**: Evaluates and lexicographically ranks:
  1. Completion before user's desired deadline
  2. Minimal total cash outflow (avoiding unnecessary financing fees)
  3. Earlier payment start dates
  4. Minimal spending adjustments / budget sacrifices
  5. Minimal number of payment tranches

---

## 📁 Repository Structure

```
.
├── code/
│   ├── config.py              # Configuration and API key loader
│   ├── data_loader.py         # Multi-CSV loader & currency normalizer
│   ├── financial_state.py     # Canonical financial timeline builder
│   ├── cashflow.py            # 90-day deterministic simulator
│   ├── affordability.py       # Binary search & earliest safe date solver
│   ├── strategy.py            # Plan generator & lexicographic ranker
│   ├── image_extractor.py     # Multimodal receipt amount parser
│   ├── message_resolver.py    # Regex & LLM message fact extractor
│   ├── explainer.py           # Natural language decision explainer
│   ├── usage_tracker.py       # Token & API cost auditor
│   └── main.py                # Pipeline orchestrator
├── dataset/                   # Problem inputs & sample data
├── evaluation/                # Usage reports & metrics
├── output.csv                 # Final 250-row submission output
├── log.txt                    # Execution log
└── README.md
```

---

## ⚡ Quick Start

### 1. Requirements & Setup
```bash
# Python 3.10+
pip install pandas google-genai
```

### 2. Configure Gemini API Key (Optional for LLM features)
Add your API key in `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(The mathematical simulation runs 100% deterministically even without an active API key).*

### 3. Run Pipeline
```bash
# Run on sample (25 requests)
python code/main.py --sample --output output_sample.csv

# Run full evaluation (250 requests)
python code/main.py --output output.csv
```

---

## 📊 Submission Deliverables
- `output.csv`: Complete decisions for all 250 requests matching the exact competition schema.
- `evaluation/usage_report.md`: Model usage, call counts, and cost breakdown.
- `log.txt`: Execution run trace and timing data.
