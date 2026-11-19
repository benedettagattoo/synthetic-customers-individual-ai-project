"""01_prepare.py — build the respondent x item long table for the 10 target products.

Real data only. Steps (all row counts printed, per AGENTS.md rule 2):
  1. Load wave-1/3 demographics (QID21 income) and wave-4 answers (QID9_k labels).
  2. Load wave-4 raw anonymized export (TWIN_ID, randDollarsString) and merge on pid.
  3. Validate randDollarsString (exactly 40 unique items 1..40); exclude malformed rows.
  4. For each of the 10 target items, look up its display position in each respondent's
     randDollarsString; the price is that position's price, the answer is QID9_{position}
     from wave4_response_label.csv (plan.md section 3, RESOLVED note).
  5. Verify each item's price ladder (11 distinct values) and save a long table.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

RAW = Path("data/raw")
DERIVED = Path("data/derived")
DERIVED.mkdir(parents=True, exist_ok=True)

# 10 target items: respondent 1's display positions 1, 5, ..., 37 (plan.md section 3)
TARGET_ITEMS = [25, 27, 31, 9, 36, 26, 15, 23, 38, 10]

INCOME_ORDER = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]
YES_LABEL = "Yes, I would purchase the product"
NO_LABEL = "No, I would not purchase the product"

# ---------------------------------------------------------------- load inputs
w13 = pd.read_csv(RAW / "wave1_3_response_label.csv", dtype=str, low_memory=False)
print(f"[load] wave1_3_response_label.csv: {len(w13)} rows, {w13['pid'].nunique()} unique pid")
demo = w13[["pid", "QID21"]].rename(columns={"QID21": "income_bracket"})

r4 = pd.read_csv(RAW / "wave4_response_label.csv", dtype=str, low_memory=False)
print(f"[load] wave4_response_label.csv: {len(r4)} rows, {r4['pid'].nunique()} unique pid")
qid9_cols = sorted((c for c in r4.columns if c.startswith("QID9_") and c.split("_")[1].isdigit()),
                   key=lambda c: int(c.split("_")[1]))
assert [int(c.split("_")[1]) for c in qid9_cols] == list(range(1, 41)), "QID9_1..40 must all exist"
answers_wide = r4[["pid"] + qid9_cols]

# check answer values are exactly the two expected labels
vals = set(pd.unique(answers_wide[qid9_cols].values.ravel()))
print(f"[check] distinct QID9 values across all 40 wave-4 columns: {sorted(vals)}")
assert vals == {YES_LABEL, NO_LABEL}, "unexpected answer labels"

raw = pd.read_csv(RAW / "wave_4_labels_anonymized.csv", dtype=str, keep_default_na=False,
                  low_memory=False)
raw = raw.iloc[2:].copy()  # rows 2-3 of the file are question text / ImportId metadata
raw["pid"] = raw["TWIN_ID"]
print(f"[load] wave_4_labels_anonymized.csv: {len(raw)} data rows, {raw['pid'].nunique()} unique TWIN_ID")

# ---------------------------------------------------------------- merge
merged = demo.merge(answers_wide, on="pid", how="inner", validate="one_to_one")
merged = merged.merge(raw[["pid", "randDollarsString"]], on="pid", how="inner", validate="one_to_one")
print(f"[merge] demographics + wave4 answers + raw strings: {len(merged)} rows "
      f"(expect 2058: 1:1 on pid, no rows lost)")

# ---------------------------------------------------------------- validate strings
PAIR_RE = r"(\d+):\$?([0-9]+(?:\.[0-9]+)?)"

def parse_string(s: str):
    """Return list of (position, item, price) in display order (positions = 1..40)."""
    return re.findall(PAIR_RE, s)

def is_valid(s: str) -> bool:
    pairs = parse_string(s)
    items = [int(i) for i, _ in pairs]
    return len(pairs) == 40 and sorted(set(items)) == list(range(1, 41))

merged["valid_string"] = merged["randDollarsString"].map(is_valid)
n_bad = int((~merged["valid_string"]).sum())
print(f"[filter] malformed randDollarsString (not exactly 40 unique items 1..40, "
      f"i.e. 41-44 pairs with a repeated item): {n_bad} rows EXCLUDED from all price-based analyses")
valid = merged[merged["valid_string"]].copy()
print(f"[filter] valid respondents retained: {len(valid)}")

# ---------------------------------------------------------------- long table
records = []
for row in valid.itertuples(index=False):
    for pos, (item, price) in enumerate(parse_string(row.randDollarsString), start=1):
        item = int(item)
        if item in TARGET_ITEMS:
            records.append({
                "pid": row.pid,
                "income_bracket": row.income_bracket,
                "item": item,
                "position": int(pos),
                "price": float(price),
                "yes": 1 if getattr(row, f"QID9_{pos}") == YES_LABEL else 0,
            })
long = pd.DataFrame(records)
print(f"[long] respondent x target-item rows: {len(long)} "
      f"(= {len(valid)} valid respondents x {len(TARGET_ITEMS)} items; "
      f"{len(long) - len(valid) * len(TARGET_ITEMS)} rows lost to parsing)")

# price ladder: each item must show exactly 11 distinct prices
for item in TARGET_ITEMS:
    prices = sorted(long.loc[long["item"] == item, "price"].unique())
    assert len(prices) == 11, f"item {item}: expected 11 ladder prices, found {len(prices)}"
    long.loc[long["item"] == item, "price_max"] = max(prices)
# dense rank of the 11 ladder points (0 = free, 10 = max)
for item in TARGET_ITEMS:
    m = long["item"] == item
    ladder = sorted(long.loc[m, "price"].unique())
    long.loc[m, "price_rank"] = long.loc[m, "price"].map({p: k for k, p in enumerate(ladder)}).astype(int)
long["price_share_of_max"] = long["price"] / long["price_max"]

long["income_bracket"] = pd.Categorical(long["income_bracket"], categories=INCOME_ORDER, ordered=True)

# sanity check: yes rate should fall with price (plan.md: verified monotone relationship)
check = long.groupby("price_rank")["yes"].mean()
print("[check] pooled yes rate by ladder rank (0=free ... 10=max):")
print(check.round(3).to_string())

cols = ["pid", "income_bracket", "item", "position", "price", "price_max",
        "price_rank", "price_share_of_max", "yes"]
long = long[cols].sort_values(["pid", "item"]).reset_index(drop=True)
print("[check] income brackets in the valid sample (respondents):")
print(long.drop_duplicates("pid")["income_bracket"].value_counts().to_string())
out = DERIVED / "wave4_target_items_long.csv"
long.to_csv(out, index=False)
print(f"[save] {out}: {len(long)} rows")
print(long.head(3).to_string(index=False))
