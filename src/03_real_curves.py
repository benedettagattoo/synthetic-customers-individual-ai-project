"""03_real_curves.py — build the REAL demand curves from the 1979 valid respondents.

Plan.md section 5, Comparison 2 (real side): for each of the 10 products
(identified by ITEM NUMBER — plan.md section 3, correction of 11 Nov 2026),
each of the 5 income brackets (QID21) and each of the item's 11 price points:
the number of respondents and the share of "Yes".

The respondent×item rows are built with the same position lookup as
src/01_prepare.py: for respondent r and item i, the position of item i in r's
randDollarsString gives the price shown (that "item:price" pair) and the
answer (raw column {pos}_Q295). The parse/validity logic and the hard checks
are duplicated from 01_prepare.py (importing it would re-run the stratified
sampling): malformed-string exclusion (plan.md section 3 item 1b), ladder
maxima, respondent-1 catalog prices, answer texts.

Outputs:
  results/real_demand_curves.csv   — one row per item × income bracket ×
        price point: item, category, product, QID21, price_dollars,
        price_text, relative_price, free, ladder_index, n_respondents,
        n_yes, n_no, share_yes.
  data/derived/real_items_long.csv — the long table, 1979 valid respondents
        × 10 items = 19,790 rows: pid, QID21, item, category, product,
        position, price_text, price_dollars, relative_price, free,
        ladder_index, answer, yes. pid stays local (data/derived; never sent
        to a model — plan.md section 3 item 2 requires anonymisation only
        before model calls).

Hard sanity anchor (fixed by the user, 12 Nov 2026): with the positional
pairing over ALL 40 items × 1979 valid respondents, the counts must be
exactly 5866 Yes of 7193 observations at $0 and 28935 Yes of 71967
observations above $0 (7193 + 71967 = 79,160 = 1979 × 40); the script stops
if they differ.

Printed: row counts and exclusion reasons (AGENTS.md rule 2), the per-item
price ladders, cell counts and min/median/max respondents per cell. No other
Yes/No shares — the shares live in the results CSV.

Run:  uv run python src/03_real_curves.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
RAW = Path("data/raw")
DERIVED = Path("data/derived")
RESULTS = Path("results")

WAVE13_FILE = RAW / "wave1_3_response_label.csv"
RAW_WAVE4_FILE = RAW / "wave_4_labels_anonymized.csv"

LONG_PATH = DERIVED / "real_items_long.csv"
CURVES_PATH = RESULTS / "real_demand_curves.csv"

DEMO_COLS = ["QID11", "QID12", "QID13", "QID14", "QID21"]  # only QID21 is used here
ANSWER_COLS = [f"{k}_Q295" for k in range(1, 41)]  # checked one by one below

YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"

# income brackets in natural order (exact CSV texts, data/README.md)
BRACKETS = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]

# the 10 products: same table as src/01_prepare.py (plan.md section 3; the
# known item 36/26 product-name caveat is recorded in plan.md section 7)
ITEMS = [
    # item, category, product, catalog_price, ladder_max
    (25, "Dairy Products", "Land O Lakes Salted Stick Butter, 16 oz, 4 sticks", 7.39, 10.56),
    (27, "Refrigerated Deli Meats", "Goya Cooked Ham 16 oz", 53.98, 59.98),
    (31, "Fresh Fruit", "Fresh Raspberries, 12 oz container", 0.95, 9.48),
    (9, "Cereal - Ready to Eat", "Cinnamon Toast Crunch, Family Size, 18.8 oz", 9.86, 9.86),
    (36, "Toilet Tissue", "Angel Soft Toilet Paper, 9 Mega Rolls", 0.00, 13.36),
    (26, "Yogurt - Refrigerated", "Chobani Non-Fat Greek Yogurt, Vanilla, 32 oz", 0.00, 11.16),
    (15, "Candy - Non-Chocolate", "HARIBO Goldbears, 28.8 oz bag", 12.96, 12.96),
    (23, "Refrigerated Bacon", "Oscar Mayer Fully Cooked Original Bacon, 2.52 oz", 4.27, 8.54),
    (38, "Batteries", "Duracell Coppertop AA, 16 pack", 25.55, 31.94),
    (10, "Cookies", "Little Debbie Oatmeal Creme Pies, 12 ct", 3.75, 5.36),
]
ITEM_NUMBERS = [row[0] for row in ITEMS]
LADDER_MAX = {row[0]: row[4] for row in ITEMS}

# hard sanity anchor (user, 12 Nov 2026): positional pairing, all 40 items,
# 1979 valid respondents
ANCHOR = {"yes_at_0": 5866, "n_at_0": 7193,
          "yes_above_0": 28935, "n_above_0": 71967}

# ---------------------------------------------------------------- column check
wave13_header = pd.read_csv(WAVE13_FILE, nrows=0).columns.tolist()
raw4_header = pd.read_csv(RAW_WAVE4_FILE, nrows=0).columns.tolist()

missing_13 = [c for c in ["pid"] + DEMO_COLS if c not in wave13_header]
missing_4 = [c for c in ["TWIN_ID", "randDollarsString"] + ANSWER_COLS if c not in raw4_header]
if missing_13 or missing_4:
    raise SystemExit(
        f"Expected columns missing. In {WAVE13_FILE.name}: {missing_13 or 'none'}. "
        f"In {RAW_WAVE4_FILE.name}: {missing_4 or 'none'}. Stopping."
    )
print("Column check: pid + 5 demographics in the wave1-3 file; TWIN_ID, "
      "randDollarsString and all 40 {k}_Q295 columns in the raw wave-4 file.")

# ---------------------------------------------------------------- read
wave13 = pd.read_csv(WAVE13_FILE, usecols=["pid"] + DEMO_COLS, dtype=str)
raw4 = pd.read_csv(RAW_WAVE4_FILE, skiprows=[1, 2], dtype=str,  # rows 2-3 = question text / ImportId
                   usecols=["TWIN_ID", "randDollarsString"] + ANSWER_COLS)
raw4 = raw4.rename(columns={"TWIN_ID": "pid"})
print(f"Rows before merge: {WAVE13_FILE.name} = {len(wave13)}, "
      f"{RAW_WAVE4_FILE.name} = {len(raw4)} (data start at row 4 of the raw file).")

# ---------------------------------------------------------------- merge (plan.md section 3, item 1)
same_pids = set(wave13["pid"]) == set(raw4["pid"])
print(f"pid sets identical between the two files: {same_pids}.")
if not same_pids:
    raise SystemExit("pid sets differ; a 1:1 merge would lose rows. Stopping.")

merged = wave13.merge(raw4, on="pid", how="inner", validate="1:1")
print(f"Rows after 1:1 inner merge on pid: {len(merged)} "
      "(expected 2058 per data/README.md; no rows lost).")

# ---------------------------------------------------------------- missing check
check_cols = DEMO_COLS + ["randDollarsString"] + ANSWER_COLS
n_missing = int(merged[check_cols].isna().sum().sum()
                + (merged[check_cols] == "").sum().sum())
print(f"Missing/empty values in the {len(check_cols)} used columns: {n_missing} "
      "(data/README.md documents 0).")
if n_missing != 0:
    raise SystemExit("Missing values found where 0 were documented. Stopping "
                     "rather than silently dropping rows.")

# ---------------------------------------------------------------- parse randDollarsString (item 1b)
# same functions as src/01_prepare.py (duplicated on purpose, see docstring)
def parse_rand_dollars(s):
    """Return the list of (item_number, price_text) pairs in display order.

    Format per data/README.md: a stray leading `""` field, then 40 'item:price'
    pairs separated by ';' (e.g. '`""`;25:$7.39;36:$0 (product is free);...').
    """
    fields = [f for f in str(s).split(";") if f.strip() not in ("", '""', '`""')]
    pairs = []
    for f in fields:
        m = re.match(r"^\s*(\d+)\s*:\s*(\$.*?)\s*$", f)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
    return pairs


def price_dollars(price_text):
    m = re.search(r"\$([\d,]+(?:\.\d+)?)", price_text)
    return float(m.group(1).replace(",", "")) if m else float("nan")


parsed = merged["randDollarsString"].map(parse_rand_dollars)
n_pairs = parsed.map(len)
items_ok = parsed.map(lambda ps: sorted(i for i, _ in ps) == list(range(1, 41)))
valid_mask = (n_pairs == 40) & items_ok

n_excluded = int((~valid_mask).sum())
pair_counts = sorted(n_pairs[~valid_mask].unique().tolist())
repeats = parsed[~valid_mask].map(lambda ps: len(ps) != len(set(i for i, _ in ps)))
print(f"Malformed randDollarsString rows excluded: {n_excluded} of {len(merged)} "
      f"(reason: {pair_counts} 'item:price' pairs instead of 40, one item repeated "
      f"with two different prices; repeated item present in {int(repeats.sum())} "
      f"of {n_excluded} excluded rows; plan.md section 3 item 1b).")
print(f"Valid rows kept for every analysis that uses prices: {int(valid_mask.sum())}.")

valid = merged[valid_mask.to_numpy()].copy()
valid.loc[:, "pairs"] = parsed[valid_mask.to_numpy()]

# every price text in the valid rows must parse to a dollar amount (all 40 items)
all_parse = valid["pairs"].map(
    lambda ps: all(not np.isnan(price_dollars(p)) for _, p in ps)).all()
print(f"All prices in the valid rows parse to a dollar amount: {bool(all_parse)}.")
if not all_parse:
    raise SystemExit("Unparseable price text in a valid randDollarsString. Stopping.")

# ---------------------------------------------------------------- ladder-maximum check (10 items)
observed_max = {}
for _, row in valid.iterrows():
    for item_no, ptxt in row["pairs"]:
        observed_max[item_no] = max(observed_max.get(item_no, 0.0), price_dollars(ptxt))
max_mismatch = [(item_no, ladder_max, observed_max.get(item_no))
                for item_no, _cat, _prod, _cp, ladder_max in ITEMS
                if abs(observed_max.get(item_no, -1.0) - ladder_max) > 0.005]
if max_mismatch:
    raise SystemExit(f"Observed per-item price maximum differs from the plan.md "
                     f"section 3 ladder maxima: {max_mismatch}. Stopping.")
print("Ladder maxima: observed maximum per item among the valid rows equals the "
      "plan.md section 3 ladder maxima for all 10 items.")

# ---------------------------------------------------------------- respondent-1 self-check
r1 = valid[valid["pid"] == "1"].iloc[0]["pairs"]
r1_bad = []
for item_no, _category, _product, catalog_price, _m in ITEMS:
    pos = next(p for p, (i, _ptxt) in enumerate(r1, start=1) if i == item_no)
    r1_price = price_dollars(r1[pos - 1][1])
    if abs(r1_price - catalog_price) > 0.005:
        r1_bad.append((item_no, pos, r1_price, catalog_price))
if r1_bad:
    raise SystemExit(f"Respondent 1's price at an item's display position does not "
                     f"equal the catalog price (item, position, seen, catalog): "
                     f"{r1_bad}. Stopping.")
print("Self-check: respondent 1's price at each of the 10 items' display positions "
      "equals the plan.md section 3 catalog price.")

# ---------------------------------------------------------------- income brackets (counts only)
bad_brackets = sorted(set(valid["QID21"]) - set(BRACKETS))
if bad_brackets:
    raise SystemExit(f"Unexpected QID21 values in the valid rows: {bad_brackets}. "
                     f"Expected exactly the 5 data/README.md texts. Stopping.")
print("Income brackets (QID21) among the valid rows:")
for bracket in BRACKETS:
    print(f"  {bracket}: {int((valid['QID21'] == bracket).sum())}")

# ---------------------------------------------------------------- long respondent×item build (all 1979)
rows = []
for _, resp in valid.iterrows():
    pairs = resp["pairs"]  # display order: pairs[k] is the pair at position k+1
    for item, category, product, _cat_price, ladder_max in ITEMS:
        pos = next(p for p, (i, _) in enumerate(pairs, start=1) if i == item)
        price_text = pairs[pos - 1][1]
        dollars = price_dollars(price_text)
        step = ladder_max / 10
        ladder_index = int(round(dollars / step))
        if abs(dollars / step - ladder_index) > 0.02:
            raise SystemExit(f"pid {resp['pid']}, item {item}: price {price_text} "
                             f"(${dollars:.2f}) is not on the item's 11-point ladder "
                             f"(step ${step:.4f}). Stopping.")
        answer = resp[f"{pos}_Q295"]
        rows.append({
            "pid": int(resp["pid"]),
            "QID21": resp["QID21"],
            "item": item, "category": category, "product": product,
            "position": pos, "price_text": price_text,
            "price_dollars": dollars,
            "relative_price": dollars / ladder_max,
            "free": int(dollars == 0.0),
            "ladder_index": ladder_index,
            "answer": answer,
            "yes": int(answer == YES_TEXT),
        })
long = pd.DataFrame(rows)
print(f"Built the long respondent×item table: {len(long)} rows "
      f"({len(valid)} valid respondents × {len(ITEMS)} items).")
if len(long) != len(valid) * len(ITEMS):
    raise SystemExit("Long table row count is not valid respondents × items. Stopping.")

bad_answers = set(long["answer"]) - {YES_TEXT, NO_TEXT}
print(f"Every 'answer' value is one of the two survey texts: {not bad_answers}.")
if bad_answers:
    raise SystemExit(f"Unexpected answer values found: {bad_answers}. Stopping.")
print(f"Respondents appear 10 times each: {(long.groupby('pid').size() == len(ITEMS)).all()}.")

# ---------------------------------------------------------------- ladder checks (10 items)
print("Observed price ladders (distinct prices in randDollarsString of the valid rows):")
ladder_prices = {}
for item_no, _category, _product, _cp, ladder_max in ITEMS:
    sub = long[long["item"] == item_no]
    dollars = sorted(sub["price_dollars"].unique().tolist())
    if len(dollars) != 11:
        raise SystemExit(f"Item {item_no}: {len(dollars)} distinct prices among the "
                         f"valid rows, expected exactly 11. Stopping.")
    per_price_texts = sub.groupby("price_dollars")["price_text"].nunique()
    if (per_price_texts != 1).any() or sub["price_text"].nunique() != 11:
        raise SystemExit(f"Item {item_no}: price texts do not map one-to-one to the "
                         f"11 dollar prices. Stopping.")
    idx = [int(sub.loc[sub["price_dollars"] == d, "ladder_index"].iloc[0]) for d in dollars]
    if idx != list(range(11)):
        raise SystemExit(f"Item {item_no}: ladder indices not 0..10 in price order: "
                         f"{idx}. Stopping.")
    ladder_prices[item_no] = dollars
    text_by_price = sub.groupby("price_dollars")["price_text"].first()
    print(f"  item {item_no:>2} ({len(dollars)} points): "
          + ", ".join(text_by_price[d] for d in dollars))

# ---------------------------------------------------------------- hard sanity anchor (all 40 items)
n_at_0 = yes_at_0 = n_above = yes_above = 0
for _, resp in valid.iterrows():
    ans_all = [resp[f"{pos}_Q295"] for pos in range(1, 41)]
    bad = set(ans_all) - {YES_TEXT, NO_TEXT}
    if bad:
        raise SystemExit(f"pid {resp['pid']}: unexpected answer text(s) {bad} among "
                         f"the 40 wave-4 answers. Stopping.")
    for pos, (_item_no, ptxt) in enumerate(resp["pairs"], start=1):
        dollars = price_dollars(ptxt)
        is_yes = ans_all[pos - 1] == YES_TEXT
        if dollars == 0.0:
            n_at_0 += 1
            yes_at_0 += int(is_yes)
        else:
            n_above += 1
            yes_above += int(is_yes)
print(f"Anchor check (all 40 items, {len(valid)} valid respondents, positional "
      f"pairing): Yes at $0 = {yes_at_0} of {n_at_0}; Yes above $0 = {yes_above} "
      f"of {n_above}; total observations = {n_at_0 + n_above}.")
observed_anchor = (yes_at_0, n_at_0, yes_above, n_above)
expected_anchor = (ANCHOR["yes_at_0"], ANCHOR["n_at_0"],
                   ANCHOR["yes_above_0"], ANCHOR["n_above_0"])
if observed_anchor != expected_anchor:
    raise SystemExit(f"Anchor mismatch: observed (yes_at_0, n_at_0, yes_above_0, "
                     f"n_above_0) = {observed_anchor}, expected {expected_anchor} "
                     f"(user-fixed counts, 12 Nov 2026). Stopping.")
print("Anchor matches the user-fixed counts exactly.")

# ---------------------------------------------------------------- aggregate to item × bracket × price
key_cols = ["item", "category", "product", "QID21", "price_dollars",
            "price_text", "relative_price", "free", "ladder_index"]
agg = (long.groupby(key_cols, as_index=False)
           .agg(n_respondents=("yes", "size"), n_yes=("yes", "sum")))
agg["n_no"] = agg["n_respondents"] - agg["n_yes"]
agg["share_yes"] = agg["n_yes"] / agg["n_respondents"]

if int(agg["n_respondents"].sum()) != len(long):
    raise SystemExit("Aggregated cell counts do not sum to the long-table rows. "
                     "Stopping.")

bracket_rank = {b: i for i, b in enumerate(BRACKETS)}
agg["_bracket_rank"] = agg["QID21"].map(bracket_rank)
agg = (agg.sort_values(["item", "_bracket_rank", "price_dollars"], kind="mergesort")
          .drop(columns="_bracket_rank")
          .reset_index(drop=True))

expected_cells = len(ITEMS) * len(BRACKETS) * 11
full_grid = [(it, b, d) for it in ITEM_NUMBERS for b in BRACKETS
             for d in ladder_prices[it]]
observed_cells = set(zip(agg["item"], agg["QID21"], agg["price_dollars"]))
missing_cells = [c for c in full_grid if c not in observed_cells]
print(f"Cells: {len(agg)} of the expected {expected_cells} "
      f"({len(ITEMS)} items × {len(BRACKETS)} brackets × 11 prices).")
if missing_cells:
    print(f"Empty cells (no valid respondent drew that price in that bracket): "
          f"{len(missing_cells)}: {missing_cells}")
else:
    print("No empty cells.")
cell_n = agg["n_respondents"]
print(f"Respondents per cell: min {cell_n.min()}, median {cell_n.median():.1f}, "
      f"max {cell_n.max()} (plan.md section 5 expects ≈36).")

# ---------------------------------------------------------------- save
DERIVED.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

long_out = long[["pid", "QID21", "item", "category", "product", "position",
                 "price_text", "price_dollars", "relative_price", "free",
                 "ladder_index", "answer", "yes"]]
long_out = long_out.sort_values(["pid", "item"], kind="mergesort").reset_index(drop=True)
long_out.to_csv(LONG_PATH, index=False)

agg_out = agg[["item", "category", "product", "QID21", "price_dollars",
               "price_text", "relative_price", "free", "ladder_index",
               "n_respondents", "n_yes", "n_no", "share_yes"]]
agg_out.to_csv(CURVES_PATH, index=False)

print(f"Done. Wrote {LONG_PATH} ({len(long_out)} rows: pid, QID21, item, category, "
      f"product, display position, exact price text, price in dollars, relative "
      f"price, free indicator, ladder index, real answer, yes 0/1) and "
      f"{CURVES_PATH} ({len(agg_out)} rows: item × income bracket × price point "
      f"with n_respondents, n_yes, n_no and share_yes).")
