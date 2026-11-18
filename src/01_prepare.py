"""01_prepare.py — build the 200-person sample and the long respondent×item file
for the synthetic-customers study.

Follows plan.md section 3 (items 1–4, with the 11 Nov 2026 corrections):
  1. Merge demographics (wave 1–3 file) with the raw wave-4 file
     (wave_4_labels_anonymized.csv; TWIN_ID = pid; data start at row 4).
  1b. Exclude the 79 respondents whose randDollarsString is malformed
      (41–44 "item:price" pairs, one item repeated with two different prices).
  2. Replace pid with a new random ID (C001–C200) before anything is sent to a model.
  3. Sample 200 respondents, 40 per income bracket (QID21), from the 1979 valid rows.
  4. QID9_k are DISPLAY POSITIONS, not products. The 10 products are identified by
     ITEM NUMBER (the items at respondent 1's display positions 1, 5, …, 37:
     items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10). For each sampled respondent the
     price of an item is read from that respondent's randDollarsString and the
     answer from the column at that item's display position ({pos}_Q295).

Outputs:
  data/derived/sample_200_items.csv — 2000 rows (200 respondents × 10 items), long:
        new_id, demographics, item, category, product, position, price_text,
        price_dollars, relative_price, free, answer.
  data/derived/id_map.csv           — pid -> new_id link; NEVER sent to a model.
  The former wide output data/derived/sample_200.csv is obsolete (its QID9_k
  columns pair answers with the wrong prices) and is deleted if present.

This script prints NO Yes/No shares (overall or by group). The only counts printed
are row counts, exclusion reasons and demographic (income-bracket) counts.

Run:  uv run python src/01_prepare.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
SEED = 20261111  # fixed seed: stratified draw AND new-ID shuffle (AGENTS.md rule 5)

RAW = Path("data/raw")
DERIVED = Path("data/derived")

WAVE13_FILE = RAW / "wave1_3_response_label.csv"
RAW_WAVE4_FILE = RAW / "wave_4_labels_anonymized.csv"

DEMO_COLS = ["QID11", "QID12", "QID13", "QID14", "QID21"]
ANSWER_COLS = [f"{k}_Q295" for k in range(1, 41)]  # checked one by one below

N_PER_BRACKET = 40

# The 10 products: fixed rule from plan.md section 3 (every 4th display position of
# respondent 1's order: positions 1, 5, …, 37), applied to ITEM NUMBERS (plan.md
# correction of 11 Nov 2026). Columns: item no., category, product, catalog price
# (respondent 1's draw; a self-check only, not used for anyone else), ladder max M.
# Known caveat (plan.md section 7): the item 36 / 26 product assignment follows the
# catalog texts; it could not be checked against an independent item->product list.
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

YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"

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

# ---------------------------------------------------------------- merge (item 1)
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

# every price text in the valid rows must start with '$' and parse to a number
all_parse = valid["pairs"].map(lambda ps: all(price_dollars(p) == price_dollars(p)
                                              for _, p in ps)).all()
print(f"All prices in the valid rows parse to a dollar amount: {bool(all_parse)}.")
if not all_parse:
    raise SystemExit("Unparseable price text in a valid randDollarsString. Stopping.")

# ---------------------------------------------------------------- ladder-maximum check
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

# ---------------------------------------------------------------- bracket counts (valid rows)
print("Income brackets (QID21) among the valid rows, in file order:")
bracket_n = valid.groupby("QID21", sort=False).size()
for bracket, n in bracket_n.items():
    print(f"  {bracket}: {n}")
if (bracket_n < N_PER_BRACKET).any():
    raise SystemExit(f"A bracket has fewer than {N_PER_BRACKET} valid respondents. Stopping.")

# ---------------------------------------------------------------- stratified draw (item 3, valid rows only)
rng = np.random.default_rng(SEED)  # one seeded stream for draw AND shuffle
parts = []
for bracket, group in valid.groupby("QID21", sort=False):
    idx = rng.choice(group.index.to_numpy(), size=N_PER_BRACKET, replace=False)
    parts.append(valid.loc[idx])
sample = pd.concat(parts)  # rows stay grouped by income bracket
print(f"Sampled {len(sample)} respondents, {N_PER_BRACKET} per bracket, from the "
      f"{len(valid)} valid rows only. Counts per bracket in the sample:")
for bracket, n in sample.groupby("QID21", sort=False).size().items():
    print(f"  {bracket}: {n}")
print("Note: the draw pool is now the 1979 valid rows (was all 2058), so the sampled "
      "respondents - and thus which respondent gets which C-number - differ from the "
      "former run even though the seed and the scheme are unchanged.")

# ---------------------------------------------------------------- new IDs (seeded)
# Shuffle the 200 sampled respondents with the seeded RNG and assign C001..C200
# in that shuffled order, so the ID carries no information about bracket or order.
shuffled_pids = sample["pid"].to_numpy()[rng.permutation(len(sample))]
new_ids = [f"C{i:03d}" for i in range(1, len(sample) + 1)]
id_map = pd.DataFrame({"pid": shuffled_pids, "new_id": new_ids})
sample["new_id"] = sample["pid"].map(id_map.set_index("pid")["new_id"])

# ---------------------------------------------------------------- long respondent×item build (item 4)
rows = []
for _, resp in sample.iterrows():
    pairs = resp["pairs"]  # display order: pairs[k] is the pair at position k+1
    for item, category, product, _cat_price, ladder_max in ITEMS:
        pos = next(p for p, (i, _) in enumerate(pairs, start=1) if i == item)
        price_text = pairs[pos - 1][1]
        dollars = price_dollars(price_text)
        rows.append({
            "new_id": resp["new_id"],
            "QID11": resp["QID11"], "QID12": resp["QID12"], "QID13": resp["QID13"],
            "QID14": resp["QID14"], "QID21": resp["QID21"],
            "item": item, "category": category, "product": product,
            "position": pos, "price_text": price_text,
            "price_dollars": dollars,
            "relative_price": dollars / ladder_max,
            "free": int(dollars == 0.0),
            "answer": resp[f"{pos}_Q295"],
        })
long = pd.DataFrame(rows)
print(f"Built the long respondent×item table: {len(long)} rows "
      f"({len(sample)} respondents × {len(ITEMS)} items).")
if len(long) != len(sample) * len(ITEMS):
    raise SystemExit("Long table row count is not respondents × items. Stopping.")

bad_answers = set(long["answer"]) - {YES_TEXT, NO_TEXT}
print(f"Every 'answer' value is one of the two survey texts: {not bad_answers}.")
if bad_answers:
    raise SystemExit(f"Unexpected answer values found: {bad_answers}. Stopping.")
print(f"Respondents appear 10 times each: {(long.groupby('new_id').size() == len(ITEMS)).all()}.")

# ---------------------------------------------------------------- save
DERIVED.mkdir(parents=True, exist_ok=True)

id_map_path = DERIVED / "id_map.csv"
id_map.to_csv(id_map_path, index=False)

long_path = DERIVED / "sample_200_items.csv"
long_out = long[["new_id"] + DEMO_COLS + ["item", "category", "product", "position",
                                          "price_text", "price_dollars",
                                          "relative_price", "free", "answer"]]
long_out.to_csv(long_path, index=False)

# the former wide output is obsolete (answers paired with the wrong prices)
stale_path = DERIVED / "sample_200.csv"
if stale_path.exists():
    stale_path.unlink()
    print(f"Deleted the obsolete {stale_path} (its QID9_k columns paired answers "
          "with the wrong prices; superseded by sample_200_items.csv).")

print(f"Done. Wrote {id_map_path} ({len(id_map)} rows, pid -> new_id, never sent to "
      f"a model) and {long_path} ({len(long_out)} rows: new ID + 5 demographics + "
      f"item, category, product, display position, exact price text, price in "
      f"dollars, relative price, free indicator, real answer).")
