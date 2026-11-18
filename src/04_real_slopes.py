"""04_real_slopes.py — real demand-curve figures and price-sensitivity slopes.

Plan.md section 5, Comparison 2 (real side):
  (a) figures: one demand curve per item (share of "Yes" vs the price shown)
      for the lowest vs highest income bracket, plus one pooled curve against
      relative price with a 95% bootstrap band;
  (b) slopes: linear probability model  yes ~ relative_price + free
      (+ intercept; the pooled model adds 9 item fixed effects, reference
      item 9), fitted per item and pooled, overall (all 5 brackets) and for
      the lowest vs highest bracket, plus the highest-minus-lowest slope
      difference. 95% confidence intervals from a bootstrap BY RESPONDENT
      (plan.md section 5): B = 2000 resamples of the 1979 pids with
      replacement, seed 20261112 (numpy default_rng), percentile CIs
      (2.5/97.5). One resample per replication drives all cells, so the
      high-minus-low difference is paired (both slopes from the same
      resample). B and the seed were fixed before any slope was computed
      (approved in chat, 12 Nov 2026).

Reads ONLY data/derived/real_items_long.csv (output of 03_real_curves.py;
19,790 rows = 1979 valid respondents x 10 items, sorted by pid, item) and
re-validates it on load (row counts, pid ordering, answers, brackets,
relative_price/free/ladder consistency). No raw file is read, no model is
called, and the bootstrap stream is the only source of randomness.

Outputs (all made by this script):
  results/real_curve_item{item}.png  — 10 figures, lowest vs highest bracket
  results/real_curve_pooled.png      — pooled curve, relative price, band
  results/real_slopes.csv            — 44 rows of estimates (see below)

results/real_slopes.csv columns: item (empty for pooled rows), product,
sample ("all" / "lowest" / "highest" / "high_minus_low"), n_respondents,
n_obs (both empty for difference rows: the difference uses both subsamples),
slope, slope_ci_low, slope_ci_high, free_coef, free_ci_low, free_ci_high
(free columns empty for difference rows). Slopes are the change in P(Yes)
over the full relative-price range 0->1.

Prints: row counts (AGENTS.md rule 2) and the full estimates table; it also
flags the items whose high-minus-low CI covers 0 — per plan.md section 6
those cannot test the direction of H5.

Run:  uv run python src/04_real_slopes.py
"""

import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
SEED = 20261112  # bootstrap stream only (AGENTS.md rule 5); fixed 12 Nov 2026
B = 2000         # bootstrap replications, fixed before any slope was computed

LONG_PATH = Path("data/derived/real_items_long.csv")
RESULTS = Path("results")

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
LOW_BRACKET, HIGH_BRACKET = BRACKETS[0], BRACKETS[-1]

# the 10 products: same table as src/01_prepare.py and src/03_real_curves.py
# (plan.md section 3; the item 36/26 product-name caveat is in plan.md section 7)
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
PRODUCT_OF = {row[0]: row[2] for row in ITEMS}
ITEMS_SORTED = sorted(ITEM_NUMBERS)

SAMPLES = ["all", "lowest", "highest", "high_minus_low"]

# ---------------------------------------------------------------- load + re-validate
long = pd.read_csv(LONG_PATH, dtype={"pid": int, "QID21": str, "item": int,
                                     "category": str, "product": str,
                                     "position": int, "price_text": str,
                                     "answer": str})
n_pids, n_rows = 1979, 1979 * 10
print(f"Read {LONG_PATH}: {len(long)} rows.")
if len(long) != n_rows:
    raise SystemExit(f"Expected {n_rows} rows (1979 respondents x 10 items), "
                     f"found {len(long)}. Stopping.")

pid_arr = long["pid"].to_numpy()
if not np.all(np.diff(pid_arr) >= 0):
    raise SystemExit("Rows are not sorted by pid. Stopping.")
if long["pid"].nunique() != n_pids or not (long.groupby("pid").size() == 10).all():
    raise SystemExit("Expected 1979 pids with exactly 10 rows each. Stopping.")

item_arr = long["item"].to_numpy()
item_grid = item_arr.reshape(n_pids, 10)
expected_row = np.array(ITEMS_SORTED)
if not np.array_equal(item_grid, np.tile(expected_row, (n_pids, 1))):
    raise SystemExit("Within-pid item blocks do not match the 10 items in "
                     "ascending order. Stopping.")

q21 = long["QID21"].to_numpy()
bad_brackets = sorted(set(q21) - set(BRACKETS))
if bad_brackets:
    raise SystemExit(f"Unexpected QID21 values: {bad_brackets}. Stopping.")
q21_pid = q21.reshape(n_pids, 10)
if not (q21_pid == q21_pid[:, :1]).all():
    raise SystemExit("QID21 is not constant within pid. Stopping.")

answers = long["answer"].to_numpy()
bad_answers = set(answers) - {YES_TEXT, NO_TEXT}
if bad_answers:
    raise SystemExit(f"Unexpected answer values: {bad_answers}. Stopping.")
yes_arr = long["yes"].to_numpy()
if not np.array_equal(yes_arr, (answers == YES_TEXT).astype(int)):
    raise SystemExit("'yes' is not consistent with 'answer'. Stopping.")

dollars = long["price_dollars"].to_numpy(float)
rel = long["relative_price"].to_numpy(float)
free = long["free"].to_numpy(float)
ladder_arr = long["ladder_index"].to_numpy(int)

ladder_max_row = long["item"].map(LADDER_MAX).to_numpy(float)
if not np.allclose(rel, dollars / ladder_max_row, rtol=0, atol=1e-12):
    raise SystemExit("relative_price != price_dollars / ladder_max. Stopping.")
if not np.array_equal(free, (dollars == 0.0).astype(float)):
    raise SystemExit("free != 1[price_dollars == 0]. Stopping.")
for it in ITEMS_SORTED:
    idx = ladder_arr[item_arr == it]
    if sorted(set(idx.tolist())) != list(range(11)):
        raise SystemExit(f"Item {it}: ladder_index values are not exactly "
                         f"0..10. Stopping.")
print("Re-validation passed: 1979 pids x 10 items in pid,item order; answers, "
      "brackets, relative_price, free and ladder_index all consistent.")

# ---------------------------------------------------------------- arrays + design matrices
# rows are pid-major with 10 consecutive rows per pid, so reshape gives
# (pid, item) blocks; column j = item ITEMS_SORTED[j]
y = yes_arr.astype(float)
y_pid = y.reshape(n_pids, 10)
rel_pid = rel.reshape(n_pids, 10)
free_pid = free.reshape(n_pids, 10)

code_pid = np.where(q21_pid[:, 0] == LOW_BRACKET, 0,
                    np.where(q21_pid[:, 0] == HIGH_BRACKET, 1, 2))  # 0=low, 1=high, 2=middle
m_low_pid = code_pid == 0
m_high_pid = code_pid == 1
n_low, n_high = int(m_low_pid.sum()), int(m_high_pid.sum())
print(f"Bracket counts used here: lowest ({LOW_BRACKET}) n={n_low}, "
      f"highest ({HIGH_BRACKET}) n={n_high}, all brackets n={n_pids}.")

X_item = [np.column_stack([np.ones(n_pids), rel_pid[:, j], free_pid[:, j]])
          for j in range(10)]
y_item = [y_pid[:, j] for j in range(10)]

# pooled design: intercept + relative_price + free + 9 item dummies (ref = item 9)
dummies = np.column_stack([(item_arr == it).astype(float) for it in ITEMS_SORTED[1:]])
X_pool = np.column_stack([np.ones(n_rows), rel, free, dummies])
rm_low_full = np.repeat(m_low_pid, 10)
rm_high_full = np.repeat(m_high_pid, 10)


def fit(X, yy):
    """OLS coefficients (intercept first)."""
    coef, _res, _rank, _s = np.linalg.lstsq(X, yy, rcond=None)
    return coef


# ---------------------------------------------------------------- point estimates
point = {}
for j, it in enumerate(ITEMS_SORTED):
    point[it] = {
        "all": fit(X_item[j], y_item[j]),
        "lowest": fit(X_item[j][m_low_pid], y_item[j][m_low_pid]),
        "highest": fit(X_item[j][m_high_pid], y_item[j][m_high_pid]),
    }
point["pooled"] = {
    "all": fit(X_pool, y),
    "lowest": fit(X_pool[rm_low_full], y[rm_low_full]),
    "highest": fit(X_pool[rm_high_full], y[rm_high_full]),
}

# ---------------------------------------------------------------- bootstrap by respondent
print(f"Bootstrap by respondent: B={B}, seed {SEED} (one resample per "
      f"replication drives all cells; percentile CIs).")
t0 = time.time()
rng = np.random.default_rng(SEED)
keys = ITEMS_SORTED + ["pooled"]
draws = {k: {s: np.empty(B) for s in SAMPLES[:3]} for k in keys}       # slopes
draws_free = {k: {s: np.empty(B) for s in SAMPLES[:3]} for k in keys}  # free coefs
curve_draws = np.empty((B, 11))  # pooled point shares per replication

for b in range(B):
    pid_idx = rng.integers(0, n_pids, size=n_pids)
    m_low_b = code_pid[pid_idx] == 0
    m_high_b = code_pid[pid_idx] == 1
    for j in range(10):
        Xb, yb = X_item[j][pid_idx], y_item[j][pid_idx]
        it = ITEMS_SORTED[j]
        c = fit(Xb, yb)
        draws[it]["all"][b], draws_free[it]["all"][b] = c[1], c[2]
        cl = fit(Xb[m_low_b], yb[m_low_b])
        draws[it]["lowest"][b], draws_free[it]["lowest"][b] = cl[1], cl[2]
        ch = fit(Xb[m_high_b], yb[m_high_b])
        draws[it]["highest"][b], draws_free[it]["highest"][b] = ch[1], ch[2]
    rows = (pid_idx[:, None] * 10 + np.arange(10)).ravel()
    Xb, yb = X_pool[rows], y[rows]
    c = fit(Xb, yb)
    draws["pooled"]["all"][b], draws_free["pooled"]["all"][b] = c[1], c[2]
    cl = fit(Xb[rm_low := np.repeat(m_low_b, 10)], yb[rm_low])
    draws["pooled"]["lowest"][b], draws_free["pooled"]["lowest"][b] = cl[1], cl[2]
    ch = fit(Xb[rh := np.repeat(m_high_b, 10)], yb[rh])
    draws["pooled"]["highest"][b], draws_free["pooled"]["highest"][b] = ch[1], ch[2]
    lab = ladder_arr[rows]
    cnt = np.bincount(lab, minlength=11)
    share = np.bincount(lab, weights=yb, minlength=11) / np.maximum(cnt, 1)
    share[cnt == 0] = np.nan
    curve_draws[b] = share
    if (b + 1) % 250 == 0:
        print(f"  bootstrap {b + 1}/{B} replications done")

# paired high-minus-low difference draws (same resample for both slopes)
for k in keys:
    draws[k]["high_minus_low"] = draws[k]["highest"] - draws[k]["lowest"]
print(f"Bootstrap done in {time.time() - t0:.0f} s.")


def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- results table
def n_of(key, sample):
    """(n_respondents, n_obs) in the ORIGINAL data; NaN for difference rows."""
    if sample == "high_minus_low":
        return np.nan, np.nan
    n = {"all": n_pids, "lowest": n_low, "highest": n_high}[sample]
    return n, n * (1 if key != "pooled" else 10)


records = []
for key in ITEMS_SORTED + ["pooled"]:
    product = "Pooled (item fixed effects)" if key == "pooled" else PRODUCT_OF[key]
    for sample in SAMPLES:
        if sample == "high_minus_low":
            slope_pt = point[key]["highest"][1] - point[key]["lowest"][1]
            lo, hi = ci(draws[key]["high_minus_low"])
            n_resp, n_obs = n_of(key, sample)
            records.append({"item": np.nan if key == "pooled" else key,
                            "product": product, "sample": sample,
                            "n_respondents": n_resp, "n_obs": n_obs,
                            "slope": slope_pt, "slope_ci_low": lo,
                            "slope_ci_high": hi, "free_coef": np.nan,
                            "free_ci_low": np.nan, "free_ci_high": np.nan})
        else:
            slope_pt, free_pt = point[key][sample][1], point[key][sample][2]
            lo, hi = ci(draws[key][sample])
            flo, fhi = ci(draws_free[key][sample])
            n_resp, n_obs = n_of(key, sample)
            records.append({"item": np.nan if key == "pooled" else key,
                            "product": product, "sample": sample,
                            "n_respondents": n_resp, "n_obs": n_obs,
                            "slope": slope_pt, "slope_ci_low": lo,
                            "slope_ci_high": hi, "free_coef": free_pt,
                            "free_ci_low": flo, "free_ci_high": fhi})

out = pd.DataFrame(records)
out["item"] = out["item"].astype("Int64")
out["n_respondents"] = out["n_respondents"].astype("Int64")
out["n_obs"] = out["n_obs"].astype("Int64")

# ---------------------------------------------------------------- print the full table
print("\nReal price-sensitivity slopes (LPM: yes ~ relative_price + free;"
      " pooled adds item FE, ref item 9). Slope = dP(Yes) over relative price 0->1."
      f"\n95% CI: percentile bootstrap by respondent, B={B}, seed {SEED}.\n")
print(f"{'item':>5}  {'sample':<14} {'n_resp':>6} {'n_obs':>6}  "
      f"{'slope':>7}  {'95% CI':<21}  {'free':>7}  {'95% CI':<21}")
for _, r in out.iterrows():
    item_s = "pooled" if pd.isna(r["item"]) else str(r["item"])
    n_resp_s = "-" if pd.isna(r["n_respondents"]) else str(r["n_respondents"])
    n_obs_s = "-" if pd.isna(r["n_obs"]) else str(r["n_obs"])
    ci_slope = f"[{r['slope_ci_low']:+.3f}, {r['slope_ci_high']:+.3f}]"
    if pd.isna(r["free_coef"]):
        free_s, ci_free = "-", "-"
    else:
        free_s = f"{r['free_coef']:+.3f}"
        ci_free = f"[{r['free_ci_low']:+.3f}, {r['free_ci_high']:+.3f}]"
    print(f"{item_s:>5}  {r['sample']:<14} {n_resp_s:>6} {n_obs_s:>6}  "
          f"{r['slope']:+.3f}  {ci_slope:<21}  {free_s:>7}  {ci_free:<21}")

flagged = [str(r["item"]) for _, r in out.iterrows()
           if r["sample"] == "high_minus_low" and not pd.isna(r["item"])
           and r["slope_ci_low"] <= 0 <= r["slope_ci_high"]]
pooled_diff = out[(out["item"].isna()) & (out["sample"] == "high_minus_low")].iloc[0]
print(f"\nH5 direction not testable (high-minus-low CI covers 0): "
      f"items {', '.join(flagged) if flagged else 'none'}.")
print(f"Pooled high-minus-low slope difference: {pooled_diff['slope']:+.3f} "
      f"[{pooled_diff['slope_ci_low']:+.3f}, {pooled_diff['slope_ci_high']:+.3f}] "
      "(informational; H5 is judged per product).")

# ---------------------------------------------------------------- figures
RESULTS.mkdir(parents=True, exist_ok=True)

for it in ITEMS_SORTED:
    sub_item = long[long["item"] == it]
    prices = np.sort(sub_item["price_dollars"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    for bracket, label_prefix in [(LOW_BRACKET, "lowest"), (HIGH_BRACKET, "highest")]:
        sub = sub_item[sub_item["QID21"] == bracket]
        shares = sub.groupby("price_dollars")["yes"].mean().reindex(prices).to_numpy()
        ax.plot(prices, shares, marker="o", linewidth=1.5, markersize=4,
                label=f"{label_prefix}: {bracket} (n={sub['pid'].nunique()})")
    ax.set_xticks(prices,
                  ["$0" if p == 0 else f"${p:.2f}" for p in prices],
                  rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Price shown ($)")
    ax.set_ylabel("Share answering Yes")
    ax.set_title(textwrap.fill(f"Item {it}: {PRODUCT_OF[it]}", 70))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS / f"real_curve_item{it}.png")
    plt.close(fig)

x = np.arange(11) / 10
cnt_full = np.bincount(ladder_arr, minlength=11)
point_curve = np.bincount(ladder_arr, weights=y, minlength=11) / cnt_full
band_lo = np.nanpercentile(curve_draws, 2.5, axis=0)
band_hi = np.nanpercentile(curve_draws, 97.5, axis=0)
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
ax.fill_between(x, band_lo, band_hi, alpha=0.2, color="C0",
                label=f"95% bootstrap band (B={B} respondent resamples)")
ax.plot(x, point_curve, marker="o", color="C0",
        label="All 10 items, all 5 income brackets pooled")
ax.set_xticks(x, [f"{v:.1f}" for v in x])
ax.set_ylim(0, 1)
ax.set_xlabel("Relative price (price / item's ladder maximum)")
ax.set_ylabel("Share answering Yes")
ax.set_title("Pooled real demand curve (1979 respondents x 10 items)")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(RESULTS / "real_curve_pooled.png")
plt.close(fig)

# ---------------------------------------------------------------- save CSV
csv_path = RESULTS / "real_slopes.csv"
out.to_csv(csv_path, index=False, na_rep="")
print(f"\nDone. Wrote {csv_path} ({len(out)} rows), "
      f"{RESULTS / 'real_curve_pooled.png'} and 10 per-item figures "
      f"{RESULTS / 'real_curve_item<item>.png'} "
      f"(items {', '.join(str(i) for i in ITEMS_SORTED)}).")
