"""08_compare_curves.py -- Comparison 2: demand curves and price sensitivity (H4, H5).

plan.md section 5 (comparison 2), decision rules of section 6, choices logged in
section 7 (entry of 16 Nov 2026). Comparison 1 (H1-H3) is src/07_compare_twins.py.

Real side: READ from results/real_slopes.csv and results/real_demand_curves.csv
(the deliverables of src/04_real_slopes.py and src/03_real_curves.py), validated
on load, NOT recomputed. The real microdata data/derived/real_items_long.csv is
used only as the real half of the joint real+synthetic regressions.

Synthetic side: computed from data/derived/synthetic_answers.csv (22,000 rows =
200 twins x 10 items x 11 ladder prices), mirroring 04 exactly (same LPM, same
samples, B = 2000, seed 20261112 fresh stream, one twin resample per replication
drives all cells so the high-minus-low difference is paired).

Writes:
  results/compare2_curves_synthetic.csv   item x bracket x price shares (550 cells)
  results/compare2_slopes_synthetic.csv   44 rows, same format as real_slopes.csv
  results/compare2_regression_pooled.csv  pooled joint H4 model, both price controls
  results/compare2_h4_h5.csv              per-item real vs synthetic slopes and gaps
  results/compare2_hypotheses.csv         H4/H5 verdicts (section 6 rules)
  results/compare2_curve_item<item>.png   10 overlays (dollars)
  results/compare2_curve_pooled.png       pooled overlay (relative price)
  results/compare2_curve_brackets.png     pooled overlay, highest vs lowest bracket

Prints (AGENTS.md rule 2): every row count and reason; cell counts; B and seed;
the verdicts. The EXPLORATORY block is clearly labelled and used in no verdict.
"""

import argparse
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

# Command line (plan.md section 7, entry 17 Nov 2026, point (b)): --syn-csv selects
# the synthetic answers file (default: the Gemini run); --out-prefix is prepended to
# every results file name written below, figures included (default "" = unchanged
# file names). With no arguments the script must reproduce the existing results
# files byte-identically. Read-only real-side inputs are NOT prefixed.
parser = argparse.ArgumentParser(description="Comparison 2: demand curves and price sensitivity (H4, H5)")
parser.add_argument("--syn-csv", default=str(ROOT / "data/derived/synthetic_answers.csv"),
                    help="synthetic answers CSV (default: data/derived/synthetic_answers.csv)")
parser.add_argument("--out-prefix", default="",
                    help="prefix prepended to every results file name, figures included (default: none)")
parser.add_argument("--model", default="gemini-3.5-flash-lite",
                    help="expected single model in the synthetic answers file (default: gemini-3.5-flash-lite)")
ARGS = parser.parse_args()

SYN_CSV = Path(ARGS.syn_csv)
SAMPLE_CSV = ROOT / "data/derived/sample_200_items.csv"
LONG_PATH = ROOT / "data/derived/real_items_long.csv"
REAL_SLOPES_CSV = ROOT / "results/real_slopes.csv"
REAL_CURVES_CSV = ROOT / "results/real_demand_curves.csv"
RESULTS = ROOT / "results"
OUT_PREFIX = ARGS.out_prefix


def out(name: str) -> Path:
    """Results file name with the optional --out-prefix prepended."""
    return RESULTS / f"{OUT_PREFIX}{name}"

SEED = 20261112  # synthetic bootstrap stream only (AGENTS.md rule 5); mirrors 04
B = 2000         # same as src/04_real_slopes.py

YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"
BRACKETS = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]
LOW_BRACKET, HIGH_BRACKET = BRACKETS[0], BRACKETS[-1]
REF_ITEM = 9
MODEL_NAME = ARGS.model  # expected single model in the synthetic file (--model; default: the Gemini run)

# the 10 products (same table as src/04_real_slopes.py; plan.md section 3)
ITEMS = [
    (25, "Dairy Products", "Land O Lakes Salted Stick Butter, 16 oz, 4 sticks", 7.39, 10.56),
    (27, "Refrigerated Deli Meats", "Goya Cooked Ham 16 oz", 53.98, 59.98),
    (31, "Fresh Fruit", "Fresh Raspberries, 12 oz container", 0.95, 9.48),
    (9, "Cereal - Ready to Eat", "Cinnamon Toast Crunch, Family Size, 18.8 oz", 9.86, 9.86),
    (36, "Toilet Tissue", "Angel Soft Toilet Paper, 9 Mega Rolls", 0.00, 13.36),
    (26, "Yogurt - Refrigerated", "Chobani Non-Fat Greek Yogurt, Vanilla Blended 32 oz", 0.00, 11.16),
    (15, "Candy - Non-Chocolate", "HARIBO Goldbears Original Gummy Bears, 28.8oz Stand Up Bag", 12.96, 12.96),
    (23, "Refrigerated Bacon", "Oscar Mayer Fully Cooked Original Bacon, 2.52 oz Box", 4.27, 8.54),
    (38, "Batteries", "Duracell Coppertop AA Battery, Long Lasting Double A Batteries, 16 Pack", 25.55, 31.94),
    (10, "Cookies", "Little Debbie Oatmeal Creme Pies, 12 ct, 16.2 oz", 3.75, 5.36),
]
ITEM_NUMBERS = [row[0] for row in ITEMS]
LADDER_MAX = {row[0]: row[4] for row in ITEMS}
PRODUCT_OF = {row[0]: row[2] for row in ITEMS}
CATEGORY_OF = {row[0]: row[1] for row in ITEMS}
ITEMS_SORTED = sorted(ITEM_NUMBERS)
SAMPLES = ["all", "lowest", "highest", "high_minus_low"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def fit(X, yy):
    """OLS coefficients (intercept first) -- same estimator as src/04."""
    coef, _res, _rank, _s = np.linalg.lstsq(X, yy, rcond=None)
    return coef


def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 1. Load and validate
# ---------------------------------------------------------------------------
section("1. LOAD AND VALIDATE")

syn = pd.read_csv(SYN_CSV)
print(f"{SYN_CSV.name}: {len(syn)} rows loaded (expected 22,000 = 200 twins x 10 items x 11 prices)")
assert len(syn) == 22000, "synthetic_answers.csv must have 22,000 rows"
assert (syn.groupby(["new_id", "item"]).size() == 11).all(), "exactly 11 ladder rows per (twin, item) expected"
assert list(syn["model"].unique()) == [MODEL_NAME], f"unexpected synthetic model(s): {syn['model'].unique()}"
syn["yes"] = syn["answer"].map({YES_TEXT: 1, NO_TEXT: 0})
assert syn["yes"].notna().all(), "unexpected answer texts in synthetic_answers.csv"
syn["yes"] = syn["yes"].astype(int)

# ladder consistency against the real curves file (same 11 prices per item)
rc = pd.read_csv(REAL_CURVES_CSV)
print(f"real_demand_curves.csv: {len(rc)} rows loaded (expected 550 = 10 items x 5 brackets x 11 prices)")
assert len(rc) == 550, "real_demand_curves.csv must have 550 rows"
assert (rc.groupby("item").size() == 55).all(), "55 cells per item expected"
for it in ITEMS_SORTED:
    real_prices = set(np.round(rc.loc[rc["item"] == it, "price_dollars"], 2))
    syn_prices = set(np.round(syn.loc[syn["item"] == it, "price_dollars"], 2))
    assert real_prices == syn_prices, f"item {it}: synthetic ladder != real ladder"
    assert len(real_prices) == 11, f"item {it}: ladder must have 11 distinct prices"
lmax_row = syn["item"].map(LADDER_MAX).to_numpy(float)
assert np.allclose(syn["relative_price"], syn["price_dollars"] / lmax_row, rtol=0, atol=1e-12), (
    "synthetic relative_price != price_dollars / ladder_max"
)
assert (syn["free"] == (syn["price_dollars"] == 0.0).astype(float)).all(), "synthetic free indicator inconsistent"
syn["ladder_index"] = np.round(syn["price_dollars"] / (syn["item"].map(LADDER_MAX) / 10)).astype(int)
assert (syn.groupby("item")["ladder_index"].nunique() == 11).all(), "11 ladder positions per item expected"
assert set(syn["ladder_index"].unique()) == set(range(11)), "ladder_index must be 0..10"
print("validation OK: 22,000 rows, 11 per (twin, item), ladders identical to the real ones,")
print("  relative_price/free/ladder_index consistent with the item's ladder maximum")

# synthetic demographics: QID21 from the stratified sample (200 twins, 40 per bracket)
sample = pd.read_csv(SAMPLE_CSV)
bracket_map = sample.drop_duplicates("new_id").set_index("new_id")["QID21"]
assert len(bracket_map) == 200, "sample must have 200 twins"
twins_per_bracket = bracket_map.value_counts()
assert set(twins_per_bracket.index) == set(BRACKETS) and (twins_per_bracket == 40).all(), (
    f"40 twins per bracket expected, got {twins_per_bracket.to_dict()}"
)
syn = syn.merge(bracket_map.rename("QID21"), on="new_id", how="left", validate="many_to_one")
assert syn["QID21"].notna().all(), "every synthetic row must get its twin's QID21"
print(f"QID21 joined from sample_200_items.csv: {syn['new_id'].nunique()} twins, 40 per bracket")

# real slopes and CIs: read (not recomputed) from src/04's deliverable
rs = pd.read_csv(REAL_SLOPES_CSV)
assert len(rs) == 44, f"real_slopes.csv must have 44 rows, got {len(rs)}"
real_all = rs[(rs["sample"] == "all") & rs["item"].notna()].set_index("item")
real_gap = rs[(rs["sample"] == "high_minus_low") & rs["item"].notna()].set_index("item")
real_pooled_all = rs[rs["item"].isna() & (rs["sample"] == "all")].iloc[0]
real_pooled_gap = rs[rs["item"].isna() & (rs["sample"] == "high_minus_low")].iloc[0]
assert set(real_all.index.astype(int)) == set(ITEMS_SORTED), "real_slopes.csv items mismatch"
assert (real_all["slope_ci_low"] < 0).all() and (real_all["slope_ci_high"] < 0).all(), (
    "expected all real slopes to decline with CIs excluding 0 (src/04 outcome)"
)
print("real_slopes.csv: 44 rows read; all 10 real slopes negative with CIs excluding 0 (as in src/04)")

# real microdata for the joint regressions (same re-validation as src/04)
long = pd.read_csv(LONG_PATH, dtype={"pid": int, "QID21": str, "item": int, "answer": str})
assert len(long) == 19790, f"real_items_long.csv must have 19,790 rows, got {len(long)}"
assert long["pid"].nunique() == 1979 and (long.groupby("pid").size() == 10).all(), "1,979 pids x 10 rows expected"
assert np.all(np.diff(long["pid"].to_numpy()) >= 0), "rows must be sorted by pid"
assert set(long["answer"].unique()) == {YES_TEXT, NO_TEXT}, "unexpected real answer texts"
assert np.array_equal(long["yes"].to_numpy(), (long["answer"] == YES_TEXT).astype(int)), "yes inconsistent with answer"
ladder_arr_real = long["ladder_index"].to_numpy(int)
for it in ITEMS_SORTED:
    assert sorted(set(ladder_arr_real[long["item"].to_numpy() == it].tolist())) == list(range(11)), (
        f"item {it}: real ladder_index must be exactly 0..10"
    )
print(f"real_items_long.csv: {len(long)} rows, 1,979 respondents (real half of the joint regressions)")

# ---------------------------------------------------------------------------
# 2. Synthetic demand curves (item x bracket x price) + pooled curves
# ---------------------------------------------------------------------------
section("2. SYNTHETIC DEMAND CURVES (item x bracket x price)")

meta = rc.groupby("item")[["category", "product"]].first().reset_index()
cur = (
    syn.groupby(["item", "QID21", "price_dollars", "price_text", "relative_price", "free", "ladder_index"],
                 as_index=False)
    .agg(n_twins=("yes", "size"), n_yes=("yes", "sum"))
)
cur["share_yes"] = cur["n_yes"] / cur["n_twins"]
cur = cur.merge(meta, on="item")
cur["item"] = cur["item"].astype(int)
cur["free"] = cur["free"].astype(int)
bracket_order = {b: i for i, b in enumerate(BRACKETS)}
cur = cur.sort_values(["item", "QID21", "price_dollars"],
                      key=lambda s: s.map(bracket_order) if s.name == "QID21" else s)
cur = cur[["item", "category", "product", "QID21", "price_dollars", "price_text", "relative_price",
           "free", "ladder_index", "n_twins", "n_yes", "share_yes"]]
assert len(cur) == 550, f"synthetic curve table must have 550 cells, got {len(cur)}"
assert (cur["n_twins"] == 40).all(), "every (item, bracket, price) cell must have exactly 40 twins"
print(f"curve table: {len(cur)} cells = 10 items x 5 brackets x 11 prices")
print(f"cell counts (AGENTS.md rule 2): n_twins per cell min={cur['n_twins'].min()}, "
      f"median={cur['n_twins'].median():.0f}, max={cur['n_twins'].max()} (each twin answers every price once)")

pooled_real = rc.groupby("ladder_index", as_index=False).agg(n=("n_respondents", "sum"), n_yes=("n_yes", "sum"))
pooled_real["share_yes"] = pooled_real["n_yes"] / pooled_real["n"]
pooled_syn = syn.groupby("ladder_index", as_index=False).agg(n=("yes", "size"), n_yes=("yes", "sum"))
pooled_syn["share_yes"] = pooled_syn["n_yes"] / pooled_syn["n"]
print("\npooled curve by ladder position (real 19,790 obs vs synthetic 22,000 obs):")
pooled_print = pooled_real.merge(pooled_syn, on="ladder_index", suffixes=("_real", "_syn"))
print(pooled_print[["ladder_index", "n_real", "share_yes_real", "n_syn", "share_yes_syn"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# 3. Synthetic slopes, mirroring src/04 exactly (bootstrap by twin)
# ---------------------------------------------------------------------------
section(f"3. SYNTHETIC SLOPES (LPM yes ~ relative_price + free; pooled adds item FE, ref item {REF_ITEM})")

twins = sorted(syn["new_id"].unique())
n_twin = len(twins)
n_lad = 11
# twin x ladder rows (2,200), one column per item -- the synthetic analogue of 04's
# (pid, item) matrix: each (twin, ladder) row holds that twin's answer for each item
M_y = syn.pivot(index=["new_id", "ladder_index"], columns="item", values="yes")[ITEMS_SORTED]
M_rel = syn.pivot(index=["new_id", "ladder_index"], columns="item", values="relative_price")[ITEMS_SORTED]
M_free = syn.pivot(index=["new_id", "ladder_index"], columns="item", values="free")[ITEMS_SORTED]
assert M_y.shape == (n_twin * n_lad, 10) and M_y.notna().all().all(), "pivot must be complete"
assert list(M_y.index.get_level_values(0).unique()) == twins, "pivot rows must be in twin order"
yes_m = M_y.to_numpy(float)
rel_m = M_rel.to_numpy(float)
free_m = M_free.to_numpy(float)
code_t = np.where(bracket_map.reindex(twins) == LOW_BRACKET, 0,
                  np.where(bracket_map.reindex(twins) == HIGH_BRACKET, 1, 2))  # 0=low, 1=high, 2=middle
n_low, n_high = int((code_t == 0).sum()), int((code_t == 1).sum())
print(f"twin counts: lowest ({LOW_BRACKET}) n={n_low}, highest ({HIGH_BRACKET}) n={n_high}, all n={n_twin}")

X_item = [np.column_stack([np.ones(n_twin * n_lad), rel_m[:, j], free_m[:, j]]) for j in range(10)]
y_item = [yes_m[:, j] for j in range(10)]
item_row = np.tile(ITEMS_SORTED, n_twin * n_lad)  # ravel order: per (twin, ladder) block, items ascending
dummies = np.column_stack([(item_row == it).astype(float) for it in ITEMS_SORTED[1:]])
X_pool = np.column_stack([np.ones(n_twin * n_lad * 10), rel_m.ravel(), free_m.ravel(), dummies])
y_pool = yes_m.ravel()

low_rows, high_rows = np.repeat(code_t == 0, n_lad), np.repeat(code_t == 1, n_lad)
point = {}
for j, it in enumerate(ITEMS_SORTED):
    point[it] = {
        "all": fit(X_item[j], y_item[j]),
        "lowest": fit(X_item[j][low_rows], y_item[j][low_rows]),
        "highest": fit(X_item[j][high_rows], y_item[j][high_rows]),
    }
point["pooled"] = {
    "all": fit(X_pool, y_pool),
    "lowest": fit(X_pool[np.repeat(low_rows, 10)], y_pool[np.repeat(low_rows, 10)]),
    "highest": fit(X_pool[np.repeat(high_rows, 10)], y_pool[np.repeat(high_rows, 10)]),
}

print(f"Bootstrap by twin: B={B}, seed {SEED} (one resample per replication drives all cells; percentile CIs).")
t0 = time.time()
rng = np.random.default_rng(SEED)
keys = ITEMS_SORTED + ["pooled"]
draws = {k: {s: np.empty(B) for s in SAMPLES[:3]} for k in keys}
draws_free = {k: {s: np.empty(B) for s in SAMPLES[:3]} for k in keys}
for b in range(B):
    t_idx = rng.integers(0, n_twin, size=n_twin)
    m_low_b, m_high_b = code_t[t_idx] == 0, code_t[t_idx] == 1
    lad_rows = (t_idx[:, None] * n_lad + np.arange(n_lad)).ravel()      # rows of the (2200, 10) matrix
    low_lad, high_lad = np.repeat(m_low_b, n_lad), np.repeat(m_high_b, n_lad)
    for j in range(10):
        it = ITEMS_SORTED[j]
        Xb, yb = X_item[j][lad_rows], y_item[j][lad_rows]
        c = fit(Xb, yb)
        draws[it]["all"][b], draws_free[it]["all"][b] = c[1], c[2]
        cl = fit(Xb[low_lad], yb[low_lad])
        draws[it]["lowest"][b], draws_free[it]["lowest"][b] = cl[1], cl[2]
        ch = fit(Xb[high_lad], yb[high_lad])
        draws[it]["highest"][b], draws_free[it]["highest"][b] = ch[1], ch[2]
    pool_rows = (lad_rows[:, None] * 10 + np.arange(10)).ravel()       # rows of the 22,000-row pooled design
    Xb, yb = X_pool[pool_rows], y_pool[pool_rows]
    c = fit(Xb, yb)
    draws["pooled"]["all"][b], draws_free["pooled"]["all"][b] = c[1], c[2]
    rl = np.repeat(m_low_b, n_lad * 10)
    cl = fit(Xb[rl], yb[rl])
    draws["pooled"]["lowest"][b], draws_free["pooled"]["lowest"][b] = cl[1], cl[2]
    rh = np.repeat(m_high_b, n_lad * 10)
    ch = fit(Xb[rh], yb[rh])
    draws["pooled"]["highest"][b], draws_free["pooled"]["highest"][b] = ch[1], ch[2]
    if (b + 1) % 500 == 0:
        print(f"  bootstrap {b + 1}/{B} replications done")
for k in keys:
    draws[k]["high_minus_low"] = draws[k]["highest"] - draws[k]["lowest"]  # paired, same resample
print(f"Bootstrap done in {time.time() - t0:.0f} s.")

records = []
for key in keys:
    product = "Pooled (item fixed effects)" if key == "pooled" else PRODUCT_OF[key]
    for sname in SAMPLES:
        n_tw, n_obs = (np.nan, np.nan) if sname == "high_minus_low" else {
            "all": (n_twin, n_twin * (n_lad if key != "pooled" else n_lad * 10)),
            "lowest": (n_low, n_low * (n_lad if key != "pooled" else n_lad * 10)),
            "highest": (n_high, n_high * (n_lad if key != "pooled" else n_lad * 10)),
        }[sname]
        if sname == "high_minus_low":
            slope_pt = point[key]["highest"][1] - point[key]["lowest"][1]
            lo, hi = ci(draws[key]["high_minus_low"])
            records.append({"item": np.nan if key == "pooled" else key, "product": product, "sample": sname,
                            "n_twins": n_tw, "n_obs": n_obs, "slope": slope_pt, "slope_ci_low": lo,
                            "slope_ci_high": hi, "free_coef": np.nan, "free_ci_low": np.nan, "free_ci_high": np.nan})
        else:
            slope_pt, free_pt = point[key][sname][1], point[key][sname][2]
            lo, hi = ci(draws[key][sname])
            flo, fhi = ci(draws_free[key][sname])
            records.append({"item": np.nan if key == "pooled" else key, "product": product, "sample": sname,
                            "n_twins": n_tw, "n_obs": n_obs, "slope": slope_pt, "slope_ci_low": lo,
                            "slope_ci_high": hi, "free_coef": free_pt, "free_ci_low": flo, "free_ci_high": fhi})
syn_slopes = pd.DataFrame(records)
syn_all = syn_slopes[(syn_slopes["sample"] == "all") & syn_slopes["item"].notna()].set_index("item")
syn_gap = syn_slopes[(syn_slopes["sample"] == "high_minus_low") & syn_slopes["item"].notna()].set_index("item")
syn_pooled_all = syn_slopes[syn_slopes["item"].isna() & (syn_slopes["sample"] == "all")].iloc[0]
syn_pooled_gap = syn_slopes[syn_slopes["item"].isna() & (syn_slopes["sample"] == "high_minus_low")].iloc[0]

print("\nSynthetic price-sensitivity slopes (LPM: yes ~ relative_price + free;"
      " pooled adds item FE, ref item 9). Slope = dP(Yes) over relative price 0->1."
      f"\n95% CI: percentile bootstrap by twin, B={B}, seed {SEED}.\n")
print(f"{'item':>5}  {'sample':<14} {'n_twin':>6} {'n_obs':>6}  "
      f"{'slope':>7}  {'95% CI':<21}  {'free':>7}  {'95% CI':<21}")
for _, r in syn_slopes.iterrows():
    item_s = "pooled" if pd.isna(r["item"]) else str(r["item"])
    n_tw_s = "-" if pd.isna(r["n_twins"]) else str(r["n_twins"])
    n_obs_s = "-" if pd.isna(r["n_obs"]) else str(r["n_obs"])
    ci_slope = f"[{r['slope_ci_low']:+.3f}, {r['slope_ci_high']:+.3f}]"
    if pd.isna(r["free_coef"]):
        free_s, ci_free = "-", "-"
    else:
        free_s, ci_free = f"{r['free_coef']:+.3f}", f"[{r['free_ci_low']:+.3f}, {r['free_ci_high']:+.3f}]"
    print(f"{item_s:>5}  {r['sample']:<14} {n_tw_s:>6} {n_obs_s:>6}  "
          f"{r['slope']:+.3f}  {ci_slope:<21}  {free_s:>7}  {ci_free:<21}")

# ---------------------------------------------------------------------------
# 4. Joint real+synthetic regressions (pooled H4 model, plan.md section 5)
# ---------------------------------------------------------------------------
section("4. POOLED JOINT REGRESSION: purchase ~ source x price + item FE + source x free")

reg = pd.concat([
    long[["pid", "item", "yes", "relative_price", "price_dollars", "free"]].assign(source=0, cluster=long["pid"].astype(str)),
    syn[["new_id", "item", "yes", "relative_price", "price_dollars", "free"]].assign(source=1, cluster=syn["new_id"]),
], ignore_index=True)
assert len(reg) == 41790, f"joint table must have 41,790 rows, got {len(reg)}"
n_clusters = reg["cluster"].nunique()
print(f"joint table: {len(reg)} rows = 19,790 real (source=0, one obs per person x item)"
      f" + 22,000 synthetic (source=1, 11 obs per twin x item); {n_clusters} clusters (person = respondent / twin)")

FORMULA = "yes ~ source * {PRICE} + source * free + C(item, Treatment(reference=%d))" % REF_ITEM
pooled_fits = {}
for model, price_var in (("relative_price", "relative_price"), ("price_dollars", "price_dollars")):
    res = smf.ols(FORMULA.format(PRICE=price_var), data=reg).fit(
        cov_type="cluster", cov_kwds={"groups": reg["cluster"].to_numpy()}
    )
    coef_tab = pd.DataFrame({
        "term": res.params.index, "coef": res.params.to_numpy(), "se": res.bse.to_numpy(),
        "z": res.tvalues.to_numpy(), "p": res.pvalues.to_numpy(),
        "ci_low": res.conf_int().iloc[:, 0].to_numpy(), "ci_high": res.conf_int().iloc[:, 1].to_numpy(),
        "model": model,
    })
    pooled_fits[model] = coef_tab
regression_out = pd.concat([pooled_fits["relative_price"], pooled_fits["price_dollars"]], ignore_index=True)
regression_out = regression_out[["model", "term", "coef", "se", "z", "p", "ci_low", "ci_high"]]

main_fit = pooled_fits["relative_price"]
print("\nPOOLED JOINT MODEL (price = relative_price, SEs clustered by person):")
print(main_fit.drop(columns=["model"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
src_price = main_fit.loc[main_fit["term"].str.contains("source:", regex=False) &
                         ~main_fit["term"].str.contains("free", regex=False)].iloc[0]
dollar_fit = pooled_fits["price_dollars"]
src_price_d = dollar_fit.loc[dollar_fit["term"].str.contains("source:", regex=False) &
                             ~dollar_fit["term"].str.contains("free", regex=False)].iloc[0]
print(f"\npooled synthetic-minus-real slope difference (source:price): "
      f"relative = {src_price['coef']:+.4f} (SE {src_price['se']:.4f}, p = {src_price['p']:.4f}); "
      f"raw dollar price = {src_price_d['coef']:+.4f} (SE {src_price_d['se']:.4f}, p = {src_price_d['p']:.4f})")
print("ROBUSTNESS (section 5): the two price controls give the same sign for the pooled slope difference: "
      f"{np.sign(src_price['coef']) == np.sign(src_price_d['coef'])}")

# per-item joint regressions: descriptive synthetic-minus-real slope differences
joint_rows = []
for it in ITEMS_SORTED:
    sub = reg[reg["item"] == it]
    res = smf.ols("yes ~ source * relative_price + source * free", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["cluster"].to_numpy()}
    )
    term = [t for t in res.params.index if "source:" in t and "free" not in t][0]
    joint_rows.append({"item": it, "diff_coef": res.params[term], "diff_se": res.bse[term],
                       "diff_p": res.pvalues[term],
                       "diff_ci_low": res.conf_int().iloc[:, 0][term], "diff_ci_high": res.conf_int().iloc[:, 1][term]})
joint = pd.DataFrame(joint_rows).set_index("item")
print("\nper-item joint source x relative_price coefficients (synthetic minus real slope, descriptive):")
print(joint.to_string(float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# 5. H4 / H5 tables and verdicts (section 6 rules applied exactly)
# ---------------------------------------------------------------------------
section("5. H4 / H5 VERDICTS (section 6 decision rules)")

h4_rows, h5_rows = [], []
n_fail = 0
for it in ITEMS_SORTED:
    ra, sa = real_all.loc[it], syn_all.loc[it]
    outside = (sa["slope"] < ra["slope_ci_low"]) or (sa["slope"] > ra["slope_ci_high"])
    sign_opposite = sa["slope"] > 0  # all real slopes are negative (asserted in step 1)
    fail = bool(outside or sign_opposite)
    n_fail += int(fail)
    jd = joint.loc[it]
    h4_rows.append({
        "item": it, "product": PRODUCT_OF[it],
        "real_slope": ra["slope"], "real_slope_ci_low": ra["slope_ci_low"], "real_slope_ci_high": ra["slope_ci_high"],
        "synth_slope": sa["slope"], "synth_slope_ci_low": sa["slope_ci_low"], "synth_slope_ci_high": sa["slope_ci_high"],
        "outside_real_ci": outside, "sign_opposite": sign_opposite, "H4_fail": fail,
        "joint_diff_coef": jd["diff_coef"], "joint_diff_se": jd["diff_se"],
        "joint_diff_ci_low": jd["diff_ci_low"], "joint_diff_ci_high": jd["diff_ci_high"], "joint_diff_p": jd["diff_p"],
    })
    rg, sg = real_gap.loc[it], syn_gap.loc[it]
    testable = (rg["slope_ci_low"] > 0) or (rg["slope_ci_high"] < 0)
    h5_rows.append({
        "item": it, "product": PRODUCT_OF[it],
        "real_gap": rg["slope"], "real_gap_ci_low": rg["slope_ci_low"], "real_gap_ci_high": rg["slope_ci_high"],
        "synth_gap": sg["slope"], "synth_gap_ci_low": sg["slope_ci_low"], "synth_gap_ci_high": sg["slope_ci_high"],
        "sign_agree": np.sign(rg["slope"]) == np.sign(sg["slope"]), "testable_H5": testable,
    })
h4 = pd.DataFrame(h4_rows)
h5 = pd.DataFrame(h5_rows)
h4_rejected = n_fail >= 5
h4_verdict = "rejected (section 6 condition met)" if h4_rejected else "not rejected"

n_testable = int(h5["testable_H5"].sum())
n_agree = int(h5.loc[h5["testable_H5"], "sign_agree"].sum()) if n_testable else 0
if n_testable == 0:
    h5_verdict = "cannot test (0 testable products)"
else:
    share = n_agree / n_testable
    h5_verdict = ("rejected (section 6 condition met)" if share < 0.5 else "not rejected")

print(f"\nH4 (price sensitivity), per product: synthetic slope vs real slope 95% CI (src/04),"
      f" plus sign check (real slopes all decline):")
print(h4[["item", "product", "real_slope", "real_slope_ci_low", "real_slope_ci_high", "synth_slope",
          "synth_slope_ci_low", "synth_slope_ci_high", "outside_real_ci", "sign_opposite", "H4_fail"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n  products failing (outside CI or sign opposite): {n_fail} of 10 (rejected if >= 5)")
print(f"  pooled real slope {real_pooled_all['slope']:+.3f} [{real_pooled_all['slope_ci_low']:+.3f},"
      f" {real_pooled_all['slope_ci_high']:+.3f}] vs pooled synthetic slope {syn_pooled_all['slope']:+.3f}"
      f" [{syn_pooled_all['slope_ci_low']:+.3f}, {syn_pooled_all['slope_ci_high']:+.3f}]")
print(f"  H4 verdict: {h4_verdict}")

print(f"\nH5 (income gradient in price sensitivity), per product (real gaps imprecise, per src/04):")
print(h5[["item", "product", "real_gap", "real_gap_ci_low", "real_gap_ci_high", "synth_gap",
          "synth_gap_ci_low", "synth_gap_ci_high", "sign_agree", "testable_H5"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n  sign agreement: {n_agree}/{n_testable} testable products"
      + (f" = {n_agree / n_testable:.4f}" if n_testable else "")
      + " (rejected if the sign disagrees for half or more)")
print(f"  pooled real gap {real_pooled_gap['slope']:+.3f} [{real_pooled_gap['slope_ci_low']:+.3f},"
      f" {real_pooled_gap['slope_ci_high']:+.3f}] vs pooled synthetic gap {syn_pooled_gap['slope']:+.3f}"
      f" [{syn_pooled_gap['slope_ci_low']:+.3f}, {syn_pooled_gap['slope_ci_high']:+.3f}]")
print(f"  H5 verdict: {h5_verdict}")

hypotheses = pd.DataFrame([
    {"hypothesis": "H4 (price sensitivity)", "n_items": 10, "n_failing": n_fail,
     "real_pooled_slope": real_pooled_all["slope"],
     "real_pooled_slope_ci_low": real_pooled_all["slope_ci_low"],
     "real_pooled_slope_ci_high": real_pooled_all["slope_ci_high"],
     "synth_pooled_slope": syn_pooled_all["slope"],
     "synth_pooled_slope_ci_low": syn_pooled_all["slope_ci_low"],
     "synth_pooled_slope_ci_high": syn_pooled_all["slope_ci_high"],
     "n_testable": np.nan, "sign_agree_n": np.nan, "sign_agree_share": np.nan,
     "verdict": h4_verdict, "rejected": h4_rejected},
    {"hypothesis": "H5 (income gradient in price sensitivity)", "n_items": 10, "n_failing": np.nan,
     "real_pooled_slope": real_pooled_gap["slope"],
     "real_pooled_slope_ci_low": real_pooled_gap["slope_ci_low"],
     "real_pooled_slope_ci_high": real_pooled_gap["slope_ci_high"],
     "synth_pooled_slope": syn_pooled_gap["slope"],
     "synth_pooled_slope_ci_low": syn_pooled_gap["slope_ci_low"],
     "synth_pooled_slope_ci_high": syn_pooled_gap["slope_ci_high"],
     "n_testable": n_testable, "sign_agree_n": n_agree,
     "sign_agree_share": (n_agree / n_testable) if n_testable else np.nan,
     "verdict": h5_verdict, "rejected": (h5_verdict.startswith("rejected"))},
])

# ---------------------------------------------------------------------------
# 6. EXPLORATORY (NOT used in any verdict): ladder shape and the $0 jump
# ---------------------------------------------------------------------------
section("6. EXPLORATORY -- NOT USED IN ANY VERDICT")

ladders = syn.pivot(index=["new_id", "item"], columns="ladder_index", values="yes")
assert ladders.shape == (2000, 11), f"expected 2,000 ladders of 11 answers, got {ladders.shape}"
seq = ladders.to_numpy(int)
up = (np.diff(seq, axis=1) == 1).sum(axis=1)    # No -> Yes switches
down = (np.diff(seq, axis=1) == -1).sum(axis=1)  # Yes -> No switches
all_yes = seq.sum(axis=1) == 11
all_no = seq.sum(axis=1) == 0
single = (up == 0) & (down == 1)
other = ~(single | all_yes | all_no)
assert (single | all_yes | all_no | other).all() and (single.sum() + all_yes.sum() + all_no.sum() + other.sum()) == 2000
n = len(seq)
print(f"shape of the 2,000 synthetic (twin, item) ladders (11 answers ordered by price):")
print(f"  exactly one Yes->No switch (monotone single threshold): {single.sum():>5} = {single.mean():.4f}")
print(f"  all Yes (no switch, constant 1):                       {all_yes.sum():>5} = {all_yes.mean():.4f}")
print(f"  all No (no switch, constant 0):                        {all_no.sum():>5} = {all_no.mean():.4f}")
print(f"  remainder (non-monotone: a No->Yes switch or >=2 Yes->No): {other.sum():>5} = {other.mean():.4f}")

syn_free_share = syn.loc[syn["free"] == 1, "yes"].mean()
syn_free_n = int((syn["free"] == 1).sum())
real_free_mask = long["free"] == 1
real_free_share = long.loc[real_free_mask, "yes"].mean()
real_free_yes = int(long.loc[real_free_mask, "yes"].sum())
real_free_n = int(real_free_mask.sum())
print(f"\nYes share when the price shown was $0 (product free):")
print(f"  synthetic (each twin asked at $0 once per item): {syn_free_share:.4f} "
      f"({int(syn.loc[syn['free'] == 1, 'yes'].sum())}/{syn_free_n} obs; our 10 items)")
print(f"  real, our 10 items (one random price per respondent x item): "
      f"{real_free_share:.4f} ({real_free_yes}/{real_free_n} obs)")
print(f"  (the 81.6% anchor figure of 12 Nov was over ALL 40 items: 5866/7193 = 0.8158)")

# ---------------------------------------------------------------------------
# 7. Figures: overlays of real and synthetic curves (point estimates, no bands)
# ---------------------------------------------------------------------------
section("7. FIGURES (point-estimate overlays; uncertainty lives in the slope CIs)")

RESULTS.mkdir(parents=True, exist_ok=True)
for it in ITEMS_SORTED:
    prices = np.sort(rc.loc[rc["item"] == it, "price_dollars"].unique())
    g = rc[rc["item"] == it].groupby("price_dollars").agg(n=("n_respondents", "sum"), n_yes=("n_yes", "sum"))
    real_line = (g["n_yes"] / g["n"]).reindex(prices).to_numpy()
    gsyn = syn[syn["item"] == it].groupby("price_dollars")["yes"].agg(n="size", n_yes="sum")
    syn_line = (gsyn["n_yes"] / gsyn["n"]).reindex(prices).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    ax.plot(prices, real_line, marker="o", color="C0", linewidth=1.5, markersize=4,
            label=f"real, all 5 brackets pooled (1,979 respondents)")
    ax.plot(prices, syn_line, marker="s", color="C1", linewidth=1.5, markersize=4,
            label=f"synthetic, all 200 twins")
    ax.set_xticks(prices, ["$0" if p == 0 else f"${p:.2f}" for p in prices], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Price shown ($)")
    ax.set_ylabel("Share answering Yes")
    ax.set_title(textwrap.fill(f"Item {it}: {PRODUCT_OF[it]}", 70))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out(f"compare2_curve_item{it}.png"))
    plt.close(fig)

x = np.arange(11) / 10
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
ax.plot(x, pooled_real["share_yes"], marker="o", color="C0", linewidth=1.5, markersize=4,
        label="real, all 10 items pooled (19,790 obs)")
ax.plot(x, pooled_syn["share_yes"], marker="s", color="C1", linewidth=1.5, markersize=4,
        label="synthetic, all 10 items pooled (22,000 obs)")
ax.set_xticks(x, [f"{v:.1f}" for v in x])
ax.set_ylim(0, 1)
ax.set_xlabel("Relative price (price / item's ladder maximum)")
ax.set_ylabel("Share answering Yes")
ax.set_title("Pooled demand curve: real vs synthetic")
ax.grid(alpha=0.3)
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(out("compare2_curve_pooled.png"))
plt.close(fig)

rc_low = rc[rc["QID21"] == LOW_BRACKET].groupby("ladder_index").agg(n=("n_respondents", "sum"), n_yes=("n_yes", "sum"))
rc_high = rc[rc["QID21"] == HIGH_BRACKET].groupby("ladder_index").agg(n=("n_respondents", "sum"), n_yes=("n_yes", "sum"))
syn_low = syn[syn["QID21"] == LOW_BRACKET].groupby("ladder_index")["yes"].agg(n="size", n_yes="sum")
syn_high = syn[syn["QID21"] == HIGH_BRACKET].groupby("ladder_index")["yes"].agg(n="size", n_yes="sum")
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
ax.plot(x, rc_low["n_yes"] / rc_low["n"], marker="o", color="C0", linewidth=1.5, markersize=4,
        label=f"real, lowest: {LOW_BRACKET}")
ax.plot(x, rc_high["n_yes"] / rc_high["n"], marker="o", color="C0", linestyle="--", linewidth=1.5, markersize=4,
        label=f"real, highest: {HIGH_BRACKET}")
ax.plot(x, syn_low["n_yes"] / syn_low["n"], marker="s", color="C1", linewidth=1.5, markersize=4,
        label=f"synthetic, lowest: {LOW_BRACKET}")
ax.plot(x, syn_high["n_yes"] / syn_high["n"], marker="s", color="C1", linestyle="--", linewidth=1.5, markersize=4,
        label=f"synthetic, highest: {HIGH_BRACKET}")
ax.set_xticks(x, [f"{v:.1f}" for v in x])
ax.set_ylim(0, 1)
ax.set_xlabel("Relative price (price / item's ladder maximum)")
ax.set_ylabel("Share answering Yes")
ax.set_title("Pooled demand curve, highest vs lowest income bracket")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(out("compare2_curve_brackets.png"))
plt.close(fig)
print(f"wrote {out('compare2_curve_pooled.png')}, {out('compare2_curve_brackets.png')}"
      f" and 10 per-item overlays {OUT_PREFIX}compare2_curve_item<item>.png")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
section("8. SAVE")
h4_out = h4.copy(); h4_out.insert(0, "part", "H4")
h5_out = h5.copy(); h5_out.insert(0, "part", "H5")
h4_h5_out = pd.concat([h4_out, h5_out], ignore_index=True)
cur.to_csv(out("compare2_curves_synthetic.csv"), index=False, float_format="%.6f")
syn_slopes.to_csv(out("compare2_slopes_synthetic.csv"), index=False, float_format="%.6f", na_rep="")
h4_h5_out.to_csv(out("compare2_h4_h5.csv"), index=False, float_format="%.6f")
regression_out.to_csv(out("compare2_regression_pooled.csv"), index=False, float_format="%.6f")
hypotheses.to_csv(out("compare2_hypotheses.csv"), index=False, float_format="%.6f")
for f in ("compare2_curves_synthetic.csv", "compare2_slopes_synthetic.csv", "compare2_h4_h5.csv",
          "compare2_regression_pooled.csv", "compare2_hypotheses.csv"):
    print(f"wrote results/{OUT_PREFIX}{f}")
print(f"\nbootstrap: B = {B}, seed = {SEED} (numpy default_rng, fresh stream; real CIs are src/04's)")
print("done.")
