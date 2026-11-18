"""07_compare_twins.py -- Comparison 1: twin by twin at the real twin's price (H1, H2, H3).

plan.md section 5 (comparison 1), decision rules of section 6, choices logged in
section 7 (entry of 16 Nov 2026). Comparison 2 (demand curves, H4/H5) is NOT here.

Reads (read-only):
  data/derived/sample_200_items.csv   2,000 rows: real twin x item answers (wave 4),
                                      stratified sample, 40 twins per income bracket
  data/derived/synthetic_answers.csv  22,000 rows: 200 twins x 10 items x 11 ladder
                                      prices, model gemini-3.5-flash-lite
  data/derived/real_items_long.csv    19,790 rows: all 1,979 valid respondents x 10
                                      items -- used ONLY by the labelled exploratory
                                      side analysis, never in a verdict

Writes:
  results/compare1_shares.csv       Yes share per item x income bracket, real vs synthetic
  results/compare1_gaps.csv         per item: highest-minus-lowest gaps, bootstrap CIs,
                                    H2 ratios, testability flag, exploratory 1,979 gap
  results/compare1_regression.csv   both LPMs (relative price / raw dollar price)
  results/compare1_hypotheses.csv   H1/H2/H3 verdicts (section 6 rules)

Prints (AGENTS.md rule 2): every row count and its reason; B and seed; the verdicts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]

# Command line (plan.md section 7, entry 17 Nov 2026, point (b)): --syn-csv selects
# the synthetic answers file (default: the Gemini run); --out-prefix is prepended to
# every results file name written below (default "" = unchanged file names). With no
# arguments the script must reproduce the existing results files byte-identically.
parser = argparse.ArgumentParser(description="Comparison 1: twin by twin at the real twin's price (H1, H2, H3)")
parser.add_argument("--syn-csv", default=str(ROOT / "data/derived/synthetic_answers.csv"),
                    help="synthetic answers CSV (default: data/derived/synthetic_answers.csv)")
parser.add_argument("--out-prefix", default="",
                    help="prefix prepended to every results file name (default: none)")
parser.add_argument("--model", default="gemini-3.5-flash-lite",
                    help="expected single model in the synthetic answers file (default: gemini-3.5-flash-lite)")
ARGS = parser.parse_args()

SAMPLE_CSV = ROOT / "data/derived/sample_200_items.csv"
SYN_CSV = Path(ARGS.syn_csv)
REAL_LONG_CSV = ROOT / "data/derived/real_items_long.csv"
RESULTS = ROOT / "results"
OUT_PREFIX = ARGS.out_prefix


def out(name: str) -> Path:
    """Results file name with the optional --out-prefix prepended."""
    return RESULTS / f"{OUT_PREFIX}{name}"

SEED = 20261112  # bootstrap stream only (AGENTS.md rule 5); same number as 02/04, fresh stream
B = 2000  # bootstrap replications (same as src/04_real_slopes.py)
YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"
BRACKETS = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]
LOWEST, HIGHEST = BRACKETS[0], BRACKETS[-1]
REF_ITEM = 9  # item FE reference, as in src/04_real_slopes.py
MODEL_NAME = ARGS.model  # expected single model in the synthetic file (--model; default: the Gemini run)
EPS = 1e-12  # |gap| below this counts as exactly zero for the H2 ratio

YES_MAP = {YES_TEXT: 1, NO_TEXT: 0}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def to_yes(series: pd.Series, label: str) -> pd.Series:
    unmapped = set(series.unique()) - set(YES_MAP)
    assert not unmapped, f"{label}: unexpected answer texts: {sorted(unmapped)}"
    return series.map(YES_MAP).astype(int)


# ---------------------------------------------------------------------------
# 1. Load and validate
# ---------------------------------------------------------------------------
section("1. LOAD AND VALIDATE")

sample = pd.read_csv(SAMPLE_CSV)
syn = pd.read_csv(SYN_CSV)
print(f"sample_200_items.csv: {len(sample)} rows loaded (expected 2,000 = 200 twins x 10 items)")
print(f"{SYN_CSV.name}: {len(syn)} rows loaded (expected 22,000 = 200 x 10 x 11)")

assert len(sample) == 2000, "sample_200_items.csv must have 2,000 rows"
assert len(syn) == 22000, "synthetic_answers.csv must have 22,000 rows"
assert sample["new_id"].nunique() == 200, "sample must have 200 twins"
assert sorted(sample["item"].unique()) == [9, 10, 15, 23, 25, 26, 27, 31, 36, 38], (
    "sample items must be exactly the 10 chosen items"
)
twins_per_bracket = sample.drop_duplicates("new_id")["QID21"].value_counts()
assert set(twins_per_bracket.index) == set(BRACKETS), "unexpected income bracket texts"
assert (twins_per_bracket == 40).all(), (
    f"stratified sample must have 40 twins per bracket, got {twins_per_bracket.to_dict()}"
)
assert (sample.groupby(["new_id", "item"]).size() == 1).all(), "duplicate (new_id, item) in sample"
assert (syn.groupby(["new_id", "item"]).size() == 11).all(), (
    "synthetic_answers must have exactly 11 ladder rows per (twin, item)"
)
models = syn["model"].unique()
assert list(models) == [MODEL_NAME], f"unexpected synthetic model(s): {models}"
sample["yes"] = to_yes(sample["answer"], "sample answers")
syn["yes"] = to_yes(syn["answer"], "synthetic answers")
print("validation OK: 200 twins x 10 items, 40 per bracket, 11 ladder rows per (twin, item),")
print(f"  one model ({MODEL_NAME}), only the two survey answer texts")

# 2,000 real pairs joined to the synthetic answer AT THE PRICE THE REAL TWIN SAW
pairs = sample.merge(
    syn[["new_id", "item", "price_text", "yes", "relative_price", "free"]],
    on=["new_id", "item", "price_text"],
    how="inner",
    suffixes=("", "_syn"),
    validate="one_to_one",
)
print(f"join on (new_id, item, exact price_text): {len(pairs)} of 2,000 real pairs matched")
assert len(pairs) == 2000, "every real pair must match exactly one synthetic ladder row"
drift = (pairs["relative_price"] - pairs["relative_price_syn"]).abs().max()
assert drift == 0.0, f"relative_price differs within twin pair (max {drift})"
assert (pairs["free"] == pairs["free_syn"]).all(), "free indicator differs within twin pair"
pairs = pairs.drop(columns=["relative_price_syn", "free_syn"])
print("relative_price and free identical within every twin pair (asserted)")

# ---------------------------------------------------------------------------
# 2. Yes share per item x income bracket, real vs synthetic
# ---------------------------------------------------------------------------
section("2. YES SHARE PER ITEM x INCOME BRACKET (real vs synthetic, 40 twins per cell)")

meta = sample.groupby("item")[["category", "product"]].first().reset_index()


def shares(df: pd.DataFrame, yes_col: str, label: str) -> pd.DataFrame:
    g = df.groupby(["item", "QID21"])[yes_col].agg(n="size", n_yes="sum", share_yes="mean").reset_index()
    g = g.rename(columns={c: f"{c}_{label}" for c in ["n", "n_yes", "share_yes"]})
    return g


shares_wide = shares(sample, "yes", "real").merge(shares(pairs, "yes_syn", "synth"), on=["item", "QID21"])
shares_wide = shares_wide.merge(meta, on="item")
bracket_order = {b: i for i, b in enumerate(BRACKETS)}
shares_wide = shares_wide.sort_values(["item", "QID21"], key=lambda s: s.map(bracket_order) if s.name == "QID21" else s)
shares_wide = shares_wide[
    ["item", "category", "product", "QID21", "n_real", "n_yes_real", "share_yes_real",
     "n_synth", "n_yes_synth", "share_yes_synth"]
]
assert (shares_wide[["n_real", "n_synth"]] == 40).all().all(), "every (item, bracket) cell must have 40 twins"
print(shares_wide.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# 3. Twin-level matrices + paired stratified bootstrap (B = 2,000, seed 20261112)
# ---------------------------------------------------------------------------
section(f"3. BOOTSTRAP BY TWIN (B = {B}, seed {SEED}, percentile CIs, stratified by bracket, paired)")

twins = sorted(sample["new_id"].unique())
bracket_of = sample.drop_duplicates("new_id").set_index("new_id")["QID21"]
real_yes = (
    sample.pivot(index="new_id", columns="item", values="yes").reindex(index=twins).sort_index(axis=1).to_numpy()
)
syn_yes = (
    pairs.pivot(index="new_id", columns="item", values="yes_syn").reindex(index=twins).sort_index(axis=1).to_numpy()
)
assert not np.array_equal(real_yes, syn_yes), (
    "synthetic matrix must not equal the real matrix (wrong column selected)"
)
items = sorted(sample["item"].unique())
assert real_yes.shape == syn_yes.shape == (200, len(items))
member = {b: np.flatnonzero((bracket_of.reindex(twins) == b).to_numpy()) for b in BRACKETS}
assert all(len(m) == 40 for m in member.values())

rng = np.random.default_rng(SEED)
n_items = len(items)
real_gap_b = np.empty((B, n_items))
syn_gap_b = np.empty((B, n_items))
real_rate_b = np.empty(B)
syn_rate_b = np.empty(B)
for r in range(B):
    idx = {b: m[rng.integers(0, len(m), len(m))] for b, m in member.items()}
    hi, lo = idx[HIGHEST], idx[LOWEST]
    real_gap_b[r] = real_yes[hi].mean(axis=0) - real_yes[lo].mean(axis=0)
    syn_gap_b[r] = syn_yes[hi].mean(axis=0) - syn_yes[lo].mean(axis=0)
    pick = np.concatenate([idx[b] for b in BRACKETS])
    real_rate_b[r] = real_yes[pick].mean()
    syn_rate_b[r] = syn_yes[pick].mean()

ci_lo = np.percentile(real_gap_b, 2.5, axis=0)
ci_hi = np.percentile(real_gap_b, 97.5, axis=0)
syn_lo = np.percentile(syn_gap_b, 2.5, axis=0)
syn_hi = np.percentile(syn_gap_b, 97.5, axis=0)

point_real_gap = real_yes[member[HIGHEST]].mean(axis=0) - real_yes[member[LOWEST]].mean(axis=0)
point_syn_gap = syn_yes[member[HIGHEST]].mean(axis=0) - syn_yes[member[LOWEST]].mean(axis=0)

# H2 ratio: synthetic gap / real gap; CI over replications with a non-zero real gap
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_b = np.where(np.abs(real_gap_b) > EPS, syn_gap_b / real_gap_b, np.nan)
ratio_valid = (~np.isnan(ratio_b).all(axis=0)) & ((np.abs(real_gap_b) > EPS).sum(axis=0) > 0)
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_point = np.where(np.abs(point_real_gap) > EPS, point_syn_gap / point_real_gap, np.nan)
ratio_ci = np.full((2, n_items), np.nan)
for j in range(n_items):
    vals = ratio_b[:, j][~np.isnan(ratio_b[:, j])]
    if len(vals):
        ratio_ci[:, j] = np.percentile(vals, [2.5, 97.5])
n_ratio_valid = (~np.isnan(ratio_b)).sum(axis=0)

# ---------------------------------------------------------------------------
# 4. Gaps table + exploratory 1,979-respondent gap (NOT used in any verdict)
# ---------------------------------------------------------------------------
section("4. INCOME GAPS PER ITEM (highest minus lowest bracket)")

real_long = pd.read_csv(REAL_LONG_CSV)
assert len(real_long) == 19790, f"real_items_long.csv must have 19,790 rows, got {len(real_long)}"
assert real_long["pid"].nunique() == 1979, "real_items_long must cover all 1,979 valid respondents"
assert (real_long.groupby("pid").size() == 10).all(), "10 rows per respondent expected"
print(f"real_items_long.csv: {len(real_long)} rows, {real_long['pid'].nunique()} valid respondents (exploratory input)")

rl_yes = real_long.pivot(index="pid", columns="item", values="yes").sort_index(axis=1).to_numpy()
rl_bracket = real_long.drop_duplicates("pid").set_index("pid")["QID21"]
pids = real_long.drop_duplicates("pid")["pid"].sort_values().to_numpy()
member1979 = {
    b: np.flatnonzero((rl_bracket.reindex(pids) == b).to_numpy()) for b in (LOWEST, HIGHEST)
}
print(f"exploratory bracket sizes: '{LOWEST}' n={len(member1979[LOWEST])}, "
      f"'{HIGHEST}' n={len(member1979[HIGHEST])} (of 1,979)")
gap1979_b = np.empty((B, n_items))
for r in range(B):
    hi = member1979[HIGHEST][rng.integers(0, len(member1979[HIGHEST]), len(member1979[HIGHEST]))]
    lo = member1979[LOWEST][rng.integers(0, len(member1979[LOWEST]), len(member1979[LOWEST]))]
    gap1979_b[r] = rl_yes[hi].mean(axis=0) - rl_yes[lo].mean(axis=0)
gap1979_point = rl_yes[member1979[HIGHEST]].mean(axis=0) - rl_yes[member1979[LOWEST]].mean(axis=0)
gap1979_lo = np.percentile(gap1979_b, 2.5, axis=0)
gap1979_hi = np.percentile(gap1979_b, 97.5, axis=0)

gaps = pd.DataFrame({
    "item": items,
    "product": meta.set_index("item").reindex(items)["product"].to_numpy(),
    "gap_real": point_real_gap,
    "gap_real_ci_lo": ci_lo,
    "gap_real_ci_hi": ci_hi,
    "gap_synth": point_syn_gap,
    "gap_synth_ci_lo": syn_lo,
    "gap_synth_ci_hi": syn_hi,
    "sign_agree": np.sign(point_real_gap) == np.sign(point_syn_gap),
    "testable_H1": (ci_lo > 0) | (ci_hi < 0),
    "ratio_synth_over_real": ratio_point,
    "ratio_ci_lo": ratio_ci[0],
    "ratio_ci_hi": ratio_ci[1],
    "n_ratio_valid_reps": n_ratio_valid,
    "diff_synth_minus_real": point_syn_gap - point_real_gap,
    "EXPLORATORY_gap_real_all1979": gap1979_point,
    "EXPLORATORY_gap_real_all1979_ci_lo": gap1979_lo,
    "EXPLORATORY_gap_real_all1979_ci_hi": gap1979_hi,
})
print("\n--- 200-twin real vs synthetic gaps (used in the H1/H2/H3 verdicts) ---")
print(gaps[["item", "product", "gap_real", "gap_real_ci_lo", "gap_real_ci_hi", "gap_synth",
            "gap_synth_ci_lo", "gap_synth_ci_hi", "sign_agree", "testable_H1"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))
print("\n--- EXPLORATORY, NOT USED IN ANY VERDICT: real gap on all 1,979 valid respondents ---")
print(gaps[["item", "gap_real", "EXPLORATORY_gap_real_all1979",
            "EXPLORATORY_gap_real_all1979_ci_lo", "EXPLORATORY_gap_real_all1979_ci_hi"]].to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# 5. Regression (plan.md section 5): LPM, SEs clustered by twin (new_id)
# ---------------------------------------------------------------------------
section("5. REGRESSION: purchase ~ income bracket x source + item FE + relative price + free")

reg = pd.concat([
    pairs.assign(source=0, yes=pairs["yes"], relative_price=pairs["relative_price"], free=pairs["free"]),
    pairs.assign(source=1, yes=pairs["yes_syn"]),
], ignore_index=True)
reg["source"] = reg["source"].astype(int)
assert len(reg) == 4000, f"regression table must have 4,000 rows, got {len(reg)}"
print(f"regression table: {len(reg)} rows = 2,000 real (source=0) + 2,000 synthetic (source=1)")
print("coding: bracket treatment-coded (ref = lowest), item FE ref = item 9, source real=0/synthetic=1")

FORMULA = (
    f'yes ~ C(QID21, Treatment(reference="{LOWEST}")) * source '
    f"+ C(item, Treatment(reference={REF_ITEM})) + {{PRICE}} + free"
)


def fit_lpm(price_var: str, groups: np.ndarray):
    fit = smf.ols(FORMULA.format(PRICE=price_var), data=reg).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    return pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.to_numpy(),
        "se": fit.bse.to_numpy(),
        "z": fit.tvalues.to_numpy(),
        "p": fit.pvalues.to_numpy(),
        "ci_low": fit.conf_int().iloc[:, 0].to_numpy(),
        "ci_high": fit.conf_int().iloc[:, 1].to_numpy(),
    })


cluster200 = reg["new_id"].to_numpy()
cluster400 = pd.factorize(reg["source"].astype(str) + "|" + reg["new_id"])[0]

tables = {}
for model, price_var in (("relative_price", "relative_price"), ("price_dollars", "price_dollars")):
    tables[model] = fit_lpm(price_var, cluster200)
    tables[model]["model"] = model

regression_out = pd.concat([tables["relative_price"], tables["price_dollars"]], ignore_index=True)
regression_out = regression_out[["model", "term", "coef", "se", "z", "p", "ci_low", "ci_high"]]

main = tables["relative_price"]
print(f"\nMAIN MODEL (price = relative_price, SEs clustered by new_id, 200 clusters):")
print(main.drop(columns=["model"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

inter = main[main["term"].str.contains("source", regex=False)]
alt = fit_lpm("relative_price", cluster400).set_index("term").reindex(inter["term"])
note = pd.DataFrame({
    "term": inter["term"].to_numpy(),
    "se_200_clusters": inter["se"].to_numpy(),
    "se_400_clusters": alt["se"].to_numpy(),
})
print("\ninteraction SEs under the alternative (source x new_id) clustering, 400 clusters:")
print(note.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nROBUSTNESS (section 5): raw dollar price instead of relative price")
dollar = tables["price_dollars"]
cmp = pd.DataFrame({
    "term": inter["term"].to_numpy(),
    "coef_relative": inter["coef"].to_numpy(),
    "coef_dollar": dollar.set_index("term").reindex(inter["term"])["coef"].to_numpy(),
})
print(cmp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
same_sign = (np.sign(cmp["coef_relative"]) == np.sign(cmp["coef_dollar"])).all()
print(f"income x source interaction signs identical between the two price controls: {same_sign}")

# ---------------------------------------------------------------------------
# 6. H1 / H2 / H3 verdicts (section 6 rules, applied exactly)
# ---------------------------------------------------------------------------
section("6. HYPOTHESIS VERDICTS (section 6 decision rules)")

overall_real = pairs["yes"].mean()
overall_syn = pairs["yes_syn"].mean()
rate_lo, rate_hi = np.percentile(real_rate_b, [2.5, 97.5])
syn_rate_lo, syn_rate_hi = np.percentile(syn_rate_b, [2.5, 97.5])

testable = gaps["testable_H1"].to_numpy()
n_test = int(testable.sum())
sign_agree_all = gaps.loc[testable, "sign_agree"].to_numpy()
n_agree = int(sign_agree_all.sum())
sign_share = n_agree / n_test if n_test else np.nan
corr_all = float(np.corrcoef(point_real_gap, point_syn_gap)[0, 1])
if n_test >= 2 and np.std(point_real_gap[testable]) > EPS and np.std(point_syn_gap[testable]) > EPS:
    corr_test = float(np.corrcoef(point_real_gap[testable], point_syn_gap[testable])[0, 1])
else:
    corr_test = np.nan

h1_rejected = None
if n_test == 0:
    h1_verdict = "cannot test (0 testable products)"
elif np.isnan(corr_test):
    h1_rejected = sign_share < 0.5
    h1_verdict = f"rejected (section 6 condition met)" if h1_rejected else "not rejected"
    h1_verdict += " [correlation undefined with < 2 testable products: decided on sign share alone]"
else:
    h1_rejected = (sign_share < 0.5) or (corr_test <= 0)
    h1_verdict = "rejected (section 6 condition met)" if h1_rejected else "not rejected"

inside = ((gaps["gap_synth"] >= gaps["gap_real_ci_lo"]) & (gaps["gap_synth"] <= gaps["gap_real_ci_hi"]))
n_inside = int(inside.sum())
h2_rejected = n_inside >= 5  # half or more of the 10 items (approved choice 6)
h2_verdict = "rejected (section 6 condition met)" if h2_rejected else "not rejected"

h3_rejected = rate_lo <= overall_syn <= rate_hi
h3_verdict = "rejected (section 6 condition met)" if h3_rejected else "not rejected"

print(f"\nH1 (direction): {n_test} of 10 products testable (real gap 95% CI excludes 0)")
print(f"  sign agreement: {n_agree}/{n_test} = {sign_share:.4f} (rejected if < 0.5)")
print(f"  Pearson correlation real vs synthetic gaps: all 10 = {corr_all:.4f}; "
      f"testable only = {corr_test:.4f} (rejected if <= 0)")
print(f"  H1 verdict: {h1_verdict}")

print(f"\nH2 (size of gaps): synthetic gap inside real gap 95% CI for {n_inside} of 10 items "
      f"(rejected if >= 5)")
print(f"  per-item ratios (synthetic gap / real gap): "
      + ", ".join(f"item {i}: {r:.3f}" if not np.isnan(r) else f"item {i}: n/a (real gap 0)"
                  for i, r in zip(gaps["item"], gaps["ratio_synth_over_real"])))
print(f"  H2 verdict: {h2_verdict}")

print(f"\nH3 (overall purchase rate, 2,000 pairs each):")
print(f"  real = {overall_real:.4f}  95% bootstrap CI [{rate_lo:.4f}, {rate_hi:.4f}]")
print(f"  synthetic = {overall_syn:.4f}  95% bootstrap CI [{syn_rate_lo:.4f}, {syn_rate_hi:.4f}]")
print(f"  rejected if the synthetic rate falls inside the real CI -> "
      f"{rate_lo:.4f} <= {overall_syn:.4f} <= {rate_hi:.4f}: {h3_rejected}")
print(f"  H3 verdict: {h3_verdict}")

hypotheses = pd.DataFrame([
    {
        "hypothesis": "H1 (direction)",
        "n_items": 10, "n_testable": n_test,
        "sign_agree_n": n_agree, "sign_agree_share": sign_share,
        "corr_all10": corr_all, "corr_testable": corr_test,
        "n_inside_real_ci": np.nan,
        "real_rate": np.nan, "real_rate_ci_lo": np.nan, "real_rate_ci_hi": np.nan,
        "synth_rate": np.nan, "synth_rate_ci_lo": np.nan, "synth_rate_ci_hi": np.nan,
        "verdict": h1_verdict, "rejected": h1_rejected,
    },
    {
        "hypothesis": "H2 (size of gaps)",
        "n_items": 10, "n_testable": np.nan,
        "sign_agree_n": np.nan, "sign_agree_share": np.nan,
        "corr_all10": np.nan, "corr_testable": np.nan,
        "n_inside_real_ci": n_inside,
        "real_rate": np.nan, "real_rate_ci_lo": np.nan, "real_rate_ci_hi": np.nan,
        "synth_rate": np.nan, "synth_rate_ci_lo": np.nan, "synth_rate_ci_hi": np.nan,
        "verdict": h2_verdict, "rejected": h2_rejected,
    },
    {
        "hypothesis": "H3 (overall rate)",
        "n_items": np.nan, "n_testable": np.nan,
        "sign_agree_n": np.nan, "sign_agree_share": np.nan,
        "corr_all10": np.nan, "corr_testable": np.nan,
        "n_inside_real_ci": np.nan,
        "real_rate": overall_real, "real_rate_ci_lo": rate_lo, "real_rate_ci_hi": rate_hi,
        "synth_rate": overall_syn, "synth_rate_ci_lo": syn_rate_lo, "synth_rate_ci_hi": syn_rate_hi,
        "verdict": h3_verdict, "rejected": h3_rejected,
    },
])

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
section("7. SAVE")
RESULTS.mkdir(exist_ok=True)
shares_wide.to_csv(out("compare1_shares.csv"), index=False, float_format="%.6f")
gaps.to_csv(out("compare1_gaps.csv"), index=False, float_format="%.6f")
regression_out.to_csv(out("compare1_regression.csv"), index=False, float_format="%.6f")
hypotheses.to_csv(out("compare1_hypotheses.csv"), index=False, float_format="%.6f")
for f in ("compare1_shares.csv", "compare1_gaps.csv", "compare1_regression.csv", "compare1_hypotheses.csv"):
    print(f"wrote results/{OUT_PREFIX}{f}")
print(f"\nbootstrap: B = {B}, seed = {SEED} (numpy default_rng; one stream, main bootstrap first)")
print("done.")
