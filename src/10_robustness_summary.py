"""10_robustness_summary.py -- headline robustness summary (plan.md section 7, entry
"17 Nov 2026: robustness analysis", points (c), (d), (e), (f)).

Reads (read-only; no analysis logic of 07/08 is duplicated):
  data/derived/synthetic_answers.csv        Gemini run, 22,000 rows
  data/derived/robustness_answers_full.csv  gpt-oss run, 22,000 rows (recovered, format_kind)
  data/derived/real_items_long.csv          real microdata, 19,790 rows (our 10 items)
  data/derived/sample_200_items.csv         200-twin real answers (real twin's prices)
  results/real_slopes.csv                   real pooled slope (src/04 bootstrap)
  results/compare1_hypotheses.csv           Gemini H1-H3 verdicts and rates
  results/compare2_regression_pooled.csv    Gemini pooled joint regression
  results/compare2_hypotheses.csv           Gemini H4/H5 verdicts
  results/robust_compare1_hypotheses.csv    gpt-oss H1-H3 verdicts and rates
  results/robust_compare2_regression_pooled.csv  gpt-oss pooled joint regression
  results/robust_compare2_hypotheses.csv    gpt-oss H4/H5 verdicts

Writes:
  results/robustness_summary.csv  one row per (block, model, subset, metric)

Definitions (fixed by plan.md section 7, entry of 17 Nov 2026):
  yes_share_at_0        share of Yes among rows with price $0 (free == 1)
  share_single_threshold  share of (twin, item) ladders with exactly one Yes->No
                        switch and no No->Yes switch when answers are ordered by
                        ascending price (undefined for the real data: each
                        respondent saw one random price per product, no ladder)
  pooled_synthetic_slope  relative_price + source:relative_price from the pooled
                        joint regression file (== H4's synth_pooled_slope; subset
                        rows have no regression, hence empty)
  overall_yes_rate      H3's rate: Yes share at the real twin's price, 2,000
                        (twin, item) pairs (real: the 200-twin sample's real_rate)

Point (f) reading rule is applied in the printed output exactly as written.
"""

import numpy as np
import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SYN_GEM = ROOT / "data/derived/synthetic_answers.csv"
SYN_OSS = ROOT / "data/derived/robustness_answers_full.csv"
REAL_LONG = ROOT / "data/derived/real_items_long.csv"
SAMPLE = ROOT / "data/derived/sample_200_items.csv"
REAL_SLOPES = ROOT / "results/real_slopes.csv"
HYP1_GEM = ROOT / "results/compare1_hypotheses.csv"
HYP1_OSS = ROOT / "results/robust_compare1_hypotheses.csv"
REG_GEM = ROOT / "results/compare2_regression_pooled.csv"
REG_OSS = ROOT / "results/robust_compare2_regression_pooled.csv"
HYP2_GEM = ROOT / "results/compare2_hypotheses.csv"
HYP2_OSS = ROOT / "results/robust_compare2_hypotheses.csv"
OUT = ROOT / "results/robustness_summary.csv"

YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"
SINGLE_MIN = 0.90  # point (f): at least 90% of gpt-oss ladders single-threshold

rows = []  # (block, model, subset, metric, value, n, note)


def add(block, model, subset, metric, value, n=np.nan, note=""):
    rows.append({"block": block, "model": model, "subset": subset, "metric": metric,
                 "value": value, "n": n, "note": note})


def yes_col(df):
    y = df["answer"].map({YES_TEXT: 1, NO_TEXT: 0})
    assert y.notna().all(), "unexpected answer texts"
    return y.astype(int)


def ladder_shape(df):
    """Per (new_id, item): (n_ladders, n_single_threshold, n_threshold_agreeable...)
    returns DataFrame with index (new_id, item) and columns single, threshold.
    single = exactly one Yes->No switch and no No->Yes switch; threshold = ladder
    position (0..10) of the last Yes, defined only for single-threshold ladders."""
    d = df.sort_values(["new_id", "item", "price_dollars"]).copy()
    d["y"] = yes_col(d)
    g = d.groupby(["new_id", "item"])
    assert (g.size() == 11).all(), "each (twin, item) ladder must have 11 answers"
    assert (g["price_dollars"].nunique() == 11).all(), "11 distinct prices per ladder"
    seq = np.stack([s.to_numpy() for _, s in g["y"]])          # (n_ladders, 11) in group order
    keys = list(g.size().index)
    up = (np.diff(seq, axis=1) == 1).sum(axis=1)   # No -> Yes switches
    down = (np.diff(seq, axis=1) == -1).sum(axis=1)  # Yes -> No switches
    single = (up == 0) & (down == 1)
    # threshold of a single-threshold ladder = number of Yes answers before the
    # single Yes->No switch (the leading Yes run); undefined otherwise
    threshold = np.where(single, seq.sum(axis=1), np.nan)
    return pd.DataFrame({"single": single, "threshold": threshold},
                        index=pd.MultiIndex.from_tuples(keys, names=["new_id", "item"]))


def at_real_price_rate(df, sample):
    """H3-style overall Yes rate: the answer at the real twin's price (one pair per
    (twin, item) in the subset)."""
    j = sample[["new_id", "item", "price_text"]].merge(
        df[["new_id", "item", "price_text", "answer"]], on=["new_id", "item", "price_text"],
        how="inner", validate="one_to_one")
    n_pairs = df.groupby(["new_id", "item"]).ngroups
    assert len(j) == n_pairs, f"join at the real twin's price must give {n_pairs} pairs, got {len(j)}"
    return yes_col(j).mean(), len(j)


def load_pair(path_reg, path_hyp2, label):
    reg = pd.read_csv(path_reg)
    assert set(reg["model"].unique()) == {"relative_price", "price_dollars"}
    rp = reg[reg["model"] == "relative_price"].set_index("term")
    slope = rp.loc["relative_price", "coef"] + rp.loc["source:relative_price", "coef"]
    hyp2 = pd.read_csv(path_hyp2)
    hyp4 = hyp2[hyp2["hypothesis"].str.startswith("H4")].iloc[0]
    assert np.isclose(slope, hyp4["synth_pooled_slope"], atol=1e-4), (
        f"{label}: pooled slope {slope:.6f} != H4 file's synth_pooled_slope {hyp4['synth_pooled_slope']:.6f}")
    real_slope = float(hyp4["real_pooled_slope"])
    print(f"{label}: pooled synthetic slope = relative_price {rp.loc['relative_price', 'coef']:+.4f} "
          f"+ source:relative_price {rp.loc['source:relative_price', 'coef']:+.4f} = {slope:+.4f} "
          f"(matches the H4 file's synth_pooled_slope {hyp4['synth_pooled_slope']:+.4f})")
    return slope, real_slope


print("=" * 78)
print("1. LOAD AND COUNT (rule 2)")
print("=" * 78)
gem = pd.read_csv(SYN_GEM)
oss = pd.read_csv(SYN_OSS)
real = pd.read_csv(REAL_LONG)
sample = pd.read_csv(SAMPLE)
assert len(gem) == 22000 and len(oss) == 22000, "both synthetic files must have 22,000 rows"
assert len(real) == 19790 and real["pid"].nunique() == 1979, "real_items_long: 19,790 rows / 1,979 pids"
assert len(sample) == 2000
print(f"synthetic_answers.csv (Gemini): {len(gem)} rows, {gem.groupby(['new_id','item']).ngroups} ladders")
print(f"robustness_answers_full.csv (gpt-oss): {len(oss)} rows, "
      f"{oss.groupby(['new_id','item']).ngroups} ladders, "
      f"{int(oss['recovered'].sum())} rows recovered ({oss.loc[oss['recovered']==1,['new_id','item']].drop_duplicates().shape[0]} keys), "
      f"format_kind: {oss['format_kind'].value_counts().to_dict()}")
print(f"real_items_long.csv: {len(real)} rows, {real['pid'].nunique()} respondents (our 10 items)")
print(f"sample_200_items.csv: {len(sample)} rows")

print()
print("=" * 78)
print("2. HEADLINE (c): per model next to the real data")
print("=" * 78)
hyp1_gem = pd.read_csv(HYP1_GEM).set_index("hypothesis")
hyp1_oss = pd.read_csv(HYP1_OSS).set_index("hypothesis")
gem_slope, real_slope_reg = load_pair(REG_GEM, HYP2_GEM, "Gemini")
oss_slope, _ = load_pair(REG_OSS, HYP2_OSS, "gpt-oss")
rs = pd.read_csv(REAL_SLOPES)
real_slope = float(rs[rs["item"].isna() & (rs["sample"] == "all")]["slope"].iloc[0])
assert np.isclose(real_slope, real_slope_reg, atol=5e-4)
print(f"real pooled slope (real_slopes.csv, src/04 bootstrap): {real_slope:+.4f}")

HEAD = "c_headline"

# --- real data -----------------------------------------------------------
m = real["free"] == 1
n0 = int(m.sum())
add(HEAD, "real (200-twin sample)", "all", "yes_share_at_0", real.loc[m, "yes"].mean(), n0,
    "our 10 items; one random price per respondent x item (no $0 question per person)")
add(HEAD, "real (200-twin sample)", "all", "share_single_threshold", np.nan, 0,
    "undefined for real data: each respondent saw ONE random price per product, no ladder")
add(HEAD, "real (200-twin sample)", "all", "pooled_slope", real_slope, np.nan,
    "src/04 respondent bootstrap, all 5 brackets pooled")
r_gem_rate, _ = at_real_price_rate(gem, sample)
real_rate = float(hyp1_gem.loc["H3 (overall rate)", "real_rate"])
assert np.isclose(sample["answer"].map({YES_TEXT: 1, NO_TEXT: 0}).mean(), real_rate)
add(HEAD, "real (200-twin sample)", "all", "overall_yes_rate", real_rate, 2000,
    "H3 real rate: the 200 twins' wave-4 answers at the price each twin saw (plan c: 0.4750)")

# --- Gemini --------------------------------------------------------------
m = gem["free"] == 1
add(HEAD, "gemini-3.5-flash-lite", "all", "yes_share_at_0", gem.loc[m, "answer"].eq(YES_TEXT).mean(), int(m.sum()))
lad_gem = ladder_shape(gem)
add(HEAD, "gemini-3.5-flash-lite", "all", "share_single_threshold", lad_gem["single"].mean(), len(lad_gem))
add(HEAD, "gemini-3.5-flash-lite", "all", "pooled_synthetic_slope", gem_slope, 22000,
    "compare2_regression_pooled.csv: relative_price + source:relative_price")
gem_rate, _ = at_real_price_rate(gem, sample)
assert np.isclose(gem_rate, hyp1_gem.loc["H3 (overall rate)", "synth_rate"])
add(HEAD, "gemini-3.5-flash-lite", "all", "overall_yes_rate", gem_rate, 2000,
    "H3 synth rate at the real twin's price")

# --- gpt-oss (all) -------------------------------------------------------
m = oss["free"] == 1
add(HEAD, "gpt-oss:20b", "all", "yes_share_at_0", oss.loc[m, "answer"].eq(YES_TEXT).mean(), int(m.sum()))
lad_oss = ladder_shape(oss)
oss_single_share = float(lad_oss["single"].mean())
add(HEAD, "gpt-oss:20b", "all", "share_single_threshold", oss_single_share, len(lad_oss))
add(HEAD, "gpt-oss:20b", "all", "pooled_synthetic_slope", oss_slope, 22000,
    "robust_compare2_regression_pooled.csv: relative_price + source:relative_price")
oss_rate, _ = at_real_price_rate(oss, sample)
assert np.isclose(oss_rate, hyp1_oss.loc["H3 (overall rate)", "synth_rate"])
add(HEAD, "gpt-oss:20b", "all", "overall_yes_rate", oss_rate, 2000,
    "H3 synth rate at the real twin's price")

for r in rows:
    if r["block"] == HEAD:
        v = "" if pd.isna(r["value"]) else f"{r['value']:.4f}"
        print(f"  {r['model']:<24} {r['metric']:<24} {v:>8}  (n={r['n']:.0f})")

print()
print("=" * 78)
print("3. MODEL AGREEMENT (d)")
print("=" * 78)
AG = "d_agreement"
j = gem[["new_id", "item", "price_text", "answer"]].merge(
    oss[["new_id", "item", "price_text", "answer"]], on=["new_id", "item", "price_text"],
    how="inner", suffixes=("_gem", "_oss"), validate="one_to_one")
assert len(j) == 22000, f"(d) join must give 22,000 cells, got {len(j)}"
ident = (j["answer_gem"] == j["answer_oss"])
add(AG, "gemini-3.5-flash-lite vs gpt-oss:20b", "all", "identical_answer_share", ident.mean(), len(j),
    "share of identical Yes/No answers over the 22,000 (twin, item, price) cells")
both_single = lad_gem["single"] & lad_oss["single"].reindex(lad_gem.index).fillna(False)
same_thr = (lad_gem["threshold"] == lad_oss["threshold"])[both_single]
add(AG, "gemini-3.5-flash-lite vs gpt-oss:20b", "all", "both_single_threshold_share", both_single.mean(), len(lad_gem),
    "share of the 2,000 (twin, item) ladders where BOTH models give a single-threshold ladder")
add(AG, "gemini-3.5-flash-lite vs gpt-oss:20b", "all", "same_threshold_share", same_thr.mean(), int(both_single.sum()),
    "share of the both-single-threshold ladders with the SAME threshold (last-Yes position)")
n_diff = int((~ident).sum())
print(f"  identical answers: {ident.sum():,} of 22,000 cells = {ident.mean():.4f} "
      f"({n_diff:,} cells differ)")
print(f"  ladders single-threshold in BOTH models: {int(both_single.sum())} of {len(lad_gem)} "
      f"= {both_single.mean():.4f}")
print(f"  same threshold among those: {int(same_thr.sum())} of {int(both_single.sum())} = {same_thr.mean():.4f}")

print()
print("=" * 78)
print("4. CHECKS (e) for gpt-oss: without the 16 recovered keys; by format_kind")
print("=" * 78)
EC = "e_check"
subsets = [
    ("excl 16 recovered keys", oss[oss["recovered"] == 0]),
    ("format_kind = root_array", oss[oss["format_kind"] == "root_array"]),
    ("format_kind = wrapper", oss[oss["format_kind"] == "wrapper"]),
]
for label, sub in subsets:
    keys = sub.groupby(["new_id", "item"]).ngroups
    assert len(sub) == keys * 11, f"{label}: {len(sub)} rows must be {keys} keys x 11"
    m = sub["free"] == 1
    add(EC, "gpt-oss:20b", label, "yes_share_at_0", sub.loc[m, "answer"].eq(YES_TEXT).mean(), int(m.sum()))
    lad = ladder_shape(sub)
    add(EC, "gpt-oss:20b", label, "share_single_threshold", lad["single"].mean(), len(lad))
    rate, n_pairs = at_real_price_rate(sub, sample)
    add(EC, "gpt-oss:20b", label, "overall_yes_rate", rate, n_pairs,
        "pairs at the real twin's price within the subset")
    add(EC, "gpt-oss:20b", label, "pooled_synthetic_slope", np.nan, np.nan,
        "not recomputed for subsets (no regression run; check only, plan e)")
    print(f"  {label:<28} {keys:>5} keys x 11 = {len(sub):>6,} rows; "
          f"Yes@$0 = {sub.loc[m, 'answer'].eq(YES_TEXT).mean():.4f} (n={int(m.sum())}), "
          f"single-threshold = {lad['single'].mean():.4f} ({len(lad)} ladders), "
          f"overall rate = {rate:.4f}")

print()
print("=" * 78)
print("5. H1-H5 VERDICTS SIDE BY SIDE (f) + READING RULE")
print("=" * 78)
hyp2_gem = pd.read_csv(HYP2_GEM).set_index("hypothesis")
hyp2_oss = pd.read_csv(HYP2_OSS).set_index("hypothesis")
verdicts = {
    "H1 (direction)": (hyp1_gem.loc["H1 (direction)", "verdict"], hyp1_oss.loc["H1 (direction)", "verdict"]),
    "H2 (size of gaps)": (hyp1_gem.loc["H2 (size of gaps)", "verdict"], hyp1_oss.loc["H2 (size of gaps)", "verdict"]),
    "H3 (overall rate)": (hyp1_gem.loc["H3 (overall rate)", "verdict"], hyp1_oss.loc["H3 (overall rate)", "verdict"]),
    "H4 (price sensitivity)": (hyp2_gem.loc["H4 (price sensitivity)", "verdict"], hyp2_oss.loc["H4 (price sensitivity)", "verdict"]),
    "H5 (income gradient)": (hyp2_gem.loc["H5 (income gradient in price sensitivity)", "verdict"], hyp2_oss.loc["H5 (income gradient in price sensitivity)", "verdict"]),
}
w = 60
print(f"{'':4}{'Gemini':<{w}} gpt-oss:20b")
for h, (vg, vo) in verdicts.items():
    print(f"{h:<24} {vg:<{w}} {vo}")
print("\npoint (f) reading rule, applied exactly as written:")
c1 = verdicts["H3 (overall rate)"][0] == verdicts["H3 (overall rate)"][1]
c2 = verdicts["H4 (price sensitivity)"][0] == verdicts["H4 (price sensitivity)"][1]
c3 = oss_single_share >= SINGLE_MIN
print(f"  condition 1: same H3 verdict as Gemini          -> {verdicts['H3 (overall rate)'][1]} "
      f"vs {verdicts['H3 (overall rate)'][0]} : {'MET' if c1 else 'NOT met'}")
print(f"  condition 2: same H4 verdict as Gemini          -> {verdicts['H4 (price sensitivity)'][1]} "
      f"vs {verdicts['H4 (price sensitivity)'][0]} : {'MET' if c2 else 'NOT met'}")
print(f"  condition 3: >= 90% of gpt-oss ladders single-threshold -> "
      f"{oss_single_share:.4f} (n = {len(lad_oss)}) >= 0.90 : {'MET' if c3 else 'NOT met'}")
if c1 and c2 and c3:
    print("  -> all conditions met: the conclusion 'synthetic twins answer in a step")
    print("     pattern, more price sensitive than real people' is reported as ROBUST")
    print("     to the model.")
else:
    print("  -> at least one condition not met: the conclusion is reported as")
    print("     MODEL-DEPENDENT (a limitation).")
print("  H1, H2, H5 verdicts are reported side by side above WITHOUT a robustness")
print("  claim (low power, plan section 6 / mistakes.md 35).")

summary = pd.DataFrame(rows)
summary.to_csv(OUT, index=False, float_format="%.6f")
print(f"\nrows written: {len(summary)} to {OUT.name}")
print("done.")
