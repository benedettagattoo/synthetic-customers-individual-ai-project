"""03_slopes.py — slope of the Yes share on price for the 10 target products, real data.

Per product (all 1979 valid respondents, one observation per respondent per product):
  (a) OLS linear-probability slope of yes on price in dollars, HC1-robust 95% Wald CI
      (primary; each respondent contributes exactly one row per product, so rows are
      independent and no clustering applies);
  (b) highest-minus-lowest income-bracket difference in that slope, from
      yes ~ price + bracket + price x bracket (reference = lowest bracket),
      coefficient on price x "$100,000 or more", HC1-robust 95% Wald CI;
  (c) sensitivity: the same two quantities with price coded as price / ladder max
      (0-1) instead of dollars, for cross-product comparability;
  (d) sensitivity: respondent-level (pairs) bootstrap of the slope gap, 2000
      replications, seed 20261117, percentile 95% CI, to check the analytic Wald CI in (b).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DERIVED = Path("data/derived")
RESULTS = Path("results")

TARGET_ITEMS = [25, 27, 31, 9, 36, 26, 15, 23, 38, 10]
PRODUCT_LABEL = {
    25: "Land O Lakes Butter 16 oz", 27: "Goya Cooked Ham 16 oz",
    31: "Fresh Raspberries 12 oz", 9: "Cinnamon Toast Crunch 18.8 oz",
    36: "Angel Soft Toilet Paper 9 Mega Rolls", 26: "Chobani Greek Yogurt 32 oz",
    15: "HARIBO Goldbears 28.8 oz", 23: "Oscar Mayer Cooked Bacon 2.52 oz",
    38: "Duracell Coppertop AA 16 pack", 10: "Little Debbie Oatmeal Creme Pies 12 ct",
}
INCOME_ORDER = [
    "Less than $30,000", "$30,000-$50,000", "$50,000-$75,000",
    "$75,000-$100,000", "$100,000 or more",
]
HIGHEST, LOWEST = INCOME_ORDER[-1], INCOME_ORDER[0]
Z = 1.959963984540054  # two-sided 95% normal critical value
SEED = 20261117
N_BOOT = 2000

long = pd.read_csv(DERIVED / "wave4_target_items_long.csv")
print(f"[load] {len(long)} rows, {long['pid'].nunique()} respondents")
for item in TARGET_ITEMS:
    n = (long["item"] == item).sum()
    assert long.loc[long["item"] == item, "pid"].nunique() == n, "one row per respondent per item"


def fit_item(item, price_col="price"):
    """OLS yes ~ price (HC1) + interaction model for the slope gap; return dict.

    price_col: 'price' (dollars) or 'price_share_of_max' (ladder point / max, in [0,1]);
    the regressor is always assigned to d['price'] before fitting.
    """
    d = long[long["item"] == item].copy()
    d["price"] = d[price_col]
    m1 = smf.ols("yes ~ price", data=d).fit(cov_type="HC1")
    m2 = smf.ols(f"yes ~ price * C(income_bracket, Treatment(reference='{LOWEST}'))",
                 data=d).fit(cov_type="HC1")
    gap_terms = [t for t in m2.params.index if t.startswith("price:")]
    assert len(gap_terms) == 4, gap_terms
    hi_term = [t for t in gap_terms if HIGHEST in t]
    assert len(hi_term) == 1, hi_term
    hi_term = hi_term[0]
    return d, m1, m2, {
        "item": item,
        "product": PRODUCT_LABEL[item],
        "n": int(m1.nobs),
        "slope": m1.params["price"],
        "slope_se": m1.bse["price"],
        "slope_ci_low": m1.params["price"] - Z * m1.bse["price"],
        "slope_ci_high": m1.params["price"] + Z * m1.bse["price"],
        "slope_p": m1.pvalues["price"],
        "slope_low_bracket": m2.params["price"],
        "slope_gap": m2.params[hi_term],
        "gap_se": m2.bse[hi_term],
        "gap_ci_low": m2.params[hi_term] - Z * m2.bse[hi_term],
        "gap_ci_high": m2.params[hi_term] + Z * m2.bse[hi_term],
        "gap_p": m2.pvalues[hi_term],
        "gap_ci_includes_0": bool(m2.params[hi_term] - Z * m2.bse[hi_term] <= 0
                                  <= m2.params[hi_term] + Z * m2.bse[hi_term]),
    }


def run_all(price_col):
    rows, fits = [], {}
    for item in TARGET_ITEMS:
        d, m1, m2, row = fit_item(item, price_col=price_col)
        fits[item] = (d, m1, m2)
        rows.append(row)
    return pd.DataFrame(rows), fits


res_dollar, fits_dollar = run_all("price")
res_share, _ = run_all("price_share_of_max")

print("\n[primary] slope of Yes share on price (per $1), OLS LPM, HC1 robust:")
print(res_dollar[["item", "product", "n", "slope", "slope_ci_low", "slope_ci_high", "slope_p"]]
      .round(4).to_string(index=False))
print("\n[primary] highest-minus-lowest income-bracket slope gap (per $1):")
print(res_dollar[["item", "product", "slope_low_bracket", "slope_gap", "gap_ci_low", "gap_ci_high",
                  "gap_p", "gap_ci_includes_0"]].round(4).to_string(index=False))

res_dollar["income_gap_testable"] = ~res_dollar["gap_ci_includes_0"]
res_dollar.to_csv(RESULTS / "slopes_and_income_gaps.csv", index=False)
res_share.to_csv(RESULTS / "slopes_and_income_gaps_share_of_max.csv", index=False)
print(f"\n[save] results/slopes_and_income_gaps.csv: {len(res_dollar)} rows (primary, dollars)")
print(f"[save] results/slopes_and_income_gaps_share_of_max.csv: {len(res_share)} rows "
      "(sensitivity, price coded as price / ladder max in [0,1])")

# ------------------------------ bootstrap sensitivity for the slope gap (fast lstsq)
def gap_beta(d, idx):
    """Slope-gap coefficient from y ~ price*brackets, reference = lowest bracket."""
    db = d.iloc[idx]
    y = db["yes"].to_numpy(float)
    p = db["price"].to_numpy(float)
    B = np.column_stack([(db["income_bracket"] == b).to_numpy(float) for b in INCOME_ORDER])
    X = np.column_stack([np.ones(len(db)), p, B[:, 1:], p[:, None] * B[:, 1:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[-1]  # price x highest bracket = highest - lowest slope gap


rng = np.random.default_rng(SEED)
boot_rows = []
for item in TARGET_ITEMS:
    d = fits_dollar[item][0].reset_index(drop=True)
    gaps = np.array([gap_beta(d, rng.integers(0, len(d), len(d))) for _ in range(N_BOOT)])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    boot_rows.append({"item": item, "product": PRODUCT_LABEL[item], "n_boot": N_BOOT, "seed": SEED,
                      "boot_gap_mean": gaps.mean(), "boot_ci_low": lo, "boot_ci_high": hi})
boot = pd.DataFrame(boot_rows).merge(res_dollar[["item", "slope_gap", "gap_ci_low", "gap_ci_high"]],
                                     on="item")
boot["wald_inside_boot_ci"] = ((boot["gap_ci_low"] >= boot["boot_ci_low"])
                               & (boot["gap_ci_high"] <= boot["boot_ci_high"]))
print("\n[sensitivity] pairs bootstrap of the slope gap (2000 reps, seed 20261117) vs Wald CI:")
print(boot.round(4).to_string(index=False))
boot.to_csv(RESULTS / "slope_gap_bootstrap_sensitivity.csv", index=False)
print(f"[save] results/slope_gap_bootstrap_sensitivity.csv: {len(boot)} rows")

# ------------------------------ figure: slopes and gaps with CIs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
y = np.arange(len(res_dollar))
for ax, col, cl, ch, title, xlabel in [
    (axes[0], "slope", "slope_ci_low", "slope_ci_high",
     "Slope of Yes share on price, per product", "change in P(Yes) per $1"),
    (axes[1], "slope_gap", "gap_ci_low", "gap_ci_high",
     "Highest minus lowest income-bracket slope gap", "slope difference per $1"),
]:
    ax.errorbar(res_dollar[col], y, xerr=[res_dollar[col] - res_dollar[cl],
                                          res_dollar[ch] - res_dollar[col]],
                fmt="o", ms=4, lw=1, capsize=3, color="steelblue")
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} (item {i})" for p, i in zip(res_dollar["product"], res_dollar["item"])],
                       fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
fig.suptitle("Real data: OLS linear-probability estimates, 95% HC1-robust CIs", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "fig_slopes_and_gaps.png", dpi=200)
plt.close(fig)
print("[save] results/fig_slopes_and_gaps.png")
