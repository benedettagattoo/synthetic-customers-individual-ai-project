"""02_demand_curves.py — real demand curves for the 10 target products.

Yes share at each of the 11 ladder prices, overall and by income bracket (QID21).
Input: data/derived/wave4_target_items_long.csv (from src/01_prepare.py, 1979 valid
respondents x 10 items). Outputs: results CSVs and figures.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DERIVED = Path("data/derived")
RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET_ITEMS = [25, 27, 31, 9, 36, 26, 15, 23, 38, 10]
PRODUCT_LABEL = {
    25: "Land O Lakes Butter 16 oz (item 25)",
    27: "Goya Cooked Ham 16 oz (item 27)",
    31: "Fresh Raspberries 12 oz (item 31)",
    9: "Cinnamon Toast Crunch 18.8 oz (item 9)",
    36: "Angel Soft Toilet Paper 9 Mega Rolls (item 36)",
    26: "Chobani Greek Yogurt Vanilla 32 oz (item 26)",
    15: "HARIBO Goldbears 28.8 oz (item 15)",
    23: "Oscar Mayer Cooked Bacon 2.52 oz (item 23)",
    38: "Duracell Coppertop AA 16 pack (item 38)",
    10: "Little Debbie Oatmeal Creme Pies 12 ct (item 10)",
}
INCOME_ORDER = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]

long = pd.read_csv(DERIVED / "wave4_target_items_long.csv")
print(f"[load] {len(long)} respondent x item rows, {long['pid'].nunique()} respondents, "
      f"items = {sorted(long['item'].unique().tolist())}")

n_resp = long["pid"].nunique()
assert n_resp == 1979, "expected 1979 valid respondents"

def curve_table(group_cols, label):
    """Yes share, binomial SE, n at each ladder point."""
    g = (long.groupby(group_cols, observed=True)
              .agg(n=("yes", "size"), yes_share=("yes", "mean"))
              .reset_index())
    g["yes_se"] = np.sqrt(g["yes_share"] * (1 - g["yes_share"]) / g["n"])
    g["table"] = label
    return g

overall = curve_table(["item", "price_rank"], "overall")
# attach the 11 ladder dollar values per item (rank -> price)
ladder = (long.groupby(["item", "price_rank"])["price"].agg(["min", "max"]).reset_index())
assert (ladder["min"] == ladder["max"]).all(), "each ladder rank must map to a single price"
ladder = ladder.rename(columns={"min": "price"})[["item", "price_rank", "price"]]

overall = overall.merge(ladder, on=["item", "price_rank"])
by_income = curve_table(["item", "income_bracket", "price_rank"], "by_income")
by_income = by_income.merge(ladder, on=["item", "price_rank"])

# empty-cell check: 10 items x 11 ranks (overall) and x 5 brackets
print(f"[check] overall cells per item: min={overall.groupby('item').size().min()}/11, "
      f"max={overall.groupby('item').size().max()}/11 (expect 11)")
empty = by_income[by_income["n"] == 0]
print(f"[check] empty bracket x price cells in the 10x5x11 grid: {len(empty)}")
small = by_income[(by_income["n"] > 0) & (by_income["n"] < 10)]
print(f"[check] cells with n < 10: {len(small)} of {len(by_income)}")
if len(small):
    print(small[["item", "income_bracket", "price", "n"]].to_string(index=False))

# ---------------------------------------------------------------- save tables
cols_o = ["item", "price", "price_rank", "n", "yes_share", "yes_se"]
overall[cols_o].to_csv(RESULTS / "demand_curves_overall.csv", index=False)
cols_i = ["item", "income_bracket", "price", "price_rank", "n", "yes_share", "yes_se"]
by_income[cols_i].to_csv(RESULTS / "demand_curves_by_income.csv", index=False)
print(f"[save] results/demand_curves_overall.csv: {len(overall)} rows (10 items x 11 ladder points)")
print(f"[save] results/demand_curves_by_income.csv: {len(by_income)} rows "
      f"(10 items x {by_income['income_bracket'].nunique()} brackets x 11 ladder points)")

# ---------------------------------------------------------------- figures
def plot_panels(data, hue_col, hue_order, fname, title):
    fig, axes = plt.subplots(2, 5, figsize=(22, 8), sharey=False)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(hue_order)))
    for ax, item in zip(axes.ravel(), TARGET_ITEMS):
        d = data[data["item"] == item]
        lab = PRODUCT_LABEL[item]
        for c, (h, color) in zip(colors, [(h, col) for h, col in zip(hue_order, colors)]):
            dh = d[d[hue_col] == h].sort_values("price_rank")
            if len(dh) == 0:
                continue
            ax.plot(dh["price"], dh["yes_share"], marker="o", ms=3, lw=1.2, color=color, label=h)
            ax.set_title(lab, fontsize=9)
        ax.set_xlabel("Price ($)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("Share answering Yes", fontsize=8)
    axes[1, 0].set_ylabel("Share answering Yes", fontsize=8)
    fig.suptitle(title, fontsize=12)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="lower center", ncol=len(labels_), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(RESULTS / fname, dpi=200)
    plt.close(fig)
    print(f"[save] results/{fname}")

plot_panels(overall.assign(overall="Overall"), "overall", ["Overall"],
            "fig_demand_curves_overall.png",
            "Real demand curves: share of Yes at each of the 11 ladder prices (1979 valid respondents)")
plot_panels(by_income, "income_bracket", INCOME_ORDER,
            "fig_demand_curves_by_income.png",
            "Real demand curves by family income bracket (QID21)")

# quick numeric summary to stdout: yes share at min and max ladder point, overall
ends = []
for item in TARGET_ITEMS:
    d = overall[overall["item"] == item].sort_values("price_rank")
    ends.append({"item": item, "product": PRODUCT_LABEL[item],
                 "price_free": d["price"].iloc[0], "yes_free": d["yes_share"].iloc[0],
                 "price_max": d["price"].iloc[-1], "yes_max": d["yes_share"].iloc[-1]})
print(pd.DataFrame(ends).round(3).to_string(index=False))
