"""Price-sensitivity slopes with confidence intervals — plan.md section 5,
comparison 2 (H4/H5, real side). Reads only results/agent_run/ files written by
src/agent_01_real_curves.py (built from data/raw); writes only into
results/agent_run/.

Models (linear probability model, OLS): yes ~ 1 + relative_price + free
  - per item, brackets pooled ("All brackets")            -> H4 real slope
  - per item, lowest and highest income bracket            -> H5 real slopes
  - per item, high-minus-low difference (paired)           -> H5 real gap
  - all items pooled: yes ~ 1 + relative_price + free + 9 item dummies (ref item 9)
Slope = change in P(Yes) over the full 0->1 relative-price range.
Bootstrap: B = 2,000 replications, seed 20261112 (numpy default_rng; same number
as the 02_generate price-order seed but a different RNG stream), respondents
(pids) resampled with replacement; one resample per replication drives all
models, so the high-minus-low difference is paired. Percentile CIs (2.5/97.5).
No model calls (real data only) -> nothing to log under rule 5.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "agent_run"
ITEMS = [9, 10, 15, 23, 25, 26, 27, 31, 36, 38]  # ascending item number
REF_ITEM = 9  # reference item of the pooled fixed effects
LOWEST = "Less than $30,000"
HIGHEST = "$100,000 or more"
INCOME_ORDER = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]
YES = "Yes, I would purchase the product"
NO = "No, I would not purchase the product"
SEED = 20261112
B = 2000


def main():
    # ---- 1. load and re-validate the long table
    long = pd.read_csv(RUN / "agent_real_items_long.csv")
    assert len(long) == 1979 * 10
    assert long["pid"].nunique() == 1979
    assert long.groupby("pid").size().eq(10).all()
    assert set(long["answer_text"]) == {YES, NO}
    assert (long["yes"] == (long["answer_text"] == YES).astype(int)).all()
    assert set(long["QID21"]) <= set(INCOME_ORDER)
    assert set(long["item"]) == set(ITEMS)
    assert long.groupby("item")["ladder_index"].nunique().eq(11).all()
    assert long["ladder_index"].between(0, 10).all()
    assert (long["free"] == (long["price_dollars"] == 0).astype(int)).all()
    assert (long["relative_price"]
            - long["price_dollars"] / long["ladder_max"]).abs().max() < 1e-9
    print(f"loaded agent_real_items_long.csv: {len(long)} rows, 1979 respondents,"
          f" 10 items — all validation checks OK")
    n_br = long.groupby("QID21")["pid"].nunique()
    print("valid respondents by income bracket: "
          + ", ".join(f"{b}: {n_br[b]}" for b in INCOME_ORDER))
    lowest_n, highest_n = int(n_br[LOWEST]), int(n_br[HIGHEST])

    # ---- 2. arrays; rows sorted by pid,item so a pid owns one 10-row block and
    #         every block holds the same 10 items in ascending order
    long = long.sort_values(["pid", "item"], ignore_index=True)
    y = long["yes"].to_numpy(float).reshape(-1, 10)          # (n_pids, 10)
    rp = long["relative_price"].to_numpy(float).reshape(-1, 10)
    fr = long["free"].to_numpy(float).reshape(-1, 10)
    it = long["item"].to_numpy().reshape(-1, 10)
    br = long["QID21"].to_numpy().reshape(-1, 10)
    li = long["ladder_index"].to_numpy().reshape(-1, 10)
    n_pids = y.shape[0]
    assert np.array_equal(it[0], np.array(ITEMS)) and (it == it[0]).all()
    assert (br == br[:, :1]).all()
    br_p = br[:, 0]

    # ---- 3. per-model sufficient statistics (Gram matrices), so each bootstrap
    #         replication costs only two matrix-vector products per model
    # per-item models: design (n_pids, 3) = [1, relative_price, free]
    item_g, item_h = {}, {}
    for j, item in enumerate(ITEMS):
        Xj = np.column_stack([np.ones(n_pids), rp[:, j], fr[:, j]])
        yj = y[:, j]
        item_g[item] = np.einsum("ni,nj->nij", Xj, Xj)      # (n_pids, 3, 3)
        item_h[item] = np.einsum("ni,n->ni", Xj, yj)        # (n_pids, 3)
    # pooled model: design (n_pids, 10, 12) = [1, relative_price, free, 9 dummies]
    cols = [np.ones_like(rp), rp, fr] + [(it == o).astype(float)
                                         for o in ITEMS if o != REF_ITEM]
    Xp = np.stack(cols, axis=-1)                            # (n_pids, 10, 12)
    pooled_g = np.einsum("nri,nrj->nij", Xp, Xp)            # (n_pids, 12, 12)
    pooled_h = np.einsum("nri,nr->ni", Xp, y)               # (n_pids, 12)
    # pooled pointwise curve: per-pid (n_yes, n_obs) sums per ladder_index
    curve_cg = np.empty((n_pids, 22))
    for k in range(11):
        m = (li == k).astype(float)
        curve_cg[:, 2 * k] = np.einsum("nr,nr->n", m * y, np.ones_like(y))
        curve_cg[:, 2 * k + 1] = m.sum(axis=1)

    masks = {"All brackets": np.ones(n_pids, bool),
             LOWEST: br_p == LOWEST,
             HIGHEST: br_p == HIGHEST}

    def fit(w):
        """Weighted OLS for all models at once; w = per-pid multiplicity.
        Returns dict: (item, group) -> beta[3], and 'pooled' -> beta[12]."""
        out = {"pooled": np.linalg.solve(
            np.einsum("n,nij->ij", w, pooled_g),
            np.einsum("n,ni->i", w, pooled_h))}
        for item in ITEMS:
            g, h = item_g[item], item_h[item]
            for grp, m in masks.items():
                wm = w * m
                out[(item, grp)] = np.linalg.solve(
                    np.einsum("n,nij->ij", wm, g), np.einsum("n,ni->i", wm, h))
        return out

    points = fit(np.ones(n_pids))
    slope_p = {k: v[1] for k, v in points.items()}          # col 1 = relative_price
    free_p = {k: v[2] for k, v in points.items() if k != "pooled"}
    pooled_free_point = points["pooled"][2]

    # ---- 4. bootstrap (one pid resample per replication drives all models)
    rng = np.random.default_rng(SEED)
    draws = {k: np.empty((B, 3 if k != "pooled" else 12)) for k in points}
    curve_band = np.empty((B, 11))
    for b in range(B):
        samp = rng.integers(0, n_pids, n_pids)
        w = np.bincount(samp, minlength=n_pids).astype(float)
        beta = fit(w)
        for k, v in beta.items():
            draws[k][b] = v
        tot = np.einsum("n,ni->i", w, curve_cg).reshape(11, 2)
        curve_band[b] = tot[:, 0] / tot[:, 1]
    print(f"bootstrap done: B={B}, seed {SEED}, 31 models + 11 pointwise "
          f"pooled-curve cells")

    def ci(a, axis=0):
        return np.percentile(a, [2.5, 97.5], axis=axis)

    # ---- 5. estimates table
    curves = pd.read_csv(RUN / "agent_real_demand_curves.csv")
    prod_map = curves.drop_duplicates("item").set_index("item")["product"]
    cat_map = curves.drop_duplicates("item").set_index("item")["category"]
    rows = []
    rows.append({"item": "", "category": "", "product": "All 10 items pooled",
                 "group": "All brackets", "n_respondents": n_pids,
                 "n_obs": n_pids * 10, "slope": slope_p["pooled"],
                 "slope_ci_low": ci(draws["pooled"][:, 1])[0],
                 "slope_ci_high": ci(draws["pooled"][:, 1])[1],
                 "free_coef": pooled_free_point,
                 "free_ci_low": ci(draws["pooled"][:, 2])[0],
                 "free_ci_high": ci(draws["pooled"][:, 2])[1]})
    for item in ITEMS:
        n = {"All brackets": n_pids, LOWEST: lowest_n, HIGHEST: highest_n}
        for grp in ["All brackets", LOWEST, HIGHEST]:
            d = draws[(item, grp)]
            lo, hi = ci(d[:, 1])
            flo, fhi = ci(d[:, 2])
            rows.append({"item": item, "category": cat_map[item],
                         "product": prod_map[item], "group": grp,
                         "n_respondents": n[grp], "n_obs": n[grp],
                         "slope": slope_p[(item, grp)],
                         "slope_ci_low": lo, "slope_ci_high": hi,
                         "free_coef": free_p[(item, grp)],
                         "free_ci_low": flo, "free_ci_high": fhi})
        d = draws[(item, HIGHEST)][:, 1] - draws[(item, LOWEST)][:, 1]
        lo, hi = ci(d)
        rows.append({"item": item, "category": cat_map[item],
                     "product": prod_map[item],
                     "group": "high-minus-low difference",
                     "n_respondents": lowest_n + highest_n,
                     "n_obs": lowest_n + highest_n,
                     "slope": slope_p[(item, HIGHEST)] - slope_p[(item, LOWEST)],
                     "slope_ci_low": lo, "slope_ci_high": hi,
                     "free_coef": "", "free_ci_low": "", "free_ci_high": ""})
    est = pd.DataFrame(rows)
    est.to_csv(RUN / "agent_real_slopes.csv", index=False)
    print(f"wrote {RUN / 'agent_real_slopes.csv'} ({len(est)} rows)")

    # ---- 6. pooled pointwise curve with bootstrap band
    tot = curve_cg.sum(axis=0).reshape(11, 2)
    pooled_curve = pd.DataFrame({
        "ladder_index": np.arange(11),
        "relative_price": np.arange(11) / 10,
        "n": tot[:, 1].astype(int),
        "n_yes": tot[:, 0].astype(int),
        "share_yes": tot[:, 0] / tot[:, 1],
        "band_low": ci(curve_band)[0], "band_high": ci(curve_band)[1]})
    pooled_curve.to_csv(RUN / "agent_real_pooled_curve.csv", index=False)
    print(f"wrote {RUN / 'agent_real_pooled_curve.csv'} ({len(pooled_curve)} rows)")

    # ---- 7. figures
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharey=True)
    bracket_colors = dict(zip(INCOME_ORDER,
                              ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                               "#d62728"]))
    for ax, item in zip(axes.ravel(), ITEMS):
        c = curves[curves["item"] == item]
        for b in INCOME_ORDER:
            cb = c[c["QID21"] == b]
            ax.plot(cb["price_dollars"], cb["share_yes"], "-o", ms=2.5,
                    lw=0.9, color=bracket_colors[b], label=b, alpha=0.85)
        ca = c[c["QID21"] == "All brackets"]
        ax.plot(ca["price_dollars"], ca["share_yes"], "-o", ms=3.5, lw=1.8,
                color="black", label="All brackets")
        name = str(prod_map[item])
        ax.set_title(f"item {item} — {name[:38]}", fontsize=7.5)
        ax.set_xlabel("price shown ($)")
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("share Yes")
    axes[1, 0].set_ylabel("share Yes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Real demand curves by income bracket — 1979 valid respondents, "
                 "one random price per respondent per item", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(RUN / "agent_real_curves_by_item.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.fill_between(pooled_curve["relative_price"], pooled_curve["band_low"],
                    pooled_curve["band_high"], alpha=0.25, color="#1f77b4",
                    label="95% bootstrap band (respondent resample)")
    ax.plot(pooled_curve["relative_price"], pooled_curve["share_yes"], "-o",
            color="#1f77b4", label="real, all 10 items pooled")
    ax.set_xlabel("relative price (price / item's ladder maximum)")
    ax.set_ylabel("share Yes")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Pooled real demand curve (1979 respondents, 19,790 observations)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN / "agent_real_pooled_curve.png", dpi=200)
    plt.close(fig)
    print("wrote agent_real_curves_by_item.png and agent_real_pooled_curve.png")

    # ---- 8. stdout: the estimates (this step's deliverable, as in plan 4g)
    pd.set_option("display.width", 160)
    show = est.copy()
    show["slope [95% CI]"] = show.apply(
        lambda r: f"{r['slope']:+.3f} [{r['slope_ci_low']:+.3f}, "
                  f"{r['slope_ci_high']:+.3f}]", axis=1)
    show["free [95% CI]"] = show.apply(
        lambda r: ("" if r["free_coef"] == "" else
                   f"{r['free_coef']:+.3f} [{r['free_ci_low']:+.3f}, "
                   f"{r['free_ci_high']:+.3f}]"), axis=1)
    cols = ["item", "group", "n_respondents", "n_obs", "slope [95% CI]",
            "free [95% CI]"]
    print(show[cols].to_string(index=False))
    diff = est[est["group"] == "high-minus-low difference"]
    covering = diff[(diff["slope_ci_low"] <= 0) & (diff["slope_ci_high"] >= 0)]
    print(f"high-minus-low difference CIs covering 0: {len(covering)} of "
          f"{len(diff)} items (per plan.md section 6 these cannot test the "
          f"H5 direction on the real side)")
    h4_out = est[(est["group"] == "All brackets") & (est["item"] != "")]
    neg = h4_out[h4_out["slope_ci_high"] < 0]
    print(f"per-item pooled-bracket slopes strictly negative (CI < 0): "
          f"{len(neg)} of {len(h4_out)}")


if __name__ == "__main__":
    main()
