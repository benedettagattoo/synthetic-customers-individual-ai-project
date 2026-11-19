# choices.md — analysis choices made for the real-data task (plan.md section 4)

Made by the agent without asking; one line each on why. Scripts: `src/01_prepare.py` (steps 1–7),
`src/02_demand_curves.py` (steps 8–11), `src/03_slopes.py` (steps 12–18). All numbers below are printed by those
scripts or stored in `results/`.

1. **Sample / exclusion (n = 1979).** All 2058 respondents merged 1:1 on `pid` (2058 in, 2058 after merge);
   the 79 respondents with a malformed `randDollarsString` (41–44 pairs, one item repeated with two different
   prices) are excluded from every analysis in this run, exactly as pre-specified in plan.md section 3.1b,
   because their item→price mapping is ambiguous — this matches the verified count 1979 valid / 79 excluded.
2. **No other exclusions.** No attention-check filter is applied: the raw export has `attention_1`/`attention_2`
   columns, but neither plan.md nor data/README.md defines an exclusion rule based on them, and defining a new
   data-driven exclusion rule post hoc would be researcher degrees of freedom; all 1979 valid respondents are used
   for all 10 products (no product-level further exclusions).
3. **Product identification.** The 10 products are the **items** at respondent 1's display positions
   1, 5, …, 37 (items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10), per the 11 Nov 2026 correction in plan.md:
   `QID9_k` is a display position, not a product, so products must be tracked by item number, not by column.
4. **Answer–price pairing.** For respondent r and item i: the price is taken from the `i:$p` pair in r's
   `randDollarsString`, and the answer is `QID9_{pos}` from `wave4_response_label.csv`, where pos is the display
   position of item i in r's string (position = order index of the pair, 1–40). Verified as in plan.md: the pooled
   Yes rate falls monotonically from 0.863 at the free ladder point to 0.171 at the maximum (printed by
   01_prepare.py), confirming the pairing is correct.
5. **Price coding (primary): dollars.** Price is the dollar amount drawn from the item's own 11-point ladder
   {0, M/10, …, M}; slopes are per **$1**. This is the scale on which real vs synthetic slopes will be compared
   for H4 (same product, same ladder) and it is the scale consumers face.
6. **Ladder ranks.** Each item's 11 distinct drawn prices are verified (11 per item, exactly) and converted to a
   ladder rank 0–10 (0 = free, 10 = item max) purely for aligning the demand-curve points across items;
   the ladder max M is defined empirically as the item's maximum drawn price (equals the catalog ladder max).
7. **Income coding.** QID21 is used as the 5 survey brackets in the survey's own order
   (Less than $30,000 … $100,000 or more); no recoding, collapsing, or midpoint imputation, because the analysis
   asks for the highest-vs-lowest bracket contrast and collapsing would throw away information. Bracket sizes in
   the valid sample are 347/392/401/307/532 (printed by 01_prepare.py's long-file summary); full-data sizes
   367/412/411/316/552 before the malformed-string exclusion.
8. **Outcome coding.** Yes = 1, No = 0, matched on the exact label strings
   ("Yes, I would purchase the product" / "No, I would not purchase the product"); verified that these are the
   only two values in all 40 wave-4 columns.
9. **Demand curve estimator (steps 8–9).** The demand curve is the raw (unmodelled) share of Yes at each of the 11
   ladder points, overall and per bracket — no smoothing or curve fitting, because the task asks for the share at
   each price, and a model fit would impose a functional form on it.
10. **Empty cells.** All 550 bracket × item × ladder cells are non-empty; the smallest cell has n = 15
    (item 27, $75,000–$100,000, rank 8), median cell n = 35; the smallest overall item × ladder cell has n = 140.
    No imputation or smoothing is needed; every point plotted is directly observed.
11. **Uncertainty on curve points.** Binomial SEs of the Yes share are stored with each curve point
    (`demand_curves_overall.csv`, `demand_curves_by_income.csv`) for completeness, but no per-point CIs are
    drawn: the inferential CIs the task asks for are on the slopes.
12. **Slope model (steps 12–13).** OLS linear-probability model: yes ~ price, slope = change in the Yes share per
    $1. A linear model is chosen because the task asks for "the slope of Yes share on price" — the LPM slope is
    exactly that quantity — and because logistic or probit slopes would not be interpretable as a change in the
    Yes share itself. Linearity in price is an approximation (some curves show a steep drop near the bottom of the
    ladder); the demand-curve tables let a reader judge departures from linearity.
13. **Independence / clustering.** No clustering of SEs: within a given product's regression, each respondent
    contributes exactly one observation (one answer at one price draw), so rows are independent; HC1-robust SEs
    handle the binary-outcome heteroskedasticity. (Respondents do contribute 10 rows across the 10 products, but
    the 10 product-level regressions are estimated and reported separately, so cross-product correlation does not
    affect any single estimate.)
14. **Confidence intervals (primary).** 95% two-sided Wald intervals using normal critical value 1.959964 and
    HC1 heteroskedasticity-robust standard errors (statsmodels `cov_type="HC1"`, version 0.15.0), for both the
    slope and the slope gap. HC1 is the default robust small-sample correction for a large cross-section (n = 1979);
    the normal approximation is fine at this n.
15. **Income-bracket slope gap.** Estimated from the interaction model
    yes ~ price + bracket dummies + price × bracket (reference = lowest bracket, "Less than $30,000"); the
    coefficient on price × "$100,000 or more" is the highest-minus-lowest bracket difference in slopes, and its
    HC1 Wald CI is reported directly from that model (single model rather than two separate slopes + a difference
    test, so the covariance between the two slopes is accounted for).
16. **Multiple comparisons.** No multiplicity adjustment across the 10 products: each product has its own ladder
    and its own falsification rule in plan.md section 5 (per-product CI for H4, per-product sign for H5), and the
    hypotheses are about the fraction of products failing, not about any single product being "significant".
17. **Bootstrap sensitivity for the gap.** Pairs (respondent-level) bootstrap of the slope-gap coefficient:
    resample the 1979 rows of a product with replacement 2000 times (replications = 2000, seed = 20261117,
    fixed per AGENTS.md rule 5), re-estimate the gap by plain least squares (no SEs needed), take percentile 2.5/97.5
    CIs. Purpose: check that the Wald CI is not badly off for the interaction model; bootstrap and Wald CIs agree
    closely in every product (means within 0.0005; e.g. item 23: Wald [−0.0469, −0.0029] vs bootstrap
    [−0.0469, −0.0025]) — the analytic Wald CIs are therefore kept as primary.
18. **Price-scale sensitivity for the slopes.** The same slope and gap models are re-estimated with price coded as
    price / ladder max (a 0–1 scale, "per full ladder") in `slopes_and_income_gaps_share_of_max.csv`; dollar slopes
    are not comparable across products with very different ladder maxima (e.g. $0–$5.36 cookies vs $0–$59.98 ham),
    while the primary comparison for H4/H5 is within-product, so dollars stay primary. Income-gap signs are
    identical under both codings.
19. **No re-weighting.** No post-stratification or weighting of respondents toward population targets: the task
    compares group differences within this sample, and the sample composition affects real and synthetic arms
    symmetrically.
20. **Output order and naming.** Tables and figures are ordered by the plan's product order
    (items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10) and labelled with item numbers, since the item-number →
    product-name mapping is consistent with all price evidence but could not be independently verified
    (caveat in data/README.md).
21. **Seeds / reproducibility.** The only randomness is the bootstrap (seed 20261117, choice 17). No model calls
    to an LLM occur in this run (real data only), so no prompts/outputs to log; no `.env` or API key is touched.
    Environment: Python 3.14.6 via `uv run`, pandas 3.1.1, statsmodels 0.15.0, numpy 2.5.4, matplotlib 3.11.2.

## Result files (which script made which)
- `results/demand_curves_overall.csv` (110 rows), `results/demand_curves_by_income.csv` (550 rows),
  `results/fig_demand_curves_overall.png`, `results/fig_demand_curves_by_income.png` — `src/02_demand_curves.py`.
- `results/slopes_and_income_gaps.csv` (primary, per $1), `results/slopes_and_income_gaps_share_of_max.csv`
  (sensitivity), `results/slope_gap_bootstrap_sensitivity.csv` (bootstrap check),
  `results/fig_slopes_and_gaps.png` — `src/03_slopes.py`.
- `data/derived/wave4_target_items_long.csv` (19,790 rows = 1979 × 10) — `src/01_prepare.py`.
