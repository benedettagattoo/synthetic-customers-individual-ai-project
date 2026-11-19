# plan.md

Idea 12 – Synthetic customers. Dataset: Twin-2K-500 (LLM-Digital-Twin/Twin-2K-500 on Hugging Face), files in `data/raw/`.

## 1. Question
Do LLM-generated "synthetic customers", given only the demographics of real respondents, reproduce the **differences between groups** in purchase intention that we see in the real data?

Main grouping: **family income** (QID21, 5 brackets, from `wave1_3_response_label.csv`).
Secondary grouping (only if time allows): **age** (QID13, 4 brackets).
Outcome: purchase intention, Yes/No, in the pricing study (QID9_1 ... QID9_40, from `wave4_response_label.csv`).

The question is about groups, not individuals: I do not test whether each synthetic person matches its real "twin", only whether the gaps between groups match.


## 2. Theory (what I expect and why)
- Argyle et al. (2023): models given people's backgrounds reproduce how groups answer → the **direction** of income gaps should mostly match.
- Bisbee et al. (2024), Wang et al. (2025): synthetic answers have too little variation and flatten within-group differences; Brand, Israeli & Ngwe: synthetic willingness to pay is often off → the **size** of the gaps will not match (I expect it to be exaggerated or flattened), and the overall purchase rate may be off.

Hypotheses:
- H1 (direction): for most products, the sign of the high-income minus low-income gap in purchase rate is the same for synthetic and real customers.
- H2 (size): the synthetic gaps differ in size from the real ones.
- H3 (level): the overall synthetic purchase rate differs from the real one.
- H4 (price sensitivity): the slope of the Yes share on price is the same for synthetic and real customers.
- H5 (income gradient in price sensitivity): the (highest minus lowest income bracket) difference in that slope has the same sign for synthetic and real customers.

## 3. Data and sample
1. Merge demographics (wave 1–3 file) with purchase answers (wave 4 file) on `pid`. Print rows before and after the merge.
1b. Load `data/raw/wave_4_labels_anonymized.csv` (`TWIN_ID` = `pid`; data start at row 4).
    **Malformed strings:** 79 of 2058 respondents have a `randDollarsString` with 41–44
    "item:price" pairs (one item repeated with two different prices). Exclude these 79
    from every analysis that uses `randDollarsString` (product/price mapping); keep them only for
    analyses that do not touch prices. The script must print how many were excluded and
    why (verified count: 1979 valid, 79 excluded).
2. Products: 10 of the 40 products, chosen by a fixed rule (every 4th display position,
   starting from position 1: positions 1, 5, 9, …, 37) applied to **respondent 1's (pid 1)
   display order** in `data/raw/wave_4_labels_anonymized.csv` (`randDollarsString`) —
   i.e. **items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10**.

   **Correction (11 Nov 2026).** The `QID9_k` columns are NOT products. `QID9_k` in
   `wave4_response_label.csv` = raw column `k_Q295` = the answer to the k-th question
   *displayed*; the display order is randomised per respondent (2058 distinct orders in
   `randDollarsString`), and the same per-respondent order is reused in wave 3
   (same-column wave-3 vs wave-4 answers agree for 84.4% of respondents, far above the
   ~50% expected under independent re-randomisation). The catalog in
   `question_catalog.json` lists, for `QID9_k`, the product shown at position k **to
   respondent 1** and that respondent's price at position k (verified 40/40). The
   products must therefore be tracked by **item number** (the product index in
   `randDollarsString`), not by column; the selection rule ("every 4th, starting at 1")
   is unchanged, only its unit changed from column to respondent 1's display position.

   | Item no. | Category | Product | Catalog price (respondent 1's draw) | Item's price ladder |
   |---|---|---|---|---|
   | 25 | Dairy Products | Land O Lakes Salted Stick Butter, 16 oz, 4 sticks | $7.39 | 11 points, $0–$10.56 |
   | 27 | Refrigerated Deli Meats | Goya Cooked Ham 16 oz | $53.98 | 11 points, $0–$59.98 |
   | 31 | Fresh Fruit | Fresh Raspberries, 12 oz container | $0.95 | 11 points, $0–$9.48 |
   | 9 | Cereal - Ready to Eat | Cinnamon Toast Crunch, Family Size, 18.8 oz | $9.86 | 11 points, $0–$9.86 |
   | 36 | Toilet Tissue | Angel Soft Toilet Paper, 9 Mega Rolls | $0.00 | 11 points, $0–$13.36 |
   | 26 | Yogurt - Refrigerated | Chobani Non-Fat Greek Yogurt, Vanilla, 32 oz | $0.00 | 11 points, $0–$11.16 |
   | 15 | Candy - Non-Chocolate | HARIBO Goldbears, 28.8 oz bag | $12.96 | 11 points, $0–$12.96 |
   | 23 | Refrigerated Bacon | Oscar Mayer Fully Cooked Original Bacon, 2.52 oz | $4.27 | 11 points, $0–$8.54 |
   | 38 | Batteries | Duracell Coppertop AA, 16 pack | $25.55 | 11 points, $0–$31.94 |
   | 10 | Cookies | Little Debbie Oatmeal Creme Pies, 12 ct | $3.75 | 11 points, $0–$5.36 |

   What respondents saw (wording unchanged from the catalog): intro QID8 ("You will now
   be asked a series of 40 questions ... Please answer truthfully according to your
   preferences."), then for each product: "Please consider the following product
   category: [category]. Suppose you are in a grocery store, and you see the following
   product in that category: [product]. The product is priced at: $[price]. Would you or
   would you not purchase this product?" Options: "Yes, I would purchase the product" /
   "No, I would not purchase the product" — except that the price is **not** the catalog
   price: each respondent saw, for each item, a random draw from that item's own 11-point
   ladder {0, M/10, M/10×2, …, M} (≈9% of draws are $0, product free).

   **RESOLVED — prices (11 Nov 2026; replaces the former "OPEN PROBLEM – prices").**
   Per-respondent prices exist: `data/raw/wave_4_labels_anonymized.csv`, column
   `randDollarsString`, 40 "item:price" pairs in the order the products were displayed.
   For respondent r and item i: the **price** is the "i:$p" pair in r's
   `randDollarsString`; the **answer** is the column at the display position where item
   i appears in r's string (raw column `{pos}_Q295` in the raw file = `QID9_{pos}` in
   `wave4_response_label.csv`). Verified read-only: pairing answers with same-position prices reproduces a strong monotone
   price–answer relationship; pairing with item-number prices does not.

## 4. Task for this run (real data only)
Only the real-data part of the analysis is done here; there are no synthetic answers in this folder.
For each of the 10 products above, using all valid respondents:
- the real demand curve: share of "Yes" at each of the 11 ladder prices, overall and by income bracket (QID21);
- the slope of the Yes share on price for each product, with a 95% confidence interval;
- the highest-minus-lowest income bracket difference in that slope, with a 95% confidence interval.
How to code price, which model to estimate, how to compute confidence intervals and every other analysis choice are left to the analyst, and must be listed.

## 5. What would prove me wrong
- H4 is wrong if, for half or more of the 10 products, the synthetic slope lies outside the 95% confidence interval of the real slope, or has the opposite sign.
- H5 is wrong if the sign of the (highest − lowest income) slope gap disagrees for half or more of the products. If the real slope gap for a product has a 95% CI that includes 0, that product cannot test direction and is reported separately.
