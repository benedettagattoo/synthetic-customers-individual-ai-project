# data/README.md — Columns used by this project

Dataset: `LLM-Digital-Twin/Twin-2K-500` (Toubia et al. 2025, arXiv 2505.17479), `wave_split` config.
Only the columns this project uses are described here. Provenance of each fact:
**[catalog]** = from `data/raw/question_catalog.json` (or `question_catalog_README.md`), **[CSV]** = from the CSVs.
All values verified with `uv run python` (session logs, 10–11 Nov 2026).

## Files

| File | Rows | Relevant columns | Content |
|---|---|---|---|
| `wave1_3_response_label.csv` | 2058 [CSV] | pid, QID11, QID12, QID13, QID14, QID21 (demographics) + QID9_1, QID9_5, QID9_9, QID9_13, QID9_17, QID9_21, QID9_25, QID9_29, QID9_33, QID9_37 (purchase answers, wave 3) | Label (text) version of a subset of columns |
| `wave4_response_label.csv` | 2058 [CSV] | pid + QID9_1, QID9_5, QID9_9, QID9_13, QID9_17, QID9_21, QID9_25, QID9_29, QID9_33, QID9_37 (purchase answers, wave 4) | Label (text) version of a subset of columns |
| `wave_4_labels_anonymized.csv` | 2058 data rows (2061 lines: row 1 = column names, rows 2–3 = question text / ImportId, data from row 4) [CSV] | `TWIN_ID` (= `pid`), `randDollarsString`, `{k}_Q295` for k = 1…40 (wave-4 answers in display order) | Original Qualtrics export of wave 4 |

- **One row = one survey respondent.** [catalog]
- Both files have `pid` as the first column, participant ID 1–2058. [catalog + CSV]
- Merge: **join the two files on `pid`** (see plan.md section 3). Both files have 2058 rows, all `pid` values unique,
  no empty `pid`, and the two `pid` sets are identical — the merge should be 1:1 and lose no rows (still verify and
  print row counts after the merge). [CSV]
  `wave4_response_label.csv` has 127 columns in total, including pid. [CSV]
- `pid` in both files is an anonymised sequential number (1–2058) [dataset card on Hugging Face; numbers checked in
  CSV]; we still replace it with a random ID before anything is sent to a model as a precaution
  (plan.md section 3, item 2).

## Demographics — only in `wave1_3_response_label.csv`

All five are single-choice questions (type `MC`, selector `SAVR`, force response ON, block "Demographics",
source `wave1_3_persona_json`). [catalog] Values below are the exact option texts from the catalog; all appeared
in the CSV and none is missing in any row. [CSV]

| Column | Question text [catalog] | Possible values, exactly as in the CSV [catalog options = CSV values] | Missing [CSV] | Units |
|---|---|---|---|---|
| QID11 (region) | "Which part of the United States do you currently live in?" | Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME); Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH); South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, NC, SC, GA, FL); West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM); Pacific (HI, AK) | 0 of 2058 | US census-style region |
| QID12 (sex) | "What is the sex that you were assigned at birth?" | Male; Female | 0 of 2058 | binary category |
| QID13 (age) | "How old are you?" | 18-29; 30-49; 50-64; 65+ | 0 of 2058 | age brackets, not raw age |
| QID14 (education) | "What is the highest level of schooling or degree that you have completed?" | Less than high school; High school graduate; Some college, no degree; Associate's degree; College graduate/some postgrad; Postgraduate | 0 of 2058 | education categories |
| QID21 (family income) | "Last year, what was your total family income from all sources, before taxes?" | Less than $30,000; $30,000-$50,000; $50,000-$75,000; $75,000-$100,000; $100,000 or more | 0 of 2058 | income brackets in US dollars (the CSV stores only the bracket text, not an amount) |

## Purchase decisions (QID9) — in BOTH files

The pricing study: an intro (QID8, a descriptive block "DB", no answer collected) says "You will now be asked a
series of 40 questions. … There is no right or wrong answer. Please answer truthfully according to your
preferences." [catalog] Each product question then asks
"Suppose you are in a grocery store … Would you or would you not purchase this product?" with options
1 = "Yes, I would purchase the product", 2 = "No, I would not purchase the product" (type MC). [catalog]

`QID9_1` through `QID9_40` are the 40 product questions; here only the 10 that this project uses
(chosen by the fixed rule in plan.md section 3) are listed. All 10 exist as columns in **both** files, and in
fact all 40 of QID9_1–QID9_40 exist in both files (none missing in either file). [CSV]
In `wave1_3_response_label.csv` they are the **wave 3** answers; in `wave4_response_label.csv` the **wave 4**
answers — the same respondents answered the same questions twice. [catalog README + CSV]
The listed values are the only distinct values in the CSVs; **0 of 2058 missing/empty** for each column
in each file. [CSV] Units: Yes/No choice, no scale.

| Column | Category | Product in the question text [catalog] | Price in catalog [catalog] |
|---|---|---|---|
| QID9_1 | Dairy Products | Land O Lakes Salted Stick Butter, 16 oz, 4 Sticks | $7.39 |
| QID9_5 | Refrigerated Deli Meats | Goya Cooked Ham 16 oz | $53.98 |
| QID9_9 | Fresh Fruit | Fresh Raspberries, 12 oz Container | $0.95 |
| QID9_13 | Cereal - Ready to Eat | Cinnamon Toast Crunch Breakfast Cereal, Crispy Cinnamon Cereal, Family Size, 18.8 oz | $9.86 |
| QID9_17 | Toilet Tissue | Angel Soft Toilet Paper, 9 Mega Rolls, Soft and Strong Toilet Tissue | $0.00 |
| QID9_21 | Yogurt - Refrigerated | Chobani Non-Fat Greek Yogurt, Vanilla Blended 32 oz, Plastic | $0.00 |
| QID9_25 | Candy - Non-Chocolate | HARIBO Goldbears Original Gummy Bears, 28.8oz Stand Up Bag | $12.96 |
| QID9_29 | Refrigerated Bacon | Oscar Mayer Fully Cooked Original Bacon, 2.52 oz Box | $4.27 |
| QID9_33 | Batteries | Duracell Coppertop AA Battery, Long Lasting Double A Batteries, 16 Pack | $25.55 |
| QID9_37 | Cookies | Little Debbie Oatmeal Creme Pies, 12 ct, 16.2 oz | $3.75 |

Option order is identical for all 10 columns: option 1 "Yes, I would purchase the product",
option 2 "No, I would not purchase the product". [catalog]

**Correction (11 Nov 2026): `QID9_k` is a display position, not a product.**
`QID9_k` in `wave4_response_label.csv` is identical to raw column `k_Q295` in
`wave_4_labels_anonymized.csv` (2058/2058 rows, all 40 k). The wave-4 questions were a
randomised loop: the product shown at position k differs per respondent (2058 distinct
orders in `randDollarsString`); each respondent's order is reused in wave 3
(same-column agreement 84.4%). The product names and prices in the table above and in
`question_catalog.json` are what **respondent 1 (pid 1)** saw at those positions
(catalog price = pid 1's price at position k, 40/40; for all other respondents at
chance, 0.79%). The 10 products of this project are therefore identified by **item
number** — the items at respondent 1's display positions 1, 5, 9, …, 37:
items 25, 27, 31, 9, 36, 26, 15, 23, 38, 10 (products as in the table above).
For any product-level analysis, a respondent's answer and price for item i must be
looked up via the position of item i in that respondent's `randDollarsString`
(see "Prices" below). Caveat: the item-number → product-name assignment follows the
catalog texts and is consistent with all price evidence, but could not be verified
against an independent item→product list.

## Prices per respondent — in `wave_4_labels_anonymized.csv` (resolves the former "known problem")

Column `randDollarsString` holds, per respondent, 40 "item:price" pairs **in the order
the products were displayed** (the leading `` `""` `` is a stray empty field). The item
number is the product index. Prices are drawn per respondent from the item's own
11-point ladder {0, M/10, M/10×2, …, M}, where M is the item's maximum price (e.g.
item 27: $0, $6.00, …, $59.98; item 5: $0–$1.76); ≈9% of draws are $0 (product free),
so many products can appear free — the catalog's two $0.00 entries are just respondent
1's draws at positions 17 and 21. Verified: pairing answers with same-position prices
reproduces a strong monotone price–answer relationship; pairing with item-number
prices does not; and when the shown price was $0, the purchase share matches the
positional pairing rather than the item-number pairing (checked by the user with
OpenCode: 81.6% vs 45.5% Yes when the price was $0, 1979 valid rows; first spotted
with the help of Claude).

**Data quality:** 79 of 2058 respondents have a malformed `randDollarsString`
(41–44 pairs, one item repeated with two different prices); they are excluded from all
price-based analyses (1979 valid rows), and any script prints the count and reason.

## Summary of missing values

`pid`, QID11, QID12, QID13, QID14, QID21 and all ten QID9 columns used here have **0 missing/empty values** in
2058 rows (demographics checked in the wave1_3 file only — these columns do not exist in the wave4 file). [CSV]
