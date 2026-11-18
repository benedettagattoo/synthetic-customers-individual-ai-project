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

## 3. Data and sample
1. Merge demographics (wave 1–3 file) with purchase answers (wave 4 file) on `pid`. Print rows before and after the merge.
1b. Load `data/raw/wave_4_labels_anonymized.csv` (`TWIN_ID` = `pid`; data start at row 4).
    **Malformed strings:** 79 of 2058 respondents have a `randDollarsString` with 41–44
    "item:price" pairs (one item repeated with two different prices). Exclude these 79
    from every analysis that uses `randDollarsString` (product/price mapping, the
    stratified sample of 200 in item 3, and the synthetic step); keep them only for
    analyses that do not touch prices. The script must print how many were excluded and
    why (verified count: 1979 valid, 79 excluded).
2. Replace `pid` with a new random ID before any row is sent to a model.
3. Sample: 200 respondents, stratified by income bracket (40 per bracket), fixed random seed.
4. Products: 10 of the 40 products, chosen by a fixed rule (every 4th display position,
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
   `wave4_response_label.csv`). Verified read-only: pairing answers with same-position
   prices reproduces a strong monotone price–answer relationship, pairing with
   item-number prices does not; and when the price shown was $0, the purchase share
   agrees with the positional pairing, not the item-number pairing (checked by the user
   with OpenCode: 81.6% vs 45.5% Yes when the price was $0, 1979 valid rows; first
   spotted with the help of Claude). Decision (a) from the former open problem is
   adopted: each synthetic twin gets the price (and product order) its real twin saw.

## 4. Synthetic customers
- One model call per synthetic person **and product** (10 calls per person): persona =
  region, sex, age bracket, education, income bracket (QID11, QID12, QID13, QID14,
  QID21); then the product question, asked **11 times — once per price point of that
  item's ladder {0, M/10, …, M}** — with the same wording and options as the real
  survey; answers returned as JSON.
- **Gabor–Granger full ladder (12 Nov 2026; replaces the former rule "each twin answers
  each product once, at the price its real twin saw, in the real twin's product
  order"):** every synthetic twin answers each of the 10 products at **all 11 price
  points of that item's ladder**, not only at its real twin's price. The real twin's
  price is not marked or treated specially in the prompt; it is used only afterwards,
  in the analysis. **Because each product is a separate call, the real twin's product
  order is no longer used** (the 11 Nov rule of presenting the 10 products in the real
  twin's display order belonged to the single per-person call and is superseded; the
  catalog price is still not used, and only the price order within a call is
  randomised on the synthetic side). Real twins are drawn only from the 1979
  respondents with a well-formed `randDollarsString` (plan.md section 3, item 1b). The
  same answers support two comparisons:
  1. **Twin by twin at the real twin's price** (the current plan; sections 5–6, H1–H3):
     from the 11 answers, take the one at the price the real twin saw.
  2. **Demand curves** (new, H4–H5 in section 5): for each product and income bracket,
     the share of "Yes" at each of the 11 prices — synthetic (200 twins × 11 prices) vs
     real (the 1979 valid respondents, each of whom saw one random price per product).
- Presentation inside one call (twin × product): persona block, then the real survey
  intro (QID8 wording), then the 11 product questions, one per price, each with the
  exact survey wording ("Please consider the following product category: … The product
  is priced at: $[price]. Would you or would you not purchase this product?"), options
  unchanged. Details:
  - **Price order randomised** per (twin, product) with a fixed seed (20261112), not
    ascending: the real design shows each respondent one *random* price, and a fixed
    ascending ladder would always put the $0 question first (anchoring). Randomisation
    also gives the real twin's price a random position, so comparison 1 has no
    systematic position effect.
  - **Prices written as the real prices:** nonzero prices with two decimals as in the
    catalog ("$6.00", "$59.98"); **$0 exactly as in `randDollarsString`: "$0 (product
    is free)"** (verified read-only in the raw file); no other "free" label added.
  - **JSON output validated:** the script checks that all 11 ladder prices are present,
    each answered "Yes…" or "No…"; malformed calls are retried once, then excluded,
    and the count is printed (rule 2).
- Model: Google AI Studio, free tier, a Gemini **Flash-Lite** model (Ollama gpt-oss:20b is not an option: this Mac has 8 GB of memory, the model needs about 16 GB).
  Exact model ID: `gemini-3.5-flash-lite` (version `3.5-flash-lite-07-2026`, display name "Gemini 3.5 Flash Lite"; read from the Gemini API models endpoint on 12 Nov 2026). Not used: `gemini-flash-lite-latest` (alias that can be re-routed), `-preview`, `-image`, `-tts` variants.
  Temperature: default, value: 1 (as reported by the models endpoint; set explicitly to 1 in the script so a change of default cannot alter it).
  API key: stored only in a `.env` file listed in `.gitignore`; never in chat, scripts or logs.
  Date fixed: 12 Nov 2026, 09:10 (before any synthetic answer was generated).
- **Volume: 200 twins × 10 products = 2,000 calls**, each returning 11 Yes/No answers;
  the robustness re-run (section 5) adds up to 2,000 more.
  **Free-tier limits (checked in AI Studio, 12 Nov 2026):** `gemini-3.5-flash-lite`
  15 requests per minute, 500 requests per day; `gemini-3.1-flash-lite` also 500 per day
  (separate limit). **Schedule:** the main run is split over 5 Pacific days (Thu 12 – Mon 16 Nov): 470 main calls on day 1 after the 30-call pilot, 500 on days 2–4, the last 30 on day 5 (Monday after 09:00 Italian time).
  At most 500 requests a day, counted by Pacific-time date (the free tier resets at midnight Pacific = 09:00 Italian time; the pilot and main run share the day-1 quota), and fewer than 15 a minute; the script saves each answer as it
  arrives and restarts from where it stopped (same seed, one log per day). Before the
  main run, a pilot of 3 twins × 10 products (30 calls) checks the prompt and the JSON
  format; pilot answers are kept in a separate file and are not used in the analysis.
- Calls run from a script in `src/` (Ollama gpt-oss:20b or Google AI Studio Flash-Lite, per the brief), never with personal data. Every prompt, output and model name is logged.

## 5. Test
**Comparison 1 — twin by twin at the real twin's price** (unchanged in definition; the
answer is now read out of the twin's 11 ladder responses, at the price the real twin
saw). For each product and income bracket: share of "Yes", real vs synthetic.
- H1: share of products where the sign of the (highest minus lowest income bracket) gap agrees; correlation across the 10 products between real and synthetic gaps.
- H2: compare the size of the gaps (synthetic gap / real gap).
- H3: overall purchase rate, real vs synthetic.
- Regression (linear probability model): purchase ~ income bracket × source (real / synthetic) + product fixed effects + **price shown**, standard errors clustered by person. The interaction terms measure whether the income gradient differs between real and synthetic customers.
  **Price control (11 Nov 2026):** prices differ per respondent and per product, so the price each respondent was shown is added as a control. Because price ladders differ across products (e.g. $0–$5.36 for the oatmeal creme pies, item 10, vs $0–$59.98 for the ham, item 27), the price enters as the **relative price** (price ÷ item's ladder maximum M, so 0–1 and comparable across products), plus a **"free" indicator** (1 if the price shown was $0) next to it, because the jump in Yes at $0 is large. The product fixed effects absorb the ladder levels. In comparison 1 each synthetic twin is read at the same price as its real twin, so price is identical within a twin pair and cannot mimic the source effect.

**Comparison 2 — demand curves (new, 12 Nov 2026):**
- For each product: the **real curve** is the share of "Yes" at each of the 11 ladder
  prices among the 1979 valid respondents (each contributes one observation per
  product, at its random draw); the **synthetic curve** is the share of "Yes" at each
  price among the 200 twins (each contributes one answer per price). Curves are drawn
  in dollars per product and against relative price (price ÷ M) pooled across products;
  by income bracket at least for the highest vs lowest bracket. Cell counts per
  (product, bracket, price) are printed (rule 2).
- **H4 (price sensitivity):** the slope of the Yes-share on relative price is the same
  for real and synthetic customers, per product (income brackets pooled). Estimated
  with a linear probability model: purchase ~ source × relative_price + product fixed
  effects + source × free indicator; standard errors clustered by person (real:
  respondent; synthetic: twin, whose 11 answers per product are correlated). The
  source × relative_price coefficient is the synthetic-minus-real slope difference.
  **Clarification (16 Nov 2026, before src/08 ran):** the model also includes the
  `source` main effect (purchase ~ source + relative_price + source × relative_price +
  product FE + free + source × free); without it the interaction would absorb the
  overall synthetic-vs-real level difference.
- **H5 (income gradient in price sensitivity):** the (highest minus lowest income
  bracket) difference in that slope agrees in sign between real and synthetic, for most
  products. **This test exists only because every twin now answers at every price**;
  under the former one-price design each twin × product gave a single point.
- **Real-curve noise:** ≈1,979 respondents split over 5 brackets × 11 prices gives
  ≈36 respondents per (product, bracket, price) cell, so per-price shares are noisy.
  H4/H5 are therefore judged on slopes with confidence intervals (bootstrap by
  respondent for the real side, by twin for the synthetic), not on the 11 per-price
  points one by one.
- Robustness: re-run the synthetic step with a second model and compare (2,000 further
  calls). **Second model fixed on 15 Nov 2026, before any of its answers was generated:**
  `gpt-oss:20b` via **Ollama 0.40.16**, Ollama model ID `f38aa0c53da5` (weights blob
  sha256 prefix `8ecb5d4ad617`, 13 GB), run locally on the Bocconi HPC for Students
  (partition `stud`, QOS `stud`, one NVIDIA A100 MIG slice `4g.40gb`; test call on 15 Nov
  ran 100% on GPU). A model from a different company, run locally, no daily cap, so the
  same 2,000 (twin, product) calls can run in one Slurm job. Same prompts, same price
  orders (seed 20261112), same validation as the main run; temperature set explicitly to
  1 as in the main run. gpt-oss returns a separate "thinking" field: it is logged but only
  the "response" field is parsed. HPC access was granted on 12 Nov 2026, so the fallback
  (`gemini-3.1-flash-lite`) is not used.
- Robustness (price control): re-run the regression with the raw dollar price instead of the relative price; report whether the income × source interaction changes.

## 6. What would prove me wrong
- H1 is wrong if the gap has the opposite sign for half or more of the products, or if the correlation between real and synthetic gaps is ≤ 0.
- H2/H3 are wrong if synthetic gaps and the overall rate fall within the 95% confidence interval of the real ones.
- **H4 is wrong if, for half or more of the 10 products, the synthetic slope of
  Yes-share on relative price lies outside the 95% confidence interval of the real
  slope, or if its sign is opposite (an upward synthetic curve where the real one
  declines).**
- **H5 is wrong if the sign of the (highest − lowest income) slope gap disagrees for
  half or more of the products; as with H1, if the real slope gap for a product is
  inside its 95% CI, that product cannot test direction and is reported separately.**
- Caveat decided in advance: if the real income gap is close to zero for a product (inside its 95% CI), that product cannot test direction and is reported separately.
- **New caveat (12 Nov 2026):** the synthetic curve is elicited with all 11 prices in
  one call, so the model sees its own earlier answers; the synthetic curve may be
  nearly monotone by construction, while the real curve (one random price per person)
  is noisy. This is reported, not treated as a failure of H4; it is a reason to judge
  H4 on slopes, not point by point.

## 7. Analysis choices log
(The agent lists here every choice it makes; I approve each one.)
- 10 Nov 2026: products chosen with the fixed rule "every 4th item from QID9_1". While checking the data, the overall Yes/No counts (all respondents, not by income) of QID9_1, QID9_17 and QID9_21 were seen. The selection rule was fixed before this and was not changed afterwards. No result by income has been looked at.
- 11 Nov 2026: the new raw file `wave_4_labels_anonymized.csv` (original Qualtrics
  wave-4 export) was checked read-only with `uv run python`; findings:
  * `QID9_k` in `wave4_response_label.csv` = raw `k_Q295` = k-th displayed question
    (100% agreement, all 40 k, 2058 respondents); display order randomised per respondent
    (2058 distinct orders), per-respondent order reused in wave 3 (same-column agreement
    84.4%).
  * The catalog's price for `QID9_k` = respondent 1's price at display position k
    (40/40); for all other respondents the match is at chance (0.79% positional vs
    0.49% item-number). Catalog product names are respondent 1's products at those
    positions (supported: each catalog price lies in the price ladder of respondent 1's
    item at that position for 40/40, but in the ladder of item k for only 2/40).
  * Consequence: the former section 3 table (product per QID9_k) and the "OPEN
    PROBLEM – prices" are superseded; the corrected text is in section 3.
- 11 Nov 2026: analysis choices made as a result (no purchase rates by income were
  looked at before or while fixing them):
  (a) the 10 products are defined by **item number** (the items at display positions
      1, 5, 9, …, 37 of respondent 1's order: 25, 27, 31, 9, 36, 26, 15, 23, 38, 10),
      not by column; the fixed selection rule itself is unchanged.
  (b) for each respondent, a product's answer and price come from the position of that
      item in the respondent's `randDollarsString`; respondent 1's catalog price is not
      used for anyone else.
  (c) the 79 respondents with a malformed `randDollarsString` are excluded from all
      price-based analyses; the exclusion count and reason are printed by the script.
- 11 Nov 2026: unresolved detail recorded: which of items 36/26 is Angel Soft vs
  Chobani is not resolvable from prices (both are the free-ladder items at respondent
  1's positions 17 and 21); the assignment follows the catalog texts and could not be
  checked against an independent item→product list (the Q295 loop-merge product table
  is not in the raw export).
- 11 Nov 2026: the price control in section 5 is specified as **relative price**
  (price ÷ item's ladder maximum, 0–1) because ladder scales differ across products
  (e.g. $0–$5.36 for the oatmeal creme pies, item 10, vs $0–$59.98 for the ham,
  item 27), plus a **"free" indicator** (1 if the price shown was $0), because the
  jump in Yes at $0 is large (checked by the user with OpenCode: 81.6% Yes at $0 vs
  40.2% above $0, positional pairing, 1979 valid rows); product fixed effects absorb
  the ladder levels; a raw-dollar-price version is kept as a robustness check.
  Synthetic twins get their real twin's order and prices (section 4). Choice made
  before any result by income was computed.
- 12 Nov 2026: Gabor–Granger redesign of the synthetic step (chosen before any
  synthetic answer was generated; no result by income has been looked at):
  (a) every twin answers each of the 10 products at **all 11 ladder prices** (full
      lattice, not iterative Gabor–Granger and not only the real twin's price);
      comparison 1 (H1–H3) is unchanged in definition and now reads the answer at the
      real twin's price out of the 11.
  (b) one model call per (twin, product) = **2,000 calls** (+2,000 for the robustness
      re-run); since each product is a separate call, the real twin's product order is
      no longer used (supersedes the order-and-price-per-twin rule of 11 Nov).
      Rejected alternatives: one call per twin with all 110 questions (attention
      degradation; one failure loses a whole twin) and one call per price (22,000
      calls, infeasible on the free tier).
  (c) price order within a call **randomised per (twin, product), fixed seed 20261112**
      (arbitrary but fixed); the real twin's price is not marked, and randomisation
      keeps comparison 1 free of a systematic position effect.
  (d) wording and options per price question exactly as the real survey; intro QID8
      once per call; nonzero prices with two decimals ("$6.00"); **$0 exactly as in
      `randDollarsString`: "$0 (product is free)"** (verified read-only in the raw
      file); no other "free" label added.
  (e) JSON output validated against the ladder (all 11 prices present, Yes/No only);
      malformed calls retried once, then excluded with the count printed.
  (f) new tests **H4** (real vs synthetic slope of Yes-share on relative price, per
      product) and **H5** (income gradient in that slope), with falsification criteria
      in section 6; real demand curves built from the 1979 valid respondents, each of
      whom saw one random price per product.
  (g) free-tier limits checked on 12 Nov 2026 (500 requests/day, 15/minute per model);
      the main run is therefore split over 4 days with the same seed, after a
      30-call pilot (see section 4).
  (h) known limitation accepted: synthetic answers are elicited in a multi-price
      context (the model sees its earlier answers), so the H1–H3 numbers need not
      equal what a single-price-per-product design would have produced.
- 12 Nov 2026: implementation choices for `src/02_generate.py` (all fixed before any
  synthetic answer was generated; approved by the user in chat):
  (a) the 11 exact price texts per item are hardcoded in the script, copied
      read-only from the distinct prices in `randDollarsString` of the 1979 valid
      respondents (verified: exactly 11 distinct texts per item; $0 exactly as
      "$0 (product is free)"); the script asserts every price in
      `sample_200_items.csv` lies on its item's ladder.
  (b) product and category text in the prompts is taken verbatim from
      `question_catalog.json` (`QID9_1, 5, 9, 13, 17, 21, 25, 29, 33, 37`,
      respondent 1's positions for our items), not from the abbreviated `product`
      column of `sample_200_items.csv` (9 of 10 differ); asserts: item→QuestionID
      mapping, catalog category == CSV category for every row (10/10), catalog
      price on the item's ladder.
  (c) the QID8 intro is used verbatim except "a series of 40 questions" is
      replaced by "a series of 11 questions", so the intro describes the number
      of questions the prompt actually contains; the persona block (region, sex,
      age, education, income — CSV values only, no question texts) and a short
      JSON instruction paragraph at the end are additions the real survey did
      not have.
  (d) price order randomised per (twin, product) with one `Random(20261112)`
      stream consumed in enumeration order (twins **sorted by new_id, C001,
      C002, … — 12 Nov 2026 change from the original "file order": the CSV is
      grouped by income bracket, so file-order iteration would have made the
      pilot 3 twins of one bracket and let each day cover different brackets,
      confounding day/model drift with income**), items in file order, so the
      permutation table is identical on every restart and between pilot and
      main for shared keys.
  (e) limits enforced in code: >= 5 s between request starts (<= 12/min, under
      the 15/min free-tier limit); daily cap of 500 requests counted by
      Pacific-time date from the JSONL (pilot and main share one counter;
      the free tier resets at midnight Pacific = 09:00 Italian time).
  (f) error policy: quota/429 (RESOURCE_EXHAUSTED) stops the run with nothing
      retried or excluded (the (twin, item) stays not-done; the sent request
      still counts toward the daily cap); other non-retryable API errors (4xx
      other than 429, e.g. 400/401/403/404) stop the run the same way;
      network errors/5xx leave the (twin, item) not-done and retried on the
      next restart, consuming no retry, but after 3 consecutive API errors
      the run stops so a persistent error cannot burn the daily quota; only
      invalid model outputs follow the one-retry-then-exclude rule, and
      excluded calls are skipped on restart, counted and printed. A --dry-run
      flag runs all checks and prints the first prompt and the work order
      without calling the API or writing files.
  (g) outputs: append-only JSONL (one line per request: prompt, raw output,
      model, model version, timestamp Pacific, price order, attempt, outcome)
      written and flushed per answer; a tidy CSV (one row per twin x item x
      ladder price) rebuilt from valid JSONL lines at run start and end; one
      run log per Pacific day (`run_YYYYMMDD_pacific.log`); pilot writes to
      separate `pilot_answers.*` files that the analysis never reads, and its
      30 calls share the day-1 quota (day 1 = 30 pilot + 470 main).
  (h) schedule consequence (section 4): 2,030 requests against the 500/day cap
      spread the run over 5 Pacific days (Thu 12 – Mon 16 Nov), not 4.
  (i) model-version guard fixed after the pilot (12 Nov 2026, before the main
      run): the pilot logged a false "version mismatch" warning on every
      response because the response-level `model_version` field is the model
      name ("gemini-3.5-flash-lite"), not the dated version. Now the script
      calls `client.models.get(model=...)` once at the start of every run
      (not in `--dry-run`), logs the returned `.version`, and stops before
      any call if it differs from `3.5-flash-lite-07-2026` (nothing sent,
      nothing excluded); the per-response `model_version` is only logged,
      never compared. The pilot itself ran clean under the error policy:
      25 valid answers, 5 temporary 503s handled as not-done retries, no
      exclusions.
  (j) validation change after day 1 of the main run (Thu 12 Nov 2026), based
      ONLY on output format — no answers were looked at, no result by income
      has been computed: all 41 invalid outputs of day 1 failed for one
      reason — in the "price" field the model wrote "$0" instead of the
      ladder text "$0 (product is free)". Since each item's ladder has
      exactly one zero price, this is a formatting difference, not a
      different answer. (i) The validator now accepts "$0", "$0.00" and
      "$0 (product is free)" as the same zero price and stores the answer
      under the exact ladder text; the prompt was NOT changed, so answers
      from days 1–5 are elicited identically. (ii) At startup the script
      re-validates the raw outputs already saved in the JSONL with the new
      rule, recovering the excluded keys and invalid first attempts without
      new API calls, appending a "revalidated" record per recovered key and
      printing the count (recovery is idempotent across restarts). Day-1
      facts recorded: 352 valid, 20 excluded, 66 temporary 503s; the run
      stopped with a 429 after 492 counted requests — failed requests count
      toward the daily quota too (the cap counts requests sent, which is
      correct).
- 12 Nov 2026: model fixed before any synthetic answer: `gemini-3.5-flash-lite`, version `3.5-flash-lite-07-2026`, temperature 1 (the endpoint's default, set explicitly). Chosen as the most recent stable Flash-Lite model available to the key; aliases (`-latest`) and preview/image/tts variants excluded. Use of Google AI Studio confirmed by the professor by email (12 Nov 2026). API key stored in `.env` (permissions 600, listed in `.gitignore`).
- 12 Nov 2026: plan A adopted after checking the free-tier limits (500 calls/day): keep
  one call per (twin, product) and split the main run over 4 days, rather than putting
  several products in one call (longer prompts, attention risk) or cutting the sample
  to 100 (only 20 per income bracket). Robustness re-run: GPT-OSS 20b on the Bocconi
  HPC if access arrives in time (suggested by the professor; requested by email on
  12 Nov), otherwise `gemini-3.1-flash-lite`. Decided before any synthetic answer.
- 12 Nov 2026: implementation choices for `src/03_real_curves.py` (real demand
  curves; all approved by the user in chat before the script was written; the
  script ran clean on the first pass, all checks green):
  (a) the long respondent×item table is built for ALL 1979 valid respondents
      (not only the 200 sampled) and saved to
      `data/derived/real_items_long.csv` (19,790 rows), so the later H4/H5
      slope regressions and the by-respondent bootstrap do not recompute it;
      person identifier = `pid` (local file in data/derived, never sent to a
      model; section 3 item 2 requires anonymisation only before model calls).
  (b) parse/validity/self-check logic duplicated from `src/01_prepare.py`
      (not imported, so the stratified sampling is not re-run): malformed
      `randDollarsString` exclusion (79 rows, reason printed), ladder maxima,
      respondent-1 catalog prices, answer texts.
  (c) price point = the observed price in dollars (each item has exactly 11
      distinct values among the valid rows, asserted); `price_text` (exact
      text, "$0 (product is free)" for zero) kept as the future join key with
      the synthetic ladders; `ladder_index = round(price ÷ (M/10))` (0–10;
      asserted integral within 0.02 of a step, 11 distinct, monotone in
      price) added for ordering and later real-vs-synthetic joins.
  (d) `results/real_demand_curves.csv` columns: item, category, product,
      QID21, price_dollars, price_text, relative_price, free, ladder_index,
      n_respondents, n_yes, n_no, share_yes; rows sorted by item number,
      then natural income order, then price; the bracket column keeps the
      name QID21 for consistency with `sample_200_items.csv`.
  (e) stdout prints counts only (row counts, exclusion reason, observed
      per-item ladders, cell counts, min/median/max respondents per cell);
      no Yes shares — the shares live in the results CSV.
  (f) hard sanity anchor (counts fixed by the user): positional pairing over
      ALL 40 items × 1979 valid respondents must give exactly 5866 Yes of
      7193 observations at $0 and 28935 Yes of 71967 above $0 (7193 + 71967
      = 79,160 = 1979 × 40); the script stops if they differ. Reproduces the
      11 Nov 81.6%/40.2% check as exact counts. Result: matched exactly.
  (g) no randomness (no sampling, no model calls) → no seed; reads only
      data/raw (read-only) and writes `data/derived/real_items_long.csv`
      and `results/real_demand_curves.csv`; synthetic outputs and `.env`
      untouched. Run outcome: 550 of 550 cells non-empty; respondents per
      cell min 15, median 35, max 66 (section 5 expected ≈36).
- 12 Nov 2026: implementation choices for `src/04_real_slopes.py` (real
  demand-curve figures and price-sensitivity slopes; all approved by the
  user in chat before the script was written):
  (a) B = 2,000 bootstrap replications, seed 20261112 (numpy `default_rng`;
      the same number as the 02_generate price-order seed but a different
      RNG stream), percentile CIs (2.5/97.5); B and seed fixed before any
      slope was computed.
  (b) one pid resample per replication drives all 44 cells, so the
      high-minus-low slope difference is paired (both slopes from the same
      resample); the script flags the products whose difference CI covers 0
      (section 6 caveat: those cannot test the direction of H5).
  (c) "overall" = all 5 brackets pooled (the H4 real-side object); lowest =
      "Less than $30,000" (n=347), highest = "$100,000 or more" (n=532);
      the lowest/highest fits are the same LPM on the bracket subsample.
  (d) LPM fitted by OLS with numpy least squares (statsmodels not needed;
      identical point estimates): yes ~ relative_price + free (+ intercept);
      the pooled model adds 9 item fixed effects (reference item 9, the
      lowest item number); slope = change in P(Yes) over the full 0->1
      relative-price range.
  (e) figures: PNG, 200 dpi; per-item x-axis in dollars at the item's 11
      ladder prices; pooled x-axis at the clean ladder positions k/10 (the
      continuous relative_price wobbles ≤0.004 around k/10 because the
      ladder texts are rounded to cents — the regression uses the exact
      continuous variable); the pooled figure carries a pointwise 95%
      bootstrap band from the same replicates.
  (f) `results/real_slopes.csv` also stores the free-indicator coefficient
      with its CI (transparency); difference rows have empty n_respondents,
      n_obs and free columns (a difference uses both subsamples, no single
      n, and has no free coefficient).
  (g) stdout: row-count prints plus the full 44-row estimates table (a
      deliberate deviation from the counts-only stdout of 01/03: the
      estimates are this step's deliverable); `real_items_long.csv` is
      re-validated on load (row counts, pid sorting, 10 rows per pid, item
      grid, answers, brackets, relative_price/free/ladder consistency)
      instead of re-reading the raw files.
  (h) run outcome (14 s): all 10 items' slopes are negative and their CIs
      exclude 0 (e.g. item 25 −0.96 [−1.02, −0.90], item 27 −0.21
      [−0.25, −0.16]); the free coefficient is large only for item 27
      (+0.58 [+0.50, +0.65]) and item 15 (+0.20 [+0.12, +0.28]); pooled
      slope −0.69 [−0.71, −0.66]. For ALL 10 items the high-minus-low
      difference CI covers 0 (e.g. item 23 −0.19 [−0.41, +0.03] is the
      largest-magnitude difference) — per section 6 no real product can
      test the H5 direction at this sample size; the pooled difference is
      +0.01 [−0.06, +0.09]. H5 will therefore rest on the comparison of
      real vs synthetic slope gaps, with the real gaps reported as
      imprecise.
- 15 Nov 2026: robustness model fixed before any of its answers: `gpt-oss:20b`, Ollama
  0.40.16, model ID `f38aa0c53da5`, on the Bocconi HPC (A100 MIG 4g.40gb, partition and
  QOS `stud`). Setup checked on the cluster: GPU visible to Ollama (library=CUDA, 39.2 GiB),
  model loaded 100% on GPU, a test call answered correctly. Planned: temperature 1 (as in
  the main run); "thinking" output logged, not parsed.
- 15 Nov 2026: implementation choices for the robustness run (all approved by the user in
  chat before any robustness answer was generated; no result by income has been looked at):
  (a) prompt export (`src/05_export_prompts.py`, Mac): the exact stored prompt and
      price_order of each of the 2000 valid main-run calls are copied verbatim to
      `data/derived/prompts_for_robustness.jsonl` (fields: call_index, new_id, item,
      price_order, prompt; the prompt is byte-identical to the main-run one, not rebuilt).
      Asserts: exactly 2000 valid keys, at most one valid line per key, key set exactly
      equal to `sample_200_items.csv` (200 twins x 10 items), every price_order a
      permutation of its item's 11 ladder texts; output order deterministic (sorted by
      new_id, item). No pid anywhere (`id_map.csv` never leaves the Mac); no demographic
      data beyond the prompt text.
  (b) per-call seed = 20261112 + call_index (call_index = 0-based position in the prompts
      file, recorded in every robustness record), replacing an earlier idea of one shared
      seed: every call is reproducible and seeds differ across calls.
  (c) generation (`hpc/06_generate_ollama.py`, cluster, Python standard library only):
      POST `http://127.0.0.1:11434/api/generate`, model `gpt-oss:20b`, stream false,
      keep_alive 12h, options temperature 1, seed per (b), num_ctx 8192 (Ollama's smaller
      default could silently truncate thinking + answer; `done_reason` "length" counts as
      invalid output). Validation identical to `src/02_generate.py`, including the
      "$0"/"$0.00" tolerance (stored under "$0 (product is free)"); one retry for invalid
      output, then exclusion with the count printed. Every request appended + flushed
      immediately to `robustness_answers.jsonl` with prompt, response, thinking (logged,
      never parsed), model, ollama_version, format_kind, seed, done_reason,
      eval_count/eval_duration, outcome, parsed. Resume: valid/excluded keys skipped;
      server errors (timeout/5xx) leave the key not-done and stop after 3 consecutive; no
      rate limit, no daily cap (local server). The script re-checks the Ollama version
      (0.40.16) before any call.
  (d) JSON schema in Ollama's "format" parameter: a root-level array first (matches the
      prompt's "a list of 11 objects" instruction, like the main run's response_schema);
      if the root array fails (HTTP 4xx, or both attempts return invalid output), the
      script falls back permanently to the `{"answers": [...]}` wrapper, unwraps it, and
      validates identically (the fallback never causes an exclusion by itself: it costs
      one extra attempt on the same key); the schema used is recorded per line
      (format_kind) and reused on restart (user decision; the smoke test decides).
  (e) `hpc/run_ollama.sbatch`: partition `stud`, account 3341868, qos `stud`,
      gres `gpu:4g.40gb:1`, 4 CPUs, 12 h; starts `ollama serve` from `~/ollama/bin`
      (OLLAMA_KEEP_ALIVE=12h); a stdlib-python3 block waits for the server, asserts the
      Ollama version is 0.40.16 (mirrors the main-run models.get guard) and asserts the
      `gpt-oss:20b` digest from `/api/tags` starts with `f38aa0c53da5` (the digest is not
      in `/api/show`'s documented response), stopping before any call otherwise; then
      runs the script and stops the server. Smoke test: `sbatch --export=ALL,SMOKE=3`
      runs the first 3 not-done calls into the same `robustness_answers.jsonl` (valid
      records count; the full run skips them); no separate pilot file.
  (f) walltime guard: after 11 h the script stops starting new calls and says to resubmit;
      calls are sequential (one at a time). 2000 calls fit in 12 h only if they average
      <= ~21.6 s; if the smoke test's rate says otherwise, resubmit to resume across jobs
      or discuss parallelism before the full run.
  (g) housekeeping: timestamps in cluster-local time with UTC offset (the main run used
      Pacific time only for the free-tier quota); answers copied back to
      `data/derived/robustness_answers.jsonl` by scp; no .env/API key involved anywhere
      (Ollama is local).
- 16 Nov 2026: implementation choices for `src/07_compare_twins.py` (comparison 1,
  H1–H3; all approved by the user in chat before the script was written; the main
  synthetic run had completed with 22,000 rows = 200 twins × 10 items × 11 prices,
  `gemini-3.5-flash-lite`, all 2,000 calls valid):
  (a) real side = the 200 sampled twins only (their wave-4 answers in
      `sample_200_items.csv`); the 1,979 valid respondents enter comparison 1 only
      through the exploratory side analysis in (h).
  (b) synthetic answer read by joining `sample_200_items.csv` to
      `synthetic_answers.csv` on (new_id, item, exact price_text); asserted
      2,000/2,000 matched, exactly 11 ladder rows per (twin, item), one model
      (`gemini-3.5-flash-lite`), only the two survey answer texts, and
      relative_price/free identical within each twin pair.
  (c) highest/lowest income brackets = "$100,000 or more" / "Less than $30,000"
      (same as 04); gap = highest minus lowest Yes share, per item.
  (d) bootstrap: B = 2,000, seed 20261112 (numpy `default_rng`, one stream consumed
      first by the 200-twin bootstrap, then by the exploratory 1,979 bootstrap),
      percentile 2.5/97.5 CIs, twins resampled with replacement **stratified within
      each income bracket** (40 per bracket, preserves the stratified design); one
      twin-index resample per replication drives both the real and the synthetic
      quantities (paired bootstrap, the twins are matched with identical prices).
  (e) H1: sign agreement of real vs synthetic gap per item and Pearson correlation
      across items; per the section 6 caveat, an item is "testable" only if its
      real gap's 95% bootstrap CI excludes 0; testable items are excluded from the
      sign-share denominator; the correlation is reported on all 10 items and on
      testable items only and the decision uses the testable-only correlation
      (with < 2 testable items it is undefined → decision on sign-share alone,
      noted in the output). H1 rejected if opposite sign for half or more of the
      testable products or correlation ≤ 0.
  (f) H2: per item synthetic gap / real gap (ratio undefined when the real gap is
      0; the synthetic-minus-real difference is reported as well); H2 rejected/
      wrong if the synthetic gap lies inside the real gap's 95% CI for at least
      half of the 10 items (all 10 items, no testability exclusion — the real CI
      is itself the yardstick for size).
  (g) H3: overall = all 200 twins × 10 items pooled, at the real twins' prices;
      rejected/wrong if the synthetic rate falls inside the real rate's 95%
      bootstrap CI. In the output "rejected" always means the section 6 condition
      holds (H1: direction does not match; H2/H3: the theory's "differs" claim is
      falsified).
  (h) exploratory secondary analysis, clearly labelled in the output, NOT used in
      any verdict: real highest-minus-lowest gap per item on all 1,979 valid
      respondents from `data/derived/real_items_long.csv`, with a respondent-level
      bootstrap CI (resampling respondents within each of the two extreme
      brackets), shown next to the 200-twin real gap.
  (i) regression: LPM `yes ~ C(QID21, ref = lowest bracket) * source +
      C(item, ref = 9) + relative_price + free`, OLS (statsmodels), cluster-robust
      SEs clustered by new_id (200 clusters; the real and synthetic rows of one
      twin share a cluster, per the user's "clustered by twin"); the 400-cluster
      variant (source × new_id) printed as a one-line note on the interaction SEs.
  (j) section 5 price robustness run in the same script: relative_price replaced by
      the raw dollar price; the income × source interaction coefficients of the two
      models compared in the output.
  (k) outputs: `results/compare1_shares.csv`, `results/compare1_gaps.csv`,
      `results/compare1_regression.csv` (both models, `model` column),
      `results/compare1_hypotheses.csv`; no figures (comparison 2's job).
  (l) run outcome (16 Nov 2026, clean second run after the wrong-column bug recorded
      in mistakes.md, session 8): only 1 of 10 items has a testable real gap (item 10,
      −0.30 [−0.50, −0.10]; its synthetic gap −0.075 has the same sign); the all-10
      real-vs-synthetic gap correlation is 0.28. H1 not rejected (decided on the one
      testable product; the section 6 caveat dominates at n = 40 per bracket).
      H2 rejected: the synthetic gap lies inside the real gap's 95% CI for 7 of 10
      items — the sizes do not differ detectably, falsifying the theory's
      "gaps differ in size" expectation. H3 not rejected: synthetic overall rate
      0.5155 lies outside the real 95% CI [0.4490, 0.5015]. Regression: all four
      income × source interactions small and insignificant (largest +0.070,
      SE 0.047, p = 0.14, highest vs lowest bracket); the raw-dollar-price
      robustness gives identical interaction coefficients. Exploratory (not in any
      verdict): on all 1,979 respondents the real gaps are mostly small; items 26
      (+0.167 [0.100, 0.232]), 27 (+0.047 [0.005, 0.090]) and 38 (+0.077
      [0.010, 0.142]) have CIs excluding 0, and several 1979 gap signs differ from
      the 200-twin point gaps (e.g. item 10: −0.060 vs −0.300), consistent with the
      noise of 40-twin-per-bracket cells.
- 16 Nov 2026: implementation choices for `src/08_compare_curves.py` (comparison 2,
  H4–H5; all approved by the user in chat before the script was written):
  (a) real side read from `results/real_slopes.csv` and `results/real_demand_curves.csv`
      (the deliverables of 03/04), validated on load, not recomputed — the H4/H5 real
      CIs are 04's respondent-bootstrap CIs (B = 2,000, seed 20261112).
  (b) synthetic demographics (QID21) joined from `sample_200_items.csv` on `new_id`
      (the synthetic answers file stores none); asserted 200/200 twins, 40 per bracket.
  (c) synthetic estimation mirrors 04 exactly: same LPM (yes ~ relative_price + free;
      pooled adds 9 item FE, ref item 9), same samples (all / lowest / highest /
      high-minus-low), B = 2,000, seed 20261112 (fresh `default_rng` stream), one twin
      resample per replication driving all cells so the high-minus-low difference is
      paired, percentile CIs, numpy least squares.
  (d) clarification of section 5 recorded above: the pooled joint H4 model includes the
      `source` main effect (user decision, 16 Nov).
  (e) clustering in the joint real+synthetic models: composite key (source, id) →
      2,179 clusters, per section 5's parenthetical "real: respondent; synthetic: twin".
  (f) per-item joint regressions (yes ~ source × relative_price + source × free, one
      item at a time, clustered) added as descriptive per-item synthetic-minus-real
      slope differences; the section 6 H4 verdict still uses the literal rule
      (synthetic point slope inside/outside the real slope's 95% CI).
  (g) H4 "sign opposite" = synthetic slope > 0 (all 10 real slopes are negative with
      CIs excluding 0); H4 rejected if ≥ 5 of the 10 products fail (outside CI or
      sign opposite).
  (h) H5: testable = the real high-minus-low slope-difference CI excludes 0 (read from
      real_slopes.csv); per 04's outcome all 10 cover 0, so H5 prints "cannot test
      (0 testable products)"; the full real-vs-synthetic gap table is still reported,
      real gaps flagged imprecise, and sign agreement is reported.
  (i) curve tables: synthetic shares per item × bracket × price for all 5 brackets
      (550 cells, exactly 40 twins per cell), mirroring real_demand_curves.csv;
      cell counts printed (rule 2).
  (j) figures are point-estimate overlays only, no bootstrap bands (user decision,
      16 Nov): uncertainty lives in the slope CIs, per section 5's "judge H4 on
      slopes, not point by point"; 10 per-item overlays (dollars), one pooled overlay
      (relative price), one pooled overlay for the highest vs lowest brackets.
  (k) robustness: the pooled joint H4 model re-run with the raw dollar price instead
      of the relative price; the source × price coefficients compared in the output.
  (l) exploratory block, clearly labelled, NOT used in any verdict: for the 2,000
      synthetic (twin, item) ladders ordered by price — share with exactly one
      Yes→No switch (monotone single threshold), shares all-Yes / all-No, remainder
      (non-monotone); synthetic Yes share at $0 next to the real one recomputed from
      `real_items_long.csv` (the 12 Nov anchor count 5866/7193).
  (m) verdict wording as in 07: "rejected (section 6 condition met)" / "not rejected"
      / "cannot test"; "rejected" always means the section 6 falsification holds.
  (n) outputs: `results/compare2_curves_synthetic.csv`, `compare2_slopes_synthetic.csv`
      (44 rows, same columns as real_slopes.csv with n_twins instead of
      n_respondents), `compare2_h4_h5.csv`, `compare2_regression_pooled.csv` (both
      price controls, `model` column), `compare2_hypotheses.csv`, and the figures of
      (j) as `compare2_curve_item<item>.png`, `compare2_curve_pooled.png`,
      `compare2_curve_brackets.png`.
  (o) run outcome (16 Nov 2026, clean run; an intermediate run had a misaligned
      pooled-design dummy order — caught immediately because its pooled synthetic
      slope, −0.724, contradicted the per-item slopes (≈ −1.45) and the joint
      regression (implied −1.398); fixed and all output files overwritten by the
      fixed run, whose numbers cross-check exactly: −0.6854 − 0.7122 = −1.398):
      H4 rejected — all 10 synthetic slopes lie outside the real slope's 95% CI
      (synthetic ≈ −0.90 to −1.49, real ≈ −0.21 to −0.96; none sign-opposite; pooled
      joint difference −0.712, SE 0.012). H5 cannot test (0 testable products, all 10
      real gap CIs cover 0); sign agreement 7/10 over all 10 items; the synthetic gap
      CI excludes 0 for 8 of 10 items while the real gap CI excludes 0 for none.
      Pooled synthetic slope −1.398 [−1.403, −1.392] vs real −0.686 [−0.710, −0.662].
      Exploratory (not in any verdict): all 2,000 synthetic ladders are exactly one
      Yes→No switch (0 all-Yes, 0 all-No, 0 non-monotone); synthetic Yes at $0 =
      1.0000 (2,000/2,000) vs real 0.8632 (1,495/1,732) on our 10 items (the 81.6%
      anchor was over all 40 items).
- 16 Nov 2026 (robustness run, job 767786, observed while running): at 12:58 the pair (C007, item 38) returned an empty answer list twice under the root-array schema ("missing ladder prices": all 11 missing). Under the approved rule this triggered a permanent switch to the {"answers": [...]} wrapper schema for that key and all later calls; the third attempt (wrapper) failed in the same way and the pair was excluded. So about the first 70 pairs (twins C001-C007) were collected with the root-array schema and all later ones with the wrapper; the prompt is identical for all calls, and twin order is random (not related to income). Interpretation: the failure was a model output failure on one prompt, not a schema incompatibility, so the switch was not needed; decided not to stop or restart the job (no answer had been looked at). To report: format_kind per line is recorded; the robustness analysis will check that results do not differ by format_kind, and C007/38 is excluded from the robustness comparison only.
- 17 Nov 2026 (robustness run, decided after the job ended and before any answer was analysed; based only on output format): the job finished (COMPLETED, 8h56m). The file has 2,142 lines = 2,000 first attempts + 142 retries; 1,984 keys valid, 16 excluded. Inspection of the invalid outputs showed that almost all of them contain 11 answers but write the prices without the "$" sign ("19.16" instead of "$19.16"; 139 of 158 invalid outputs), or write the zero price as "0". Correction to the 16 Nov entry above: the two failures of (C007, item 38) were of this kind too (prices without "$"), not an empty answer list. Rule: the saved gpt-oss outputs are re-validated offline, with no new model calls. Accepted in addition to the existing zero-price aliases ("$0", "$0.00", "$0 (product is free)"): (i) a price written without "$" that equals a ladder price of that item (e.g. "19.16" = "$19.16"); (ii) "0" as the zero price. Nothing else is changed: outputs with a dropped decimal point (e.g. "$223" for "$2.23"), duplicate prices or fewer than 11 answers stay invalid; a key with no attempt valid under the rule stays excluded; if a key has more than one attempt valid under the rule, the first one in attempt order is used. The re-validated file is written as a new file (the raw file is not modified), with a column marking the recovered keys; the robustness analysis reports the recovered keys separately and checks that results do not change when they are dropped. Same treatment as the Gemini run (zero-price aliases, mistakes.md entry 25).
- 17 Nov 2026 (clarification of the rule above, decided before writing the script and before any analysis; ambiguity found by the agent in a read-only dry run): "first attempt valid under the rule" would have changed 111 keys that were already valid (attempt 1 invalid only for the missing "$", attempt 2 valid), replacing their attempt-2 answers with different attempt-1 answers. Decision: the new rule is applied only to keys with no valid line in the original file (the 16 excluded keys); the 1,984 originally valid keys keep their original attempt and answers unchanged. Dry-run result (format only): all 16 excluded keys are recoverable, 0 remain excluded.
- 17 Nov 2026: robustness analysis (section 5, second model) specified before any gpt-oss answer was analysed (only the format of the outputs has been inspected; user approved in chat):
  (a) input: `data/derived/robustness_answers_clean.csv` (2,000 keys, 22,000 rows). It is first rebuilt as `data/derived/robustness_answers_full.csv` with exactly the columns of `synthetic_answers.csv` (category, product, price_dollars, relative_price, free, position_in_call taken by matching new_id, item, price_text; model = gpt-oss:20b), plus `recovered` and `format_kind`. Check: the set of (new_id, item, price_text) is identical to synthetic_answers.csv.
  (b) same hypotheses, other model: src/07 and src/08 get two optional arguments (input synthetic file, output prefix); with no arguments they must reproduce the existing results files byte-identically (checked by checksum before and after). Run with the gpt-oss file and prefix `robust_`: H1-H5 against the real data with the same section 6 rules, same B = 2,000, same seed 20261112.
  (c) headline per model (Gemini, gpt-oss) next to the real data: Yes share at $0 (real 0.8632), share of single-threshold ladders (exactly one Yes->No switch), pooled synthetic slope (real -0.686), overall Yes rate (real 0.4750).
  (d) agreement between the two models: share of identical answers over the 22,000 (twin, item, price) cells, and share of (twin, item) ladders with the same threshold.
  (e) checks, not changing any verdict: (i) (c) for gpt-oss without the 16 recovered keys; (ii) (c) for gpt-oss separately for format_kind root_array and wrapper.
  (f) reading rule, fixed now: if gpt-oss gives the same H3 and H4 verdicts as Gemini and at least 90% of its ladders are single-threshold, the conclusion "synthetic twins answer in a step pattern, more price sensitive than real people" is reported as robust to the model; otherwise it is reported as model-dependent (a limitation). H1, H2, H5 verdicts are reported side by side without a robustness claim (low power, see mistakes.md 35).
  (g) outputs: `results/robust_compare1_*.csv`, `results/robust_compare2_*.csv` (+ figures with the same prefix), `results/robustness_summary.csv` for (c)-(e); script `src/10_robustness_summary.py`.
  (h) run outcome (17 Nov 2026): 07 and 08 got --syn-csv, --out-prefix and --model (default = Gemini; no-argument re-runs reproduced all 21 earlier results files byte-identically, md5 checked twice). gpt-oss verdicts: H1 rejected (1 testable product, as with Gemini, so no claim), H2 rejected (6/10 inside real CI), H3 not rejected (0.5715 vs real CI [0.4490, 0.5015]; Gemini 0.5155), H4 rejected (10/10; pooled synthetic slope -1.250 vs Gemini -1.398 vs real -0.686), H5 cannot test. Headline: Yes at $0 0.9995 (1,999/2,000; Gemini 1.0000; real 0.8632); single-threshold ladders 0.9055 (1,811/2,000; Gemini 1.0000). Reading rule (f): all three conditions met, so the step-pattern / over-price-sensitivity conclusion is reported as robust to the model, with two caveats stated in the report: condition 3 is met narrowly (0.9055 vs 0.90; of the 189 other gpt-oss ladders 156 have 3 switches, i.e. non-monotone, 19 all-Yes, 14 other), and agreement is at the aggregate level only: identical answers 87.1% of 22,000 cells, but among ladders single-threshold in both models the threshold is the same for only 27.7% (502/1,811); gpt-oss puts the threshold on average 0.70 ladder steps higher. Checks (e): without the 16 recovered keys and by format_kind the headline numbers barely change (single-threshold 0.905 / 0.914 / 0.905). Key numbers re-checked independently from the data files (with the help of Claude): all identical.
- 17 Nov 2026: independent re-run of the real-data analysis (brief, session of 16 Nov: "the same question re-run in a fresh session or with the other free model, and the differences explained"; fixes the non-independence of the 12 Nov agent run, mistakes.md 29). Set-up: a separate folder `blind_run/` with only data/raw (copy), data/README.md, AGENTS.md, opencode.json, pyproject.toml/uv.lock and plan.md sections 1-6 (section 7, this log, removed); no src/, results/, data/derived/ or mistakes.md content. Model: DeepSeek V4.1 Flash Free (the other free OpenCode model), fresh session, agent mode (works on its own, asks only if blocked). Task: the real-data part of section 5 (demand curves by item and income bracket, slopes with 95% CIs, high-minus-low slope difference), and a list of every analysis choice it made without asking. Comparison afterwards: its choices and numbers vs results/real_slopes.csv and results/real_demand_curves.csv; differences explained, not corrected. Known leak: plan.md section 3 still contains the session-1 anchor (81.6% vs 45.5% Yes at $0 over all 40 items); it is a data check, not an analysis choice.
  Status (17 Nov 2026, 15:35): the folder was moved out of the repository to /Users/benedettagatto/blind_run (own git repo, one setup commit) because OpenCode treated it as part of my-analysis. The desktop app still does not open it as a separate project, and the command-line OpenCode conflicts with the app's session database (not repaired, to keep the existing session history). Re-run not started; no answer or result produced. Question sent to the professor: is the re-run required, or is the documented limitation enough?
  Update (17 Nov 2026, evening; after the professor's reply: the re-run is required). (i) Model changed to GLM-5.3-Flash Free, the same model as the guided run, so that only the access to earlier choices differs (professor: "use the same model if you can"). (ii) Leaks removed from the clean folder before any run (commit fda8b4f in /Users/benedettagatto/blind_run): plan.md now holds only the question, theory, hypotheses, data notes, the product list and a task section that leaves price coding, model, CI method, replications and seeds to the analyst; removed: all seeds, the regression specification (relative price, free indicator, fixed effects, clustering), the bootstrap, the synthetic-step and robustness sections, and the 81.6%/45.5% anchor (also removed from data/README.md). AGENTS.md keeps the generic starter rule "fix random seeds" (no value). (iii) Comparison afterwards: choices and conclusions, not digits; different choices are a result.
  Outcome of the independent re-run (17 Nov 2026, GLM-5.3-Flash Free, fresh session in /Users/benedettagatto/blind_run, no access to this repository; the agent created a root README.md because pyproject.toml needs one, a set-up omission of mine). Its own choices (results/choices.md there, 21 items): dollar price as primary coding (share-of-max as sensitivity); no "free" indicator; per-product OLS LPM; HC1-robust Wald CIs (no clustering, one answer per respondent per product); income gap from a price x bracket interaction model; a 2,000-replication pairs bootstrap with its own seed 20261117 only as a check; no attention-check exclusions; no weighting. Guided run: relative price + free indicator, bootstrap by respondent (B = 2,000, seed 20261112). Comparison (checked with the help of Claude): (1) demand curves identical in all 550 item x bracket x price cells (same n, same share): data preparation reproduced independently. (2) Slopes on the 0-1 price scale agree in sign for 10/10 and within about 0.05 for 9/10; the exception is the ham (item 27: -0.45 vs -0.21) because without the free indicator the large jump at $0 (76% -> 24%) is read as slope. (3) Income slope gaps agree in sign for 9/10 (item 26 flips, both CIs cover 0); for item 23 (bacon) the blind run's CI excludes 0 ([-0.401, -0.024] vs guided [-0.411, +0.027]), so under its choices H5 would have 1 testable product instead of 0: Gemini's synthetic gap agrees in sign there (-0.026), gpt-oss's does not (+0.109). Conclusion: the substantive conclusions hold (all real slopes negative, real income gaps essentially zero); one borderline product changes testability with the CI method and price coding, which shows how analysis choices can move a verdict. Verdicts in sections 5-6 stay as pre-registered (guided choices); this is reported in the Reflection, not used to change them.
