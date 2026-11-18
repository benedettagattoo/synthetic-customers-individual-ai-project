"""10a_build_robustness_file.py -- rebuild the robustness answers in the Gemini file's schema.

plan.md section 7, entry "17 Nov 2026: robustness analysis", point (a). This is a
pure format/label step: no answer is read, aggregated or compared (no Yes share is
computed or printed).

Reads (read-only):
  data/derived/robustness_answers_clean.csv  22,000 rows: the offline re-validated
                                             gpt-oss answers (src/09_revalidate_robustness.py):
                                             new_id, item, price, answer, yes,
                                             attempt_used, format_kind, recovered
  data/derived/synthetic_answers.csv         22,000 rows: the Gemini run in the target
                                             schema -- source of the product metadata
                                             (category, product, price_dollars,
                                             relative_price, free, position_in_call)

Writes:
  data/derived/robustness_answers_full.csv   one row per (twin, item, ladder price),
                                             exactly the columns of
                                             synthetic_answers.csv (same order) plus
                                             `recovered` and `format_kind`; model =
                                             "gpt-oss:20b", model_version = "f38aa0c53da5",
                                             call_ts empty.

Checks printed (AGENTS.md rule 2): 22,000 rows; the set of (new_id, item, price_text)
identical to synthetic_answers.csv; every clean row matched exactly one
synthetic_answers.csv row; answer texts valid.
"""

import pandas as pd

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data/derived/robustness_answers_clean.csv"
SYN_CSV = ROOT / "data/derived/synthetic_answers.csv"
OUT_CSV = ROOT / "data/derived/robustness_answers_full.csv"

MODEL = "gpt-oss:20b"
MODEL_VERSION = "f38aa0c53da5"

YES_TEXT = "Yes, I would purchase the product"
NO_TEXT = "No, I would not purchase the product"

# exact column order of synthetic_answers.csv, plus the two extra columns
FULL_COLUMNS = [
    "new_id", "item", "category", "product", "price_text", "price_dollars",
    "relative_price", "free", "position_in_call", "answer", "model",
    "model_version", "call_ts", "recovered", "format_kind",
]


def main():
    clean = pd.read_csv(CLEAN_CSV)
    syn = pd.read_csv(SYN_CSV)
    print(f"robustness_answers_clean.csv: {len(clean)} rows loaded (expected 22,000 = 2,000 keys x 11 prices)")
    print(f"synthetic_answers.csv: {len(syn)} rows loaded (expected 22,000; metadata source only)")

    # --- assertions on the inputs -------------------------------------------
    assert len(clean) == 22000, f"clean file must have 22,000 rows, got {len(clean)}"
    assert (clean.groupby(["new_id", "item"]).size() == 11).all(), (
        "clean file must have exactly 11 ladder rows per (new_id, item)"
    )
    assert clean["new_id"].nunique() == 200, "clean file must cover 200 twins"
    assert set(clean["answer"].unique()) == {YES_TEXT, NO_TEXT}, "unexpected answer texts in clean file"
    assert ((clean["answer"] == YES_TEXT).astype(int) == clean["yes"]).all(), "yes inconsistent with answer in clean file"
    assert set(clean["recovered"].unique()) <= {0, 1}, "recovered must be 0/1"
    assert set(clean["format_kind"].unique()) <= {"root_array", "wrapper"}, "unexpected format_kind"

    syn_keys = set(map(tuple, syn[["new_id", "item", "price_text"]].itertuples(index=False, name=None)))
    clean_keys = set(map(tuple, clean[["new_id", "item", "price"]].itertuples(index=False, name=None)))
    assert syn_keys == clean_keys, (
        f"key sets differ: {len(syn_keys - clean_keys)} only in synthetic_answers.csv, "
        f"{len(clean_keys - syn_keys)} only in the clean file"
    )
    print(f"set of (new_id, item, price_text) identical to synthetic_answers.csv: "
          f"{len(clean_keys):,} keys, set difference empty (asserted)")

    # --- join the metadata ---------------------------------------------------
    meta = syn[["new_id", "item", "price_text", "category", "product", "price_dollars",
                "relative_price", "free", "position_in_call"]]
    assert not meta.duplicated(["new_id", "item", "price_text"]).any(), "duplicate key in synthetic_answers.csv"
    full = clean.rename(columns={"price": "price_text"}).merge(
        meta, on=["new_id", "item", "price_text"], how="left", validate="many_to_one"
    )
    assert len(full) == 22000, f"join must keep 22,000 rows, got {len(full)}"
    assert full["category"].notna().all(), "every clean row must match a synthetic_answers.csv row"
    print("join: 22,000 of 22,000 clean rows matched exactly one synthetic_answers.csv row (asserted)")

    # --- fixed fields, column order, dtypes ----------------------------------
    full["model"] = MODEL
    full["model_version"] = MODEL_VERSION
    full["call_ts"] = ""  # gpt-oss run: no per-call timestamp carried over
    for col in ("free", "position_in_call", "recovered"):
        full[col] = full[col].astype(int)
    for col in ("price_dollars", "relative_price"):
        full[col] = full[col].astype(float)
    full = full[FULL_COLUMNS]

    n_rec_keys = full.loc[full["recovered"] == 1, ["new_id", "item"]].drop_duplicates().shape[0]
    n_rec_rows = int((full["recovered"] == 1).sum())
    print(f"rows from recovered keys (marked recovered = 1): {n_rec_rows} rows "
          f"= {n_rec_keys} keys x 11 prices")
    print(f"format_kind counts (rows): {full['format_kind'].value_counts().to_dict()}")
    print(f"model fields set to '{MODEL}' version '{MODEL_VERSION}', call_ts empty (all rows)")

    full.to_csv(OUT_CSV, index=False)
    print(f"rows written: {len(full)} to {OUT_CSV.name} (columns: {', '.join(FULL_COLUMNS)})")
    print("done.")


if __name__ == "__main__":
    main()
