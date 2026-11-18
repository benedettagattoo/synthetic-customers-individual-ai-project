"""05_export_prompts.py — Export the exact main-run prompts for the
robustness re-run (plan.md section 5; 15 Nov 2026 choices in section 7).

Reads data/derived/synthetic_answers.jsonl and writes
data/derived/prompts_for_robustness.jsonl, one line per valid main-run
(twin, item) call, with the prompt and price_order copied VERBATIM from
the stored record: the robustness model must answer the exact prompt the
main-run model answered (same wording, same randomised price order,
seed 20261112), so nothing is rebuilt here.

Output line fields: call_index (0-based position in this file; the
cluster script uses seed = 20261112 + call_index), new_id, item,
price_order, prompt. No pid anywhere (id_map.csv never leaves this
machine) and no demographic data beyond what is already inside the
prompt text (asserted: the key set is exactly those five fields).

Checks (AGENTS.md rule 2 — counts printed after every step):
- exactly 2000 valid keys in synthetic_answers.jsonl, at most one valid
  line per key;
- the (new_id, item) key set equals sample_200_items.csv exactly
  (200 twins x 10 items);
- every price_order is a permutation of its item's 11 ladder texts and
  every prompt is a non-empty string.

Inputs:  data/derived/synthetic_answers.jsonl, data/derived/sample_200_items.csv
Outputs: data/derived/prompts_for_robustness.jsonl
Run:     uv run python src/05_export_prompts.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SAMPLE_CSV = Path("data/derived/sample_200_items.csv")
MAIN_JSONL = Path("data/derived/synthetic_answers.jsonl")
OUT_JSONL = Path("data/derived/prompts_for_robustness.jsonl")

FREE_TEXT = "$0 (product is free)"   # exact text in every ladder, as in randDollarsString

# The 11 exact price texts per item, identical to src/02_generate.py
# (copied verbatim from the distinct prices in randDollarsString of the
# 1979 valid respondents; verified 12 Nov 2026).
LADDERS: dict[int, list[str]] = {
    25: [FREE_TEXT, "$1.06", "$2.11", "$3.17", "$4.22", "$5.28", "$6.34",
         "$7.39", "$8.45", "$9.50", "$10.56"],
    27: [FREE_TEXT, "$6.00", "$12.00", "$17.99", "$23.99", "$29.99", "$35.99",
         "$41.99", "$47.98", "$53.98", "$59.98"],
    31: [FREE_TEXT, "$0.95", "$1.90", "$2.84", "$3.79", "$4.74", "$5.69",
         "$6.64", "$7.58", "$8.53", "$9.48"],
    9:  [FREE_TEXT, "$0.99", "$1.97", "$2.96", "$3.94", "$4.93", "$5.92",
         "$6.90", "$7.89", "$8.87", "$9.86"],
    36: [FREE_TEXT, "$1.34", "$2.67", "$4.01", "$5.34", "$6.68", "$8.02",
         "$9.35", "$10.69", "$12.02", "$13.36"],
    26: [FREE_TEXT, "$1.12", "$2.23", "$3.35", "$4.46", "$5.58", "$6.70",
         "$7.81", "$8.93", "$10.04", "$11.16"],
    15: [FREE_TEXT, "$1.30", "$2.59", "$3.89", "$5.18", "$6.48", "$7.78",
         "$9.07", "$10.37", "$11.66", "$12.96"],
    23: [FREE_TEXT, "$0.85", "$1.71", "$2.56", "$3.42", "$4.27", "$5.12",
         "$5.98", "$6.83", "$7.69", "$8.54"],
    38: [FREE_TEXT, "$3.19", "$6.39", "$9.58", "$12.78", "$15.97", "$19.16",
         "$22.36", "$25.55", "$28.75", "$31.94"],
    10: [FREE_TEXT, "$0.54", "$1.07", "$1.61", "$2.14", "$2.68", "$3.22",
         "$3.75", "$4.29", "$4.82", "$5.36"],
}


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                sys.exit(f"ERROR: unreadable JSONL line in {path}")
    return out


def main() -> None:
    records = read_jsonl(MAIN_JSONL)
    print(f"Read {MAIN_JSONL}: {len(records)} lines")

    # ---- valid main-run calls (at most one valid line per key) -----------
    valid: dict[tuple[str, int], dict] = {}
    for rec in records:
        if rec.get("outcome") != "valid":
            continue
        key = (rec["new_id"], int(rec["item"]))
        if key in valid:
            sys.exit(f"ERROR: {key} has more than one valid line in {MAIN_JSONL}")
        valid[key] = rec
    print(f"Valid main-run calls: {len(valid)} (of {len(records)} lines)")
    if len(valid) != 2000:
        sys.exit(f"ERROR: expected 2000 valid keys, got {len(valid)}; "
                 "the main run is not complete — stopping")

    # ---- cross-check against the sample ----------------------------------
    sample_keys: set[tuple[str, int]] = set()
    with SAMPLE_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            sample_keys.add((row["new_id"], int(row["item"])))
    print(f"Read {SAMPLE_CSV}: {len(sample_keys)} (twin, item) keys")
    if sample_keys != set(valid):
        missing = sorted(sample_keys - set(valid))[:5]
        extra = sorted(set(valid) - sample_keys)[:5]
        sys.exit(f"ERROR: key sets differ; sample-only: {missing}..., "
                 f"answers-only: {extra}...")
    print("Key sets match exactly: 2000 = 200 twins x 10 items")

    # ---- field checks: price_order on ladder, prompt non-empty -----------
    for (new_id, item), rec in valid.items():
        po = rec.get("price_order")
        assert isinstance(po, list) and len(po) == 11 and sorted(po) == sorted(LADDERS[item]), (
            f"{new_id}/item {item}: price_order is not a permutation of the item's ladder"
        )
        assert isinstance(rec.get("prompt"), str) and rec["prompt"], (
            f"{new_id}/item {item}: empty or missing prompt"
        )
    print("All 2000 price_orders are permutations of their item's ladder; "
          "all prompts non-empty")

    # ---- write (deterministic order: sorted by new_id, then item) --------
    n_written = 0
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for i, ((new_id, item), rec) in enumerate(sorted(valid.items())):
            line = {
                "call_index": i,
                "new_id": new_id,
                "item": item,
                "price_order": rec["price_order"],
                "prompt": rec["prompt"],
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {OUT_JSONL}: {n_written} lines (call_index 0..{n_written - 1}); "
          "fields exactly {call_index, new_id, item, price_order, prompt} — "
          "no pid, no demographic data beyond the prompt text")
    print(f"Cluster seed per call = 20261112 + call_index "
          f"(0..{n_written - 1} -> seeds {20261112}..{20261112 + n_written - 1})")


if __name__ == "__main__":
    main()
