"""Offline re-validation of the robustness answers (plan.md section 7, entry 17 Nov 2026).

Reads data/derived/robustness_answers.jsonl (READ-ONLY, never modified) and applies
the re-validation rule decided on 17 Nov 2026, based only on output format:

Accepted price mappings (in order):
  1. exact match to one of the 11 ladder price strings of that item
     (the line's price_order);
  2. zero aliases -> "$0 (product is free)": "$0", "$0.00" (the original rule),
     plus "0" (new on 17 Nov 2026);
  3. NEW on 17 Nov 2026: a string without "$" whose text equals a ladder price
     without "$" (e.g. "19.16" -> "$19.16").
Nothing else: no rounding, no fixing of dropped decimal points (e.g. "$223").

A line is valid iff the 11 ladder prices each appear exactly once and every
answer is exactly one of the two survey texts.

Selection (user decision in chat, 17 Nov 2026, resolving the conflict between
the literal "first valid attempt" rule and the required identity check):
  - keys with a line valid under the ORIGINAL rule keep the original choice
    (first attempt valid under the original rule = the original attempt and
    answers); the new mappings change nothing for them;
  - keys with no line valid under the original rule take the first attempt
    (by attempt number) valid under the NEW rule -> these are the recovered
    keys, marked recovered = 1;
  - keys with no attempt valid under either rule stay excluded.

Outputs: data/derived/robustness_answers_clean.csv (one row per key and ladder
price, in the chosen call's price_order) and the printed counts/checks below.
No analysis, no model calls, no comparison with Gemini.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "derived" / "robustness_answers.jsonl"
OUT = ROOT / "data" / "derived" / "robustness_answers_clean.csv"

YES = "Yes, I would purchase the product"
NO = "No, I would not purchase the product"
ZERO = "$0 (product is free)"
ORIG_ZERO_ALIASES = {"$0": ZERO, "$0.00": ZERO}
NEW_ZERO_ALIASES = {"$0": ZERO, "$0.00": ZERO, "0": ZERO}


def parse_response(raw):
    """The JSON in "response": a root array, or an object with key "answers"."""
    d = json.loads(raw)
    if isinstance(d, dict):
        if "answers" not in d:
            raise ValueError("object without 'answers' key")
        d = d["answers"]
    if not isinstance(d, list):
        raise ValueError("not a list")
    return d


def validate(line, zero_aliases):
    """Map each entry's "price" to a ladder price under the given alias set.

    Returns {ladder price: answer} if the line is valid, else None.
    Valid = 11 ladder prices each exactly once, answers only the two texts.
    """
    ladder = line.get("price_order")
    if not isinstance(ladder, list) or len(ladder) != 11 or len(set(ladder)) != 11:
        return None
    try:
        entries = parse_response(line.get("response") or "")
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    answers = {}
    for e in entries:
        if not isinstance(e, dict):
            return None
        p, a = e.get("price"), e.get("answer")
        if not isinstance(p, str) or not isinstance(a, str):
            return None
        if p in ladder:
            t = p
        elif p in zero_aliases:
            t = zero_aliases[p]
        elif not p.startswith("$") and ("$" + p) in ladder:
            t = "$" + p
        else:
            return None
        if t in answers:  # a ladder price appearing twice -> invalid
            return None
        answers[t] = a
    if len(answers) != 11:
        return None
    if any(a not in (YES, NO) for a in answers.values()):
        return None
    return answers


def new_rules_used(line):
    """Which of the two 17 Nov rules the chosen line actually needed:
    '"0" alias' if some price was written as "0", 'no "$"' if some price was a
    ladder price written without the "$" sign."""
    ladder = line["price_order"]
    kinds = set()
    for e in parse_response(line["response"]):
        p = e["price"]
        if p == "0":
            kinds.add('"0" alias')
        elif p not in ladder and not p.startswith("$") and ("$" + p) in ladder:
            kinds.add("no \"$\"")
    return kinds


def main():
    lines = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    print(f"read {len(lines)} lines from {SRC.name} (read-only, file not modified)")

    keys = defaultdict(list)
    for l in lines:
        keys[(l["new_id"], l["item"])].append(l)
    print(f"keys (new_id, item): {len(keys)}")

    # Original validity per line, and the key set according to the run's own
    # outcome field (cross-check).
    outcome_valid_keys = {
        k for k, v in keys.items() if any(x.get("outcome") == "valid" for x in v)
    }

    chosen = {}      # key -> (line, answers, recovered)
    orig_first = {}  # key -> (line, answers) under the ORIGINAL rule
    excluded = []
    for k, v in keys.items():
        for line in sorted(v, key=lambda x: x["attempt"]):
            ans = validate(line, ORIG_ZERO_ALIASES)
            if ans is not None:
                orig_first[k] = (line, ans)
                break
    for k, v in keys.items():
        if k in orig_first:
            line, ans = orig_first[k]
            chosen[k] = (line, ans, 0)
            continue
        for line in sorted(v, key=lambda x: x["attempt"]):
            ans = validate(line, NEW_ZERO_ALIASES)
            if ans is not None:
                chosen[k] = (line, ans, 1)
                break
        else:
            excluded.append(k)

    n_orig = len(orig_first)
    recovered = sorted(k for k, (_, _, r) in chosen.items() if r == 1)
    print(f"keys valid originally: {n_orig}")
    print(f"cross-check vs the file's outcome field (1984 expected): "
          f"{len(outcome_valid_keys)} keys with an outcome=='valid' line; "
          f"identical key set: {set(orig_first) == outcome_valid_keys}")
    print(f"keys recovered by the new rule: {len(recovered)}")
    for k in recovered:
        line, _, _ = chosen[k]
        kinds = sorted(new_rules_used(line))
        print(f"  {k[0]} item {k[1]}: attempt {line['attempt']} "
              f"(format_kind {line['format_kind']}), recovered by: "
              f"{', '.join(kinds) if kinds else '(none needed - unexpected)'}")
    print(f"keys still excluded: {len(excluded)} {sorted(excluded)}")

    # Check: for the originally valid keys the chosen attempt and answers are
    # identical to the original ones (the run's own outcome=='valid' line and
    # its parsed field).
    mismatches = []
    for k, (line, ans, _) in chosen.items():
        if k not in outcome_valid_keys:
            continue
        vl = next(x for x in keys[k] if x.get("outcome") == "valid")
        if line["attempt"] != vl["attempt"] or ans != vl.get("parsed"):
            mismatches.append((k, line["attempt"], vl["attempt"]))
    if mismatches:
        print(f"CHECK FAILED: {len(mismatches)} originally valid keys where the "
              f"chosen attempt/answers differ from the original: {mismatches[:10]} ...")
    else:
        print(f"check OK: for all {n_orig} originally valid keys the chosen "
              f"attempt and answers are identical to the original ones")

    # Write the clean CSV: one row per key and price, in the chosen call's
    # price_order (same ordering convention as synthetic_answers.csv).
    rows = []
    for k in sorted(chosen):
        line, ans, rec = chosen[k]
        for price in line["price_order"]:
            a = ans[price]
            rows.append({
                "new_id": k[0],
                "item": k[1],
                "price": price,
                "answer": a,
                "yes": 1 if a == YES else 0,
                "attempt_used": line["attempt"],
                "format_kind": line["format_kind"],
                "recovered": rec,
            })
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["new_id", "item", "price", "answer",
                                          "yes", "attempt_used", "format_kind",
                                          "recovered"])
        w.writeheader()
        w.writerows(rows)
    n_r = sum(1 for r in rows if r["recovered"] == 1)
    print(f"rows written: {len(rows)} to {OUT.name} "
          f"({len(chosen)} keys x 11 ladder prices; {n_r} rows from recovered keys)")


if __name__ == "__main__":
    main()
