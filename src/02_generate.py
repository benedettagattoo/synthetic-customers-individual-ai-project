"""02_generate.py — Synthetic twin answers (Gabor–Granger full ladder).

Implements plan.md section 4 (12 Nov 2026 redesign): one model call per
(synthetic twin, product); each call contains the twin's persona block, the
real survey intro (QID8 wording, "40" replaced by "11"), and the 11 ladder
price questions with the exact survey wording, price order randomised with
seed 20261112. Model: gemini-3.5-flash-lite, temperature 1, JSON output
validated, one retry, then permanent exclusion (count printed).

Volume / limits (plan.md section 4): 2,000 main calls + 30-call pilot;
free tier 15 requests/minute and 500 requests/day. The script enforces at
least 5 s between request starts (<= 12/min) and stops at 500 requests in
the current Pacific day (the free tier resets at midnight Pacific = 09:00
Italian time). Pilot and main share the same daily counter.

Error policy (approved 12 Nov 2026, amended before the first run):
- quota / 429 / RESOURCE_EXHAUSTED: hard stop; nothing retried, nothing
  excluded; the (twin, item) stays not-done; the sent request still counts
  toward the daily cap.
- other non-retryable API errors (4xx other than 429, e.g. 400/401/403/404):
  hard stop, same consequences (nothing excluded; the key stays not-done).
- network errors / 5xx: the (twin, item) is left not-done and is retried on
  the next restart; no retry is consumed and nothing is excluded; after
  3 consecutive API errors the run stops, so a persistent error cannot burn
  the daily quota.
- only invalid model outputs (bad JSON, failed validation) follow the
  one-retry-then-exclude rule.

--dry-run: run all checks, print the full prompt of the first call to make
and the work order of the pilot calls, and exit without calling the API
and without writing any file.

Inputs:  data/derived/sample_200_items.csv (200 twins x 10 items),
         data/raw/question_catalog.json (exact product/category wording).
Outputs: data/derived/synthetic_answers.jsonl (+ .csv)   [main run]
         data/derived/pilot_answers.jsonl (+ .csv)       [--pilot]
         data/derived/logs/run_YYYYMMDD_pacific.log      [one log per day]
Restart: every answer is appended + flushed as soon as it arrives; on
restart, (twin, item) pairs already valid or permanently excluded are
skipped; the seeded permutation table is redrawn identically (same
enumeration order), so price orders never change.

The API key is read from .env with python-dotenv and is never printed,
logged or echoed (AGENTS.md rule).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# Constants (plan.md section 4, approved 12 Nov 2026)
# ---------------------------------------------------------------------------

SEED = 20261112                      # price-order randomisation (section 4)
MODEL = "gemini-3.5-flash-lite"      # exact ID, not an alias/preview variant
EXPECTED_MODEL_VERSION = "3.5-flash-lite-07-2026"  # dated version, via the
# models endpoint (12 Nov 2026 09:10); note response.model_version is the
# model NAME, so it is logged but never compared (check happens at startup)
TEMPERATURE = 1.0                    # endpoint default, set explicitly
MIN_INTERVAL_S = 5.0                 # >= 5 s between request starts -> <= 12/min
DAILY_CAP = 500                      # free tier: 500 requests/day per model
PACIFIC = ZoneInfo("America/Los_Angeles")

SAMPLE_CSV = Path("data/derived/sample_200_items.csv")
CATALOG_JSON = Path("data/raw/question_catalog.json")
DERIVED = Path("data/derived")
LOG_DIR = DERIVED / "logs"

MAIN_JSONL = DERIVED / "synthetic_answers.jsonl"
MAIN_CSV = DERIVED / "synthetic_answers.csv"
PILOT_JSONL = DERIVED / "pilot_answers.jsonl"
PILOT_CSV = DERIVED / "pilot_answers.csv"

N_PILOT_TWINS = 3                    # pilot: 3 twins x 10 products = 30 calls

FREE_TEXT = "$0 (product is free)"   # exactly as in randDollarsString
# Zero-price strings the model may write for the ladder's single $0 point;
# all are accepted and stored under the exact ladder text FREE_TEXT.
# (Validation change after day 1, 12 Nov 2026 — plan.md section 7 (j).)
ZERO_PRICE_ALIASES = {FREE_TEXT, "$0", "$0.00"}
OPTION_YES = "Yes, I would purchase the product"
OPTION_NO = "No, I would not purchase the product"

# item number -> respondent 1's display position (plan.md section 3, item 4)
ITEM2QID = {
    25: "QID9_1", 27: "QID9_5", 31: "QID9_9", 9: "QID9_13",
    36: "QID9_17", 26: "QID9_21", 15: "QID9_25", 23: "QID9_29",
    38: "QID9_33", 10: "QID9_37",
}
ITEMS = list(ITEM2QID)  # 25, 27, 31, 9, 36, 26, 15, 23, 38, 10

# The 11 exact price texts per item, copied verbatim from the distinct prices
# in randDollarsString of the 1979 valid respondents (verified read-only,
# 12 Nov 2026: exactly 11 distinct texts per item, $0 as "$0 (product is
# free)"). Ladder = {0, M/10, ..., M} with M the item's maximum price.
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

# Structured-output schema: the model must answer with exactly these strings.
class _Answer(BaseModel):
    price: str
    answer: Literal["Yes, I would purchase the product",
                    "No, I would not purchase the product"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now_pacific() -> datetime:
    return datetime.now(PACIFIC)


def price_to_dollars(text: str) -> float:
    """'$7.39' -> 7.39; '$0 (product is free)' -> 0.0."""
    if text == FREE_TEXT:
        return 0.0
    assert text.startswith("$"), f"unexpected price text: {text!r}"
    return float(text[1:])


def pacific_date(ts_iso: str) -> str:
    """Pacific calendar date (YYYY-MM-DD) of a stored ISO timestamp."""
    return datetime.fromisoformat(ts_iso).astimezone(PACIFIC).strftime("%Y-%m-%d")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: unreadable JSONL line in {path} (skipped)")
    return out


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record and flush immediately (save as it arrives)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


class StopRun(Exception):
    """Raised to end the run cleanly (daily cap, quota, 4xx, 3 consecutive
    API errors); nothing is excluded, unfinished keys stay not-done."""


class RunLog:
    """One log per Pacific day (run metadata only — never prompts, never keys).
    In dry-run mode it prints only and writes no file."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        if dry_run:
            self.path = None
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / (
            "run_" + now_pacific().strftime("%Y%m%d") + "_pacific.log"
        )

    def write(self, msg: str) -> None:
        line = f"[{now_pacific().isoformat(timespec='seconds')}] {msg}"
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        print(line, flush=True)


# ---------------------------------------------------------------------------
# Load + verify inputs (prints counts after every check — AGENTS.md rule 2)
# ---------------------------------------------------------------------------

def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Loaded {SAMPLE_CSV}: {len(df)} rows")
    assert len(df) == 2000, f"expected 2000 rows, got {len(df)}"
    assert df.duplicated(["new_id", "item"]).sum() == 0, "duplicate (twin, item)"
    twins = list(dict.fromkeys(df["new_id"]))          # file order, deduped
    assert len(twins) == 200, f"expected 200 twins, got {len(twins)}"
    print(f"  {len(twins)} twins x 10 items, 0 duplicate (twin, item) pairs")

    # every twin has exactly the 10 items, in some order
    for twin, sub in df.groupby("new_id", sort=False):
        assert sorted(sub["item"]) == sorted(ITEMS), f"twin {twin}: item set wrong"

    # every price_text/price_dollars in the sample lies on its item's ladder
    ladder_text = {i: set(v) for i, v in LADDERS.items()}
    for r in df.itertuples():
        assert r.price_text in ladder_text[r.item], (
            f"twin {r.new_id} item {r.item}: price_text {r.price_text!r} "
            "not on the item's ladder"
        )
        assert abs(price_to_dollars(r.price_text) - r.price_dollars) < 1e-9, (
            f"twin {r.new_id} item {r.item}: price_dollars vs price_text mismatch"
        )
    n_free = int((df["price_text"] == FREE_TEXT).sum())
    print(f"  ladder check passed for all 2000 rows "
          f"({n_free} rows at $0 = '{FREE_TEXT}')")
    return df


def load_catalog_texts() -> dict[int, dict]:
    """Parse (category, product, embedded price) for our 10 items from the
    exact QID9_1/5/9/13/17/21/25/29/33/37 QuestionText (respondent 1's
    display positions), and assert the three approved checks."""
    catalog = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
    by_id = {e["QuestionID"]: e for e in catalog
             if isinstance(e, dict) and "QuestionID" in e}

    # intro QID8, verbatim except "40" -> "11" (approved choice, 12 Nov 2026)
    intro_raw = by_id["QID8"]["QuestionText"]
    assert "series of 40 questions" in intro_raw, "QID8 text changed?"
    intro = intro_raw.replace("series of 40 questions", "series of 11 questions")

    q_re = re.compile(
        r"^Please consider the following product category: (.+)\. "
        r"Suppose you are in a grocery store, and you see the following "
        r"product in that category: (.+)\. The product is priced at: "
        r"\$([0-9.]+)\. Would you or would you not purchase this product\?$"
    )
    meta: dict[int, dict] = {}
    for item, qid in ITEM2QID.items():
        assert qid in by_id, f"{qid} missing from {CATALOG_JSON}"
        m = q_re.match(by_id[qid]["QuestionText"])
        assert m, f"could not parse QuestionText of {qid}"
        category, product, price = m.group(1), m.group(2), float(m.group(3))
        assert category and product, f"empty category/product in {qid}"
        # (c) respondent 1's embedded price lies on the item's ladder
        ladder_vals = {price_to_dollars(t) for t in LADDERS[item]}
        assert price in ladder_vals, f"{qid}: catalog price ${price} not on ladder"
        meta[item] = {"category": category, "product": product, "catalog_price": price}
    print(f"Parsed product/category text for all {len(meta)} items "
          f"from {CATALOG_JSON} (QID9_1/5/9/13/17/21/25/29/33/37)")
    return meta, intro


def assert_categories_match(df: pd.DataFrame, meta: dict[int, dict]) -> None:
    """(b) catalog category equals the CSV category column for every row."""
    for item, info in meta.items():
        sub = df[df["item"] == item]
        assert set(sub["category"]) == {info["category"]}, (
            f"item {item}: catalog category {info['category']!r} != CSV "
            f"{sorted(set(sub['category']))!r}"
        )
    print("  catalog category == CSV category for all 10 items (assert passed)")


# ---------------------------------------------------------------------------
# Work list + seeded price-order permutations
# ---------------------------------------------------------------------------

def build_worklist(df: pd.DataFrame, pilot: bool) -> list[tuple[str, int]]:
    """(twin, item) pairs in deterministic order: twins sorted by new_id
    (C001, C002, ... — not file order: the CSV is grouped by income bracket,
    so file-order iteration would make the pilot 3 twins of one bracket and
    let each day cover different brackets, confounding day/model drift with
    income), each twin's items in file order."""
    twins = sorted(dict.fromkeys(df["new_id"]))
    if pilot:
        twins = twins[:N_PILOT_TWINS]
    worklist = []
    for twin in twins:
        sub = df[df["new_id"] == twin]
        for item in sub["item"]:                    # file order within twin
            worklist.append((twin, int(item)))
    n = len(worklist)
    print(f"Work list: {n} calls "
          f"({'PILOT: first ' + str(N_PILOT_TWINS) + ' twins x 10 products' if pilot else 'main run: 200 twins x 10 products'})")
    if pilot:
        assert n == N_PILOT_TWINS * 10 == 30, f"pilot must be 30 calls, got {n}"
    else:
        assert n == 2000, f"main run must be 2000 calls, got {n}"
    return worklist


def draw_permutations(worklist: list[tuple[str, int]]) -> dict[tuple[str, int], list[str]]:
    """Price order randomised per (twin, product), fixed seed 20261112
    (plan.md section 4): one RNG consumed in enumeration order, so the
    permutation table is identical on every restart and between pilot and
    main for shared keys. Drawn for ALL keys up front, including ones that
    will be skipped, to keep the stream aligned."""
    rng = random.Random(SEED)
    perms = {}
    for key in worklist:
        ladder = LADDERS[key[1]]
        perms[key] = rng.sample(ladder, len(ladder))
    print(f"Price-order permutations drawn: {len(perms)} "
          f"(single Random({SEED}) stream in enumeration order)")
    return perms


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(row: dict, meta: dict, intro: str,
                 price_order: list[str]) -> str:
    """Persona block (values only), QID8 intro ("11" questions), then the 11
    product questions in the randomised price order, then the JSON
    instruction (the only non-survey text)."""
    info = meta[int(row["item"])]
    q_lines = [
        "Please consider the following product category: "
        f"{info['category']}. Suppose you are in a grocery store, and you "
        f"see the following product in that category: {info['product']}. "
        f"The product is priced at: {ptxt}. Would you or would you not "
        "purchase this product?"
        for ptxt in price_order
    ]
    return (
        "You are simulating a survey respondent. This person's background:\n"
        "\n"
        f"Region: {row['QID11']}\n"
        f"Sex: {row['QID12']}\n"
        f"Age: {row['QID13']}\n"
        f"Education: {row['QID14']}\n"
        f"Total family income last year, before taxes: {row['QID21']}\n"
        "\n"
        f"{intro}\n"
        "\n"
        + "\n\n".join(q_lines)
        + "\n\n"
        "Answer each of the 11 questions above. Respond only with JSON: a "
        "list of 11 objects, each with \"price\" (the price exactly as "
        "written in the question) and \"answer\" (one of "
        f"\"{OPTION_YES}\" or \"{OPTION_NO}\"), in the order the questions "
        "were asked."
    )


# ---------------------------------------------------------------------------
# Response validation (plan.md section 4: JSON output validated)
# ---------------------------------------------------------------------------

def validate(raw_text: str, ladder_texts: list[str]) -> tuple[dict | None, str | None]:
    """All 11 ladder price texts, each exactly once, answers Yes/No only."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if not isinstance(data, list) or len(data) != len(ladder_texts):
        return None, (f"expected a list of {len(ladder_texts)} objects, "
                      f"got {type(data).__name__}"
                      + (f" of length {len(data)}" if isinstance(data, list) else ""))
    parsed: dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict) or set(obj) != {"price", "answer"}:
            return None, f"object without exactly keys price/answer: {obj!r}"
        raw_price, answer = obj["price"], obj["answer"]
        # formatting tolerance: the model may write the single zero price as
        # "$0" or "$0.00" instead of the ladder text; it is the same point
        # (each ladder has exactly one $0) and is stored under FREE_TEXT
        price = FREE_TEXT if raw_price in ZERO_PRICE_ALIASES else raw_price
        if price not in parsed:
            parsed[price] = answer
        else:
            return None, f"price answered more than once: {raw_price!r}"
    missing = [t for t in ladder_texts if t not in parsed]
    if missing:
        return None, f"missing ladder prices: {missing}"
    extra = [t for t in parsed if t not in ladder_texts]
    if extra:
        return None, f"unexpected prices: {extra}"
    for price, answer in parsed.items():
        if answer not in (OPTION_YES, OPTION_NO):
            return None, f"answer for {price!r} is not one of the two options: {answer!r}"
    return parsed, None


# ---------------------------------------------------------------------------
# Resume state + daily cap
# ---------------------------------------------------------------------------

def resume_state(path: Path) -> tuple[set, set]:
    """(done, excluded) (twin, item) keys from a JSONL. A key is done iff
    some line is valid; excluded iff some line outcome == 'excluded'.
    api_error lines leave the key not-done (retried on next restart)."""
    done, excluded = set(), set()
    for rec in read_jsonl(path):
        key = (rec.get("new_id"), rec.get("item"))
        if rec.get("outcome") == "valid":
            done.add(key)
        elif rec.get("outcome") == "excluded":
            excluded.add(key)
    return done, excluded


def revalidate_saved(jsonl_path: Path, done: set, dry_run: bool,
                     log: RunLog) -> set:
    """Re-validate saved invalid/excluded raw outputs with the current rule
    (zero-price aliases accepted). Keys that now validate are recovered
    WITHOUT new API calls: for each, a 'revalidated' record (a copy of the
    original record with outcome 'valid', the newly parsed answers, and
    revalidated/revalidated_ts markers) is appended. Idempotent: keys that
    already have a valid line are skipped, so restarts never duplicate
    recovery records. In dry-run nothing is written; the would-be recovery
    count is printed."""
    recovered: set = set()
    reval_ts = now_pacific().isoformat(timespec="seconds")
    for rec in read_jsonl(jsonl_path):
        key = (rec.get("new_id"), rec.get("item"))
        if key in done or key in recovered:
            continue
        if rec.get("outcome") not in ("invalid_output", "excluded"):
            continue
        raw = rec.get("raw_output")
        if not raw:
            continue
        parsed, verr = validate(raw, LADDERS[int(rec["item"])])
        if verr is not None:
            continue
        recovered.add(key)
        if not dry_run:
            new_rec = dict(rec)
            new_rec.update(outcome="valid", parsed=parsed, error=None,
                           revalidated=True, revalidated_ts=reval_ts)
            append_jsonl(jsonl_path, new_rec)
    if recovered:
        log.write(f"re-validation of saved invalid outputs "
                  f"({'dry-run: would write records for ' if dry_run else 'appended revalidated records for '})"
                  f"{len(recovered)} keys, recovered without new API calls: "
                  f"{sorted(recovered)}")
    else:
        log.write("re-validation of saved invalid outputs: nothing to recover")
    return recovered


def requests_today(paths: list[Path]) -> int:
    """Requests already sent today, by Pacific date, across pilot AND main
    JSONLs (both models' pilot and main share one quota per model/day)."""
    today = now_pacific().strftime("%Y-%m-%d")
    n = 0
    for path in paths:
        for rec in read_jsonl(path):
            if rec.get("ts") and pacific_date(rec["ts"]) == today:
                n += 1
    return n


# ---------------------------------------------------------------------------
# Tidy CSV rebuilt from the JSONL (idempotent; written at run start and end)
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["new_id", "item", "category", "product", "price_text",
               "price_dollars", "relative_price", "free", "position_in_call",
               "answer", "model", "model_version", "call_ts"]


def rebuild_csv(jsonl_path: Path, csv_path: Path, meta: dict[int, dict]) -> int:
    rows = []
    for rec in read_jsonl(jsonl_path):
        if rec.get("outcome") != "valid":
            continue
        item = int(rec["item"])
        info = meta[item]
        m = price_to_dollars(LADDERS[item][-1])
        for pos, ptxt in enumerate(rec["price_order"]):
            answer = rec["parsed"][ptxt]
            dollars = price_to_dollars(ptxt)
            rows.append({
                "new_id": rec["new_id"], "item": item,
                "category": info["category"], "product": info["product"],
                "price_text": ptxt, "price_dollars": dollars,
                "relative_price": dollars / m,
                "free": int(ptxt == FREE_TEXT),
                "position_in_call": pos, "answer": answer,
                "model": rec["model"], "model_version": rec["model_version"],
                "call_ts": rec["ts"],
            })
    out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    out.to_csv(csv_path, index=False)
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true",
                    help="run only the first 3 twins x 10 products (30 calls), "
                         "writing to separate pilot files")
    ap.add_argument("--dry-run", action="store_true",
                    help="run all checks, print the full prompt of the first "
                         "call to make and the work order, then exit without "
                         "calling the API and without writing any file")
    args = ap.parse_args()

    log = RunLog(dry_run=args.dry_run)
    mode = "pilot" if args.pilot else "main"
    jsonl_path = PILOT_JSONL if args.pilot else MAIN_JSONL
    csv_path = PILOT_CSV if args.pilot else MAIN_CSV
    log.write(f"=== {mode} run start | model {MODEL} "
              f"(expected version {EXPECTED_MODEL_VERSION}, temperature {TEMPERATURE}) ===")

    # ---- inputs + asserts -------------------------------------------------
    df = load_sample()
    meta, intro = load_catalog_texts()
    assert_categories_match(df, meta)
    worklist = build_worklist(df, args.pilot)
    perms = draw_permutations(worklist)
    rows = {(r.new_id, int(r.item)): r._asdict()
            for r in df.itertuples() if (r.new_id, int(r.item)) in perms}

    # ---- resume state + daily cap ----------------------------------------
    done, excluded = resume_state(jsonl_path)
    # re-validate saved invalid/excluded outputs with the current rule
    # (recovered keys get a 'revalidated' valid record, no new API calls)
    recovered = revalidate_saved(jsonl_path, done, args.dry_run, log)
    done = done | recovered
    excluded_eff = excluded - recovered
    todo = [k for k in worklist if k not in done and k not in excluded_eff]
    sent_today_start = requests_today([MAIN_JSONL, PILOT_JSONL])
    log.write(f"resume: done={len(done)} excluded={len(excluded_eff)} "
              f"remaining={len(todo)} | recovered by re-validation="
              f"{len(recovered)} | requests already sent today "
              f"(Pacific)={sent_today_start}/{DAILY_CAP}")

    if args.dry_run:
        log.write("DRY RUN: no API calls made, no files written")
        log.write(f"work order of the {len(todo)} calls to make "
                  f"(of {len(worklist)} in the {mode} run); "
                  f"listing the first 30 only:")
        for i, key in enumerate(todo[:30], 1):
            twin, item = key
            log.write(f"  {i:2d}. {twin} | item {item:2d} | "
                      f"{meta[item]['category']} | {meta[item]['product']}")
        if len(todo) > 30:
            log.write(f"  ... and {len(todo) - 30} more calls (not printed)")
        first = todo[0]
        prompt = build_prompt(rows[first], meta, intro, perms[first])
        print(f"\n----- full prompt for the first call to make "
              f"({first[0]}, item {first[1]}) -----\n")
        print(prompt)
        print("\n----- end of prompt -----")
        return

    csv_rows = rebuild_csv(jsonl_path, csv_path, meta)
    print(f"rebuilt {csv_path}: {csv_rows} answer rows "
          f"(one per twin x item x ladder price)")

    if not todo:
        log.write(f"nothing to do for the {mode} run; exiting")
        rebuild_csv(jsonl_path, csv_path, meta)
        return

    # ---- API client (key from .env, never printed) ------------------------
    load_dotenv()  # reads .env in the project root; .env is gitignored
    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set; put it in .env (never in code or logs)")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        response_schema=list[_Answer],
    )

    # ---- rate limiter -----------------------------------------------------
    last_start: list[float] = []

    def throttle() -> None:
        if last_start:
            wait = MIN_INTERVAL_S - (time.monotonic() - last_start[0])
            if wait > 0:
                time.sleep(wait)
        last_start.clear()
        last_start.append(time.monotonic())

    # ---- counters ----------------------------------------------------------
    sent_today = sent_today_start
    counts = {"valid": 0, "invalid_retried": 0, "excluded_new": 0,
              "api_error_skipped": 0, "quota_stop": 0}
    stop_reason = "finished"
    consecutive_api_errors = 0  # reset by every response the API returns

    try:
        # ---- model version check (once per run, before any call) ----------
        # The response-level `model_version` is the model NAME, so the dated
        # version can only be verified via the models endpoint (plan.md
        # section 4). Not a generate call: does not count toward the cap.
        try:
            model_info = client.models.get(model=MODEL)
        except Exception as e:  # noqa: BLE001 — never leak the key
            stop_reason = "models_get_failed"
            log.write(f"models.get({MODEL}) failed before any call: "
                      f"{type(e).__name__}: {e} | nothing sent, nothing excluded")
            raise StopRun
        start_version = getattr(model_info, "version", None)
        log.write(f"models.get: {MODEL} version={start_version!r} "
                  f"(expected {EXPECTED_MODEL_VERSION!r})")
        if start_version != EXPECTED_MODEL_VERSION:
            stop_reason = "model_version_mismatch"
            log.write("model version mismatch: run stops before any call; "
                      "nothing sent, nothing excluded")
            raise StopRun

        for key in todo:
            twin, item = key
            row = rows[key]
            price_order = perms[key]
            prompt = build_prompt(row, meta, intro, price_order)
            ladder_texts = LADDERS[item]

            for attempt in (1, 2):
                # daily cap by Pacific date (counts every sent request)
                if sent_today >= DAILY_CAP:
                    stop_reason = "daily_cap"
                    log.write(f"daily cap of {DAILY_CAP} requests reached "
                              f"({mode} run); {key} stays not-done")
                    raise StopRun

                throttle()
                sent_today += 1
                ts = now_pacific().isoformat(timespec="seconds")
                rec = {
                    "run": mode, "ts": ts, "new_id": twin, "item": item,
                    "category": meta[item]["category"],
                    "product": meta[item]["product"],
                    "price_order": price_order, "attempt": attempt,
                    "prompt": prompt, "raw_output": None,
                    "model": MODEL, "model_version": None,
                    "temperature": TEMPERATURE, "finish_reason": None,
                    "error": None, "outcome": None, "parsed": None,
                }

                # ---- one request -------------------------------------
                try:
                    resp = client.models.generate_content(
                        model=MODEL, contents=prompt, config=config,
                    )
                except Exception as e:  # noqa: BLE001 — classify, never leak the key
                    msg = f"{type(e).__name__}: {e}"
                    code = getattr(e, "code", None)
                    text = str(e)
                    is_quota = (code == 429 or "RESOURCE_EXHAUSTED" in text
                                or "quota" in text.lower())
                    if is_quota:
                        rec.update(error=msg, outcome="quota_stop")
                        append_jsonl(jsonl_path, rec)
                        counts["quota_stop"] += 1
                        stop_reason = "quota"
                        log.write(f"QUOTA/429 at {twin}/{item}: run stops; "
                                  f"{key} stays not-done (no retry, no exclusion)")
                        raise StopRun
                    if isinstance(code, int) and 400 <= code < 500:
                        # other non-retryable 4xx (400/401/403/404, ...):
                        # stop, nothing excluded, key stays not-done
                        rec.update(error=msg, outcome="api_4xx_stop")
                        append_jsonl(jsonl_path, rec)
                        stop_reason = "api_4xx"
                        log.write(f"non-retryable API error {code} at "
                                  f"{twin}/{item}: run stops; {key} stays "
                                  f"not-done (no retry, no exclusion)")
                        raise StopRun
                    # network error / 5xx: leave not-done, no retry consumed
                    rec.update(error=msg, outcome="api_error")
                    append_jsonl(jsonl_path, rec)
                    counts["api_error_skipped"] += 1
                    consecutive_api_errors += 1
                    log.write(f"API error at {twin}/{item} attempt {attempt}: "
                              f"{msg} | {key} stays not-done, retried on next "
                              f"restart (consecutive API errors: "
                              f"{consecutive_api_errors}/3)")
                    if consecutive_api_errors >= 3:
                        stop_reason = "api_errors_3x"
                        log.write("3 consecutive API errors: run stops so a "
                                  "persistent error cannot burn the daily "
                                  "quota; nothing excluded")
                        raise StopRun
                    break  # move to next key

                # ---- response received (HTTP success) ----------------
                consecutive_api_errors = 0
                raw = None
                try:
                    raw = resp.text
                except Exception:  # no text parts -> invalid output
                    raw = None
                candidates = list(resp.candidates or [])
                finish = None
                if candidates:
                    fr = candidates[0].finish_reason
                    finish = getattr(fr, "name", str(fr)) if fr is not None else None
                version = getattr(resp, "model_version", None)
                # NB: at response level the API reports the model NAME (e.g.
                # "gemini-3.5-flash-lite"), not the dated version — logged
                # as-is, never compared (see the models.get check at startup).
                rec.update(raw_output=raw, model_version=version, finish_reason=finish)

                parsed, verr = validate(raw, ladder_texts) if raw is not None \
                    else (None, "empty response text")
                if verr is None:
                    rec.update(outcome="valid", parsed=parsed)
                    append_jsonl(jsonl_path, rec)
                    counts["valid"] += 1
                    break  # done with this key
                if attempt == 1:
                    rec.update(error=verr, outcome="invalid_output")
                    append_jsonl(jsonl_path, rec)
                    counts["invalid_retried"] += 1
                    continue  # the single retry
                rec.update(error=verr, outcome="excluded")
                append_jsonl(jsonl_path, rec)
                counts["excluded_new"] += 1
                log.write(f"excluded {twin}/{item}: malformed after 1 retry "
                          f"({verr})")

            n_done = len(done) + counts["valid"]
            if (n_done % 50) == 0 and counts["valid"]:
                log.write(f"progress: {n_done} calls valid, "
                          f"requests today={sent_today}/{DAILY_CAP}")
    except StopRun:
        pass
    finally:
        # ---- wrap-up: counts + rebuilt CSV (rule 2) -----------------------
        total_excluded = len(excluded) + counts["excluded_new"]
        log.write(
            f"=== {mode} run end ({stop_reason}) | "
            f"valid={counts['valid']} retried-once={counts['invalid_retried']} "
            f"excluded_new={counts['excluded_new']} "
            f"api_error_skipped={counts['api_error_skipped']} "
            f"quota_stop={counts['quota_stop']} | "
            f"total excluded={total_excluded} | requests today="
            f"{sent_today}/{DAILY_CAP}"
        )
        csv_rows = rebuild_csv(jsonl_path, csv_path, meta)
        print(f"rebuilt {csv_path}: {csv_rows} answer rows")
        if total_excluded:
            print(f"NOTE: {total_excluded} (twin, item) calls excluded "
                  f"(malformed JSON after 1 retry) — excluded from the analysis")
        if stop_reason in ("daily_cap", "quota", "api_4xx", "api_errors_3x",
                           "model_version_mismatch", "models_get_failed"):
            print("Restart the script (same command) to continue where it "
                  "stopped; done and excluded calls are skipped.")


if __name__ == "__main__":
    main()
