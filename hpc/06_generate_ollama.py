#!/usr/bin/env python3
"""06_generate_ollama.py — Robustness run: gpt-oss:20b via local Ollama
(plan.md section 5 "Robustness"; 15 Nov 2026 choices in section 7).

Runs on the Bocconi HPC compute node, started by run_ollama.sbatch, from
the directory that also holds prompts_for_robustness.jsonl (exported
verbatim from the valid main-run calls by src/05_export_prompts.py).
Python standard library only: no pandas, no pip installs.

Design:
- Same 2000 (twin, item) calls as the main run, same exact prompts and
  price orders. Per-call seed = 20261112 + call_index (recorded in every
  line). Temperature 1, as in the main run.
- POST http://127.0.0.1:11434/api/generate, stream false, keep_alive
  12h, options {temperature: 1, seed: per-call, num_ctx: 8192}.
- JSON schema in the Ollama "format" parameter: a ROOT-LEVEL ARRAY first
  (matches the prompt's "a list of 11 objects" instruction, like the
  main run's response_schema). If the root array fails (HTTP 4xx, or
  both attempts return invalid output), the script falls back
  PERMANENTLY to the {"answers": [...]} wrapper, unwraps it, and
  validates identically. Which schema was used is recorded per line
  ("format_kind") and reused on restart.
- Validation identical to src/02_generate.py, including the "$0"/"$0.00"
  tolerance (accepted and stored under "$0 (product is free)").
- One retry for invalid output, then permanent exclusion (count
  printed) — plus one extra attempt with the wrapper schema if the root
  array was the format in use, so a broken root-array schema can never
  exclude a key by itself.
- Every answer is appended + flushed immediately to
  robustness_answers.jsonl with prompt, response, thinking (gpt-oss's
  separate thinking field: logged, never parsed), model, Ollama version,
  timestamp, done_reason, eval stats, outcome and the parsed answers.
- Resume: keys with a valid or excluded line are skipped on restart.
  Server errors (timeouts, connection errors, 5xx) leave the key
  not-done; after 3 consecutive server errors the run stops (resubmit
  the sbatch to resume). No rate limit and no daily cap: Ollama runs
  locally on the same node. Walltime guard: after 11 h the script stops
  starting new calls and says to resubmit.

The Ollama version (0.40.16) and the model digest are asserted before any
call (the sbatch checks them first; this script re-checks the version).

No API key is involved anywhere: Ollama is local.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (plan.md section 7, 15 Nov 2026 choices)
# ---------------------------------------------------------------------------

BASE_URL = "http://127.0.0.1:11434"
MODEL = "gpt-oss:20b"
EXPECTED_OLLAMA_VERSION = "0.40.16"   # sbatch checks it too (main-run-style guard)
SEED_BASE = 20261112                  # per-call seed = SEED_BASE + call_index
TEMPERATURE = 1.0                     # as in the main run (set explicitly)
NUM_CTX = 8192                        # avoids silent context truncation (see docstring)
KEEP_ALIVE = "12h"                    # model must not unload between calls/restarts
REQUEST_TIMEOUT_S = 600
MAX_CONSECUTIVE_SERVER_ERRORS = 3     # mirrors the main-run policy
WALLTIME_GUARD_S = 11 * 3600          # stop starting new calls after 11 h of the 12 h

PROMPTS_FILE = Path("prompts_for_robustness.jsonl")
OUT_JSONL = Path("robustness_answers.jsonl")

FREE_TEXT = "$0 (product is free)"   # exactly as in randDollarsString
# Zero-price strings the model may write for the ladder's single $0 point;
# all are accepted and stored under the exact ladder text FREE_TEXT.
# Identical to src/02_generate.py (plan.md section 7 (j), 12 Nov 2026).
ZERO_PRICE_ALIASES = {FREE_TEXT, "$0", "$0.00"}
OPTION_YES = "Yes, I would purchase the product"
OPTION_NO = "No, I would not purchase the product"

# The 11 exact price texts per item, identical to src/02_generate.py.
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

# ---------------------------------------------------------------------------
# JSON schemas for Ollama's "format" parameter
# ---------------------------------------------------------------------------

_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "price": {"type": "string"},
        "answer": {"type": "string", "enum": [OPTION_YES, OPTION_NO]},
    },
    "required": ["price", "answer"],
}
_ARRAY_SCHEMA = {
    "type": "array", "minItems": 11, "maxItems": 11, "items": _ITEM_SCHEMA,
}
# root_array first (matches the prompt: "a list of 11 objects"); wrapper
# only as the documented fallback (unwrapped before validation).
FORMAT_SCHEMAS: dict[str, dict] = {
    "root_array": _ARRAY_SCHEMA,
    "wrapper": {
        "type": "object",
        "properties": {"answers": _ARRAY_SCHEMA},
        "required": ["answers"],
    },
}


class StopRun(Exception):
    """Raised to end the run cleanly (3 consecutive server errors,
    non-retryable HTTP error, walltime guard); nothing is excluded,
    unfinished keys stay not-done."""


def now_iso() -> str:
    """Cluster-local time with UTC offset (the main run used Pacific time
    only because of the free-tier quota; there is no quota here)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


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
                print(f"WARNING: unreadable JSONL line in {path} (skipped)", flush=True)
    return out


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record and flush immediately (save as it arrives)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Validation — identical checks to src/02_generate.py
# ---------------------------------------------------------------------------

def unwrap(raw_text: str) -> tuple[list | None, str | None]:
    """Parse the reply into the 11-object list. Accepts a root-level list
    (root_array schema) or the {"answers": [...]} wrapper (wrapper
    schema). Parsing is split out only because of the wrapper; the
    checks in validate() are exactly those of src/02_generate.py."""
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    if isinstance(data, dict) and set(data) == {"answers"}:
        data = data["answers"]
    if not isinstance(data, list):
        return None, f"expected a list, got {type(data).__name__}"
    return data, None


def validate(answer_list: list, ladder_texts: list[str]) -> tuple[dict | None, str | None]:
    """All 11 ladder price texts, each exactly once, answers Yes/No only
    (logic and messages as in src/02_generate.py)."""
    if len(answer_list) != len(ladder_texts):
        return None, (f"expected a list of {len(ladder_texts)} objects, "
                      f"got a list of length {len(answer_list)}")
    parsed: dict[str, str] = {}
    for obj in answer_list:
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


def unwrap_and_validate(raw_text: str,
                        ladder_texts: list[str]) -> tuple[dict | None, str | None]:
    data, err = unwrap(raw_text)
    if err is not None:
        return None, err
    return validate(data, ladder_texts)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_json(path: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_generate(prompt: str, schema: dict, seed: int) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": schema,
        "options": {"temperature": TEMPERATURE, "seed": seed, "num_ctx": NUM_CTX},
    }
    return http_json("/api/generate", payload, timeout=REQUEST_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Inputs + resume state
# ---------------------------------------------------------------------------

def load_prompts() -> list[dict]:
    if not PROMPTS_FILE.exists():
        sys.exit(f"ERROR: {PROMPTS_FILE} not found in this directory — copy it "
                 "from the Mac (data/derived/prompts_for_robustness.jsonl)")
    recs = read_jsonl(PROMPTS_FILE)
    print(f"Loaded {PROMPTS_FILE}: {len(recs)} lines", flush=True)
    if len(recs) != 2000:
        sys.exit(f"ERROR: expected 2000 prompt lines, got {len(recs)}")
    for i, rec in enumerate(recs):
        if set(rec) != {"call_index", "new_id", "item", "price_order", "prompt"}:
            sys.exit(f"ERROR: line {i} fields are {sorted(rec)}; expected exactly "
                     "{call_index, new_id, item, price_order, prompt}")
        if rec["call_index"] != i:
            sys.exit(f"ERROR: line {i} has call_index {rec['call_index']!r}")
        item = rec["item"]
        if not isinstance(item, int) or item not in LADDERS:
            sys.exit(f"ERROR: line {i} has unknown item {item!r}")
        po = rec["price_order"]
        if not isinstance(po, list) or sorted(po) != sorted(LADDERS[item]):
            sys.exit(f"ERROR: line {i} ({rec['new_id']}, item {item}): price_order "
                     "is not a permutation of the item's ladder")
        if not isinstance(rec["prompt"], str) or not rec["prompt"]:
            sys.exit(f"ERROR: line {i} ({rec['new_id']}): empty or missing prompt")
    print("All prompt checks passed: 2000 lines, call_index 0..1999, "
          "price_orders on ladders, prompts non-empty", flush=True)
    return recs


def resume_state() -> tuple[set, set, str | None]:
    """(done, excluded, last format_kind) from the answers JSONL. A key is
    done iff some line is valid; excluded iff some line outcome ==
    'excluded'. server_error / api_error lines leave the key not-done."""
    done, excluded = set(), set()
    last_kind: str | None = None
    for rec in read_jsonl(OUT_JSONL):
        key = (rec.get("new_id"), rec.get("item"))
        if rec.get("outcome") == "valid":
            done.add(key)
        elif rec.get("outcome") == "excluded":
            excluded.add(key)
        if rec.get("format_kind") in FORMAT_SCHEMAS:
            last_kind = rec["format_kind"]
    return done, excluded, last_kind


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N not-done keys "
                         "(smoke test: sbatch passes SMOKE)")
    args = ap.parse_args()

    log(f"=== robustness run start | model {MODEL} | Ollama at {BASE_URL} ===")

    # ---- server up + version guard (before any call) ----------------------
    try:
        ver = http_json("/api/version")["version"]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"FATAL: cannot reach Ollama at {BASE_URL}: {type(e).__name__}: {e}")
    log(f"Ollama server up: version {ver!r}")
    if ver != EXPECTED_OLLAMA_VERSION:
        sys.exit(f"FATAL: Ollama version {ver!r} != {EXPECTED_OLLAMA_VERSION!r}; "
                 "run stops before any call (nothing sent, nothing excluded)")

    # ---- inputs + resume ---------------------------------------------------
    prompts = load_prompts()
    done, excluded, last_kind = resume_state()
    format_kind = last_kind if last_kind in FORMAT_SCHEMAS else "root_array"
    log(f"resume: done={len(done)} excluded={len(excluded)} | "
        f"format_kind starting: {format_kind}"
        + (f" (from last saved record)" if last_kind else " (default)"))
    todo = [r for r in prompts
            if (r["new_id"], r["item"]) not in done
            and (r["new_id"], r["item"]) not in excluded]
    if args.limit is not None:
        todo = todo[:args.limit]
    log(f"this run: {len(todo)} calls"
        + (f" (limit {args.limit} — smoke test)" if args.limit is not None else ""))
    if not todo:
        log("nothing to do; exiting")
        return

    # ---- loop --------------------------------------------------------------
    counts = {"valid": 0, "invalid_retried": 0, "excluded_new": 0,
              "server_errors": 0, "schema_fallbacks": 0}
    consecutive_server_errors = 0
    processed = 0
    stop_reason = "finished"
    t0 = time.monotonic()

    try:
        for rec_in in todo:
            if time.monotonic() - t0 > WALLTIME_GUARD_S:
                stop_reason = "walltime_guard"
                log("walltime guard (11 h): stopping before the next call so the "
                    "job ends cleanly inside its 12 h limit; resubmit to resume")
                raise StopRun

            key = (rec_in["new_id"], rec_in["item"])
            item = rec_in["item"]
            prompt = rec_in["prompt"]
            ladder = LADDERS[item]
            seed = SEED_BASE + rec_in["call_index"]
            base_fields = {
                "run": "robustness",
                "call_index": rec_in["call_index"],
                "new_id": rec_in["new_id"],
                "item": item,
                "price_order": rec_in["price_order"],
                "prompt": prompt,
                "model": MODEL,
                "ollama_version": ver,
                "temperature": TEMPERATURE,
                "num_ctx": NUM_CTX,
                "seed": seed,
                "response": None,
                "thinking": None,
                "done_reason": None,
                "eval_count": None,
                "eval_duration": None,
                "error": None,
                "outcome": None,
                "parsed": None,
            }

            attempts = 0  # successful HTTP responses so far for this key
            while True:
                attempt = attempts + 1
                rec = dict(base_fields, ts=now_iso(), attempt=attempt,
                           format_kind=format_kind)

                # ---- send --------------------------------------------
                try:
                    resp = post_generate(prompt, FORMAT_SCHEMAS[format_kind], seed)
                except urllib.error.HTTPError as e:
                    try:
                        body = e.read().decode("utf-8", "replace")[:300]
                    except Exception:  # noqa: BLE001
                        body = ""
                    rec.update(error=f"HTTP {e.code}: {body}", outcome="api_error")
                    append_jsonl(OUT_JSONL, rec)
                    if 400 <= e.code < 500 and format_kind == "root_array":
                        format_kind = "wrapper"
                        counts["schema_fallbacks"] += 1
                        log(f"HTTP {e.code} on the root-array schema at {key}: "
                            "falling back to the wrapper schema for this key "
                            "and all later calls (recorded per line)")
                        continue  # same key, consumes no validation attempt
                    stop_reason = "api_error_stop"
                    log(f"non-retryable HTTP error {e.code} at {key}: run stops; "
                        "key stays not-done (fix the cause, then resubmit)")
                    raise StopRun
                except Exception as e:  # noqa: BLE001 — timeout / connection
                    rec.update(error=f"{type(e).__name__}: {e}",
                               outcome="server_error")
                    append_jsonl(OUT_JSONL, rec)
                    counts["server_errors"] += 1
                    consecutive_server_errors += 1
                    log(f"server error at {key} attempt {attempt}: "
                        f"{type(e).__name__}: {e} | key stays not-done "
                        f"({consecutive_server_errors}/"
                        f"{MAX_CONSECUTIVE_SERVER_ERRORS} consecutive)")
                    if consecutive_server_errors >= MAX_CONSECUTIVE_SERVER_ERRORS:
                        stop_reason = "server_errors_3x"
                        log("3 consecutive server errors: run stops so a "
                            "persistent problem does not loop; nothing excluded")
                        raise StopRun
                    break  # next key; this key is retried on the next restart

                # ---- response received --------------------------------
                consecutive_server_errors = 0
                attempts += 1
                rec.update(response=resp.get("response"),
                           thinking=resp.get("thinking"),
                           done_reason=resp.get("done_reason"),
                           eval_count=resp.get("eval_count"),
                           eval_duration=resp.get("eval_duration"))
                raw = rec["response"]
                parsed, verr = unwrap_and_validate(raw, ladder) \
                    if isinstance(raw, str) and raw else (None, "empty response")

                if verr is None:
                    rec.update(outcome="valid", parsed=parsed)
                    append_jsonl(OUT_JSONL, rec)
                    counts["valid"] += 1
                    log(f"valid {key} (attempt {attempt}, format {format_kind}, "
                        f"seed {seed}, {rec['eval_count']} output tokens)")
                    break
                if attempts == 1:
                    rec.update(error=verr, outcome="invalid_output")
                    append_jsonl(OUT_JSONL, rec)
                    counts["invalid_retried"] += 1
                    continue  # the single retry
                if format_kind == "root_array":
                    format_kind = "wrapper"
                    counts["schema_fallbacks"] += 1
                    log(f"root-array output invalid twice at {key}: falling back "
                        "to the wrapper schema for this key and all later calls "
                        "(recorded per line)")
                    continue  # one extra attempt with the wrapper
                rec.update(error=verr, outcome="excluded")
                append_jsonl(OUT_JSONL, rec)
                counts["excluded_new"] += 1
                log(f"excluded {key}: malformed after retry ({verr})")
                break

            processed += 1
            if processed % 50 == 0:
                elapsed = time.monotonic() - t0
                remaining = elapsed / processed * (len(todo) - processed)
                log(f"progress: {processed}/{len(todo)} calls this run | "
                    f"valid={counts['valid']} excluded_new={counts['excluded_new']} "
                    f"server_errors={counts['server_errors']} | elapsed "
                    f"{elapsed / 3600:.2f} h, projected remaining "
                    f"{remaining / 3600:.2f} h")
    except StopRun:
        pass

    # ---- wrap-up -------------------------------------------------------------
    total_done = len(done) + counts["valid"]
    total_excluded = len(excluded) + counts["excluded_new"]
    log(
        f"=== robustness run end ({stop_reason}) | "
        f"valid={counts['valid']} retried-once={counts['invalid_retried']} "
        f"excluded_new={counts['excluded_new']} server_errors={counts['server_errors']} "
        f"schema_fallbacks={counts['schema_fallbacks']} | "
        f"total done={total_done}/2000 | total excluded={total_excluded} | "
        f"format_kind in use: {format_kind}"
    )
    if total_excluded:
        print(f"NOTE: {total_excluded} (twin, item) calls excluded (malformed "
              "output after retry) — excluded from the analysis", flush=True)
    if stop_reason != "finished":
        log("Not all calls finished: resubmit the job (sbatch run_ollama.sbatch) "
            "to continue where it stopped; done and excluded calls are skipped.")


if __name__ == "__main__":
    main()
