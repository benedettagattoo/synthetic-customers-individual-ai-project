"""Real demand curves for plan.md section 5, comparison 2 — real side only.

Built from data/raw only (no reading of src/, results/ or data/derived):
  - wave_4_labels_anonymized.csv : per-respondent display order + prices
    (randDollarsString) and wave-4 answers in display order ({k}_Q295).
  - wave1_3_response_label.csv   : QID21 family income bracket.
  - question_catalog.json        : product / category texts for the 10 items
    (via respondent 1's display positions 1,5,9,...,37).

Outputs (all in results/agent_run/):
  - agent_real_items_long.csv    : 1 row per valid respondent x item (10 items)
  - agent_real_demand_curves.csv : share of Yes per item x income bracket x price
No randomness, no model calls -> no seed.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "agent_run"
OUT.mkdir(parents=True, exist_ok=True)

ITEMS = [25, 27, 31, 9, 36, 26, 15, 23, 38, 10]  # plan.md section 3 (item numbers)
POS_OF_ITEMS = {1: 25, 5: 27, 9: 31, 13: 9, 17: 36, 21: 26, 25: 15, 29: 23, 33: 38, 37: 10}

INCOME_ORDER = [
    "Less than $30,000",
    "$30,000-$50,000",
    "$50,000-$75,000",
    "$75,000-$100,000",
    "$100,000 or more",
]
YES = "Yes, I would purchase the product"
NO = "No, I would not purchase the product"
FREE_TEXT = "$0 (product is free)"

PAIR_RE = re.compile(r"^(\d+):\$([0-9]+(?:\.[0-9]+)?)( \(product is free\))?$")


def parse_dollars(s):
    """Return (order, prices) or None if malformed.

    order  : list of 40 item numbers in display order
    prices : dict item -> price in dollars
    Malformed = not exactly 40 well-formed "item:price" pairs with distinct
    items forming 1..40 (the 79 bad rows have 41-44 pairs, one item repeated
    with two different prices).
    """
    if not isinstance(s, str):
        return None
    order, prices, seen = [], {}, set()
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok or tok == '`""`':
            continue
        m = PAIR_RE.match(tok)
        if m is None:
            return None
        item, price = int(m.group(1)), float(m.group(2))
        if item in seen:
            return None
        seen.add(item)
        order.append(item)
        prices[item] = price
    if len(order) != 40 or sorted(order) != list(range(1, 41)):
        return None
    return order, prices


def price_text(p):
    return FREE_TEXT if p == 0.0 else f"${p:.2f}"


def main():
    # ---- 1. raw wave-4 file (2058 data rows; line 1 = names, 2-3 = text/ImportId)
    raw = pd.read_csv(
        RAW / "wave_4_labels_anonymized.csv",
        skiprows=[1, 2],
        encoding="utf-8-sig",
        low_memory=False,
    )
    print(f"wave_4_labels_anonymized.csv: {len(raw)} rows read (expect 2058)")
    assert len(raw) == 2058
    raw = raw.rename(columns={"TWIN_ID": "pid"})

    # ---- 2. demographics (income bracket)
    demo = pd.read_csv(RAW / "wave1_3_response_label.csv", low_memory=False)
    print(f"wave1_3_response_label.csv: {len(demo)} rows read (expect 2058)")
    assert len(demo) == 2058
    assert demo["pid"].is_unique and raw["pid"].is_unique
    assert set(demo["pid"]) == set(raw["pid"])
    merged = raw.merge(demo[["pid", "QID21"]], on="pid", how="inner", validate="1:1")
    print(f"after 1:1 merge on pid: {len(merged)} rows (expect 2058, no rows lost)")
    assert len(merged) == 2058
    assert merged["QID21"].isin(INCOME_ORDER).all(), "unexpected QID21 value"
    print("QID21 income brackets, n respondents:")
    for b in INCOME_ORDER:
        print(f"  {b}: {(merged['QID21'] == b).sum()}")

    # ---- 3. parse randDollarsString, exclude malformed (plan.md section 3, 1b)
    parsed = merged["randDollarsString"].map(parse_dollars)
    valid_mask = parsed.notna()
    n_excl = int((~valid_mask).sum())
    print(
        f"malformed randDollarsString (41-44 pairs, one item repeated with two "
        f"different prices): {n_excl} of {len(merged)} excluded; "
        f"{int(valid_mask.sum())} valid rows kept (expect 79 / 1979)"
    )
    assert n_excl == 79 and int(valid_mask.sum()) == 1979
    data = merged.loc[valid_mask].copy()
    data["parsed"] = parsed.loc[valid_mask]
    print("valid respondents by income bracket:")
    for b in INCOME_ORDER:
        print(f"  {b}: {(data['QID21'] == b).sum()}")

    # ---- 4. ladders: per item, exactly 11 distinct prices among valid rows
    all_prices = {i: [] for i in range(1, 41)}
    for _, prices in data["parsed"]:
        for it, p in prices.items():
            all_prices[it].append(p)
    ladder = {}
    for it in range(1, 41):
        distinct = sorted(set(all_prices[it]))
        assert len(distinct) == 11, f"item {it}: {len(distinct)} distinct prices"
        M = distinct[-1]
        steps = [k * M / 10 for k in range(11)]
        for p in distinct:  # ladder texts are rounded to cents
            assert min(abs(p - s) for s in steps) < 0.005, (it, p)
        ladder[it] = {"M": M, "prices": distinct}
    print("ladder maxima M of the 10 items:")
    for it in ITEMS:
        L = ladder[it]
        print(f"  item {it:2d}: M=${L['M']:.2f}  ladder: "
              + ", ".join(f"{p:.2f}" for p in L["prices"]))

    # ---- 5. respondent-1 catalog check (catalog price = pid 1's draw, 10/10)
    cat = {e["QuestionID"]: e for e in json.load(open(RAW / "question_catalog.json"))}
    order1, prices1 = data.loc[data["pid"] == 1, "parsed"].iloc[0]
    item_at_pos = {pos: it for pos, it in enumerate(order1, start=1)}
    product_info = {}
    print("respondent 1 vs catalog (QID9_1, 5, ..., 37), 10 checks:")
    for pos in sorted(POS_OF_ITEMS):
        it = item_at_pos[pos]
        assert it == POS_OF_ITEMS[pos], (pos, it)
        m = re.search(
            r"category: (.+?)\. Suppose you are in a grocery store, and you see the "
            r"following product in that category: (.+?)\. The product is priced at: "
            r"\$([0-9.]+)\.",
            cat[f"QID9_{pos}"]["QuestionText"],
        )
        assert m, pos
        assert float(m.group(3)) == prices1[it], (pos, it)
        product_info[it] = {"category": m.group(1), "product": m.group(2)}
        print(f"  position {pos:2d} -> item {it:2d}, catalog price ${m.group(3)}"
              f" == pid-1 price ${prices1[it]:.2f}  OK")
    print("  catalog check 10/10 OK")

    # ---- 6. long respondent x item table for the 10 items
    q295 = {pos: data[f"{pos}_Q295"].to_numpy() for pos in range(1, 41)}
    rows = []
    pids = data["pid"].to_numpy()
    qid21 = data["QID21"].to_numpy()
    for r_idx in range(len(data)):
        order, prices = data["parsed"].iloc[r_idx]
        for it in ITEMS:
            pos = order.index(it) + 1  # display position of item it
            rows.append(
                {
                    "pid": pids[r_idx],
                    "QID21": qid21[r_idx],
                    "item": it,
                    "position": pos,
                    "answer_text": q295[pos][r_idx],
                    "price_dollars": prices[it],
                }
            )
    long = pd.DataFrame(rows).sort_values(["pid", "item"], ignore_index=True)
    print(f"long respondent x item table: {len(long)} rows "
          f"(= 1979 respondents x 10 items)")
    assert len(long) == 1979 * 10
    assert set(long["answer_text"]) == {YES, NO}, sorted(set(long["answer_text"]))[:]
    assert long.groupby("pid").size().eq(10).all()
    long["yes"] = (long["answer_text"] == YES).astype(int)

    # ---- 7. sanity anchor over ALL 40 items (counts fixed in plan.md, 12 Nov):
    #         5866 Yes of 7193 at $0 and 28935 Yes of 71967 above $0.
    n0 = n0_yes = npos = npos_yes = 0
    for r_idx in range(len(data)):
        order, prices = data["parsed"].iloc[r_idx]
        for pos, it in enumerate(order, start=1):
            a = q295[pos][r_idx]
            if prices[it] == 0.0:
                n0 += 1
                n0_yes += a == YES
            else:
                npos += 1
                npos_yes += a == YES
    print(f"sanity anchor (all 40 items, positional pairing): "
          f"${0} -> {n0_yes} Yes of {n0} (expect 5866/7193); "
          f">${0} -> {npos_yes} Yes of {npos} (expect 28935/71967)")
    assert (n0, n0_yes) == (7193, 5866), (n0, n0_yes)
    assert (npos, npos_yes) == (71967, 28935), (npos, npos_yes)
    print("  sanity anchor matched exactly")

    # ---- 8. relative price, free indicator, ladder index, price text
    M_of = {it: ladder[it]["M"] for it in ITEMS}
    long["ladder_max"] = long["item"].map(M_of)
    long["relative_price"] = long["price_dollars"] / long["ladder_max"]
    long["free"] = (long["price_dollars"] == 0.0).astype(int)
    long["price_text"] = long["price_dollars"].map(price_text)
    step_of = {it: ladder[it]["M"] / 10 for it in ITEMS}
    ladder_idx = (long["price_dollars"] / long["item"].map(step_of)).round()
    assert ((long["price_dollars"] - ladder_idx * long["item"].map(step_of))
            .abs().max() < 0.02)
    long["ladder_index"] = ladder_idx.astype(int)
    assert long["ladder_index"].between(0, 10).all()
    assert long.groupby("item")["ladder_index"].nunique().eq(11).all()

    # ---- 9. demand-curve cells: item x income bracket x price (+ pooled brackets)
    cells = []
    groups = [(b, long[long["QID21"] == b]) for b in INCOME_ORDER]
    groups.append(("All brackets", long))
    for it in ITEMS:
        L = ladder[it]
        for bname, g in groups:
            gi = g[g["item"] == it]
            for k, p in enumerate(L["prices"]):
                sub = gi[gi["price_dollars"] == p]
                cells.append(
                    {
                        "item": it,
                        "category": product_info[it]["category"],
                        "product": product_info[it]["product"],
                        "QID21": bname,
                        "price_dollars": p,
                        "price_text": price_text(p),
                        "relative_price": p / L["M"],
                        "free": int(p == 0.0),
                        "ladder_index": k,
                        "n_respondents": len(sub),
                        "n_yes": int(sub["yes"].sum()),
                        "n_no": int((sub["yes"] == 0).sum()),
                        "share_yes": (sub["yes"].mean() if len(sub) else None),
                    }
                )
    cells = pd.DataFrame(cells)
    assert len(cells) == 10 * 6 * 11  # items x (5 brackets + pooled) x 11 prices
    empty = cells[cells["n_respondents"] == 0]
    print(f"demand-curve cells: {len(cells)} (expect 660 = 10 items x 6 groups "
          f"x 11 prices); empty cells: {len(empty)}")
    assert len(empty) == 0
    per_item_bracket = cells[cells["QID21"] != "All brackets"]
    nn = per_item_bracket["n_respondents"]
    print(f"respondents per (item, bracket, price) cell: "
          f"min {nn.min()}, median {nn.median():.0f}, max {nn.max()} "
          f"(plan section 5 expected ~36)")
    # row counts must add up: per item and bracket, sum over the 11 prices
    for b in INCOME_ORDER:
        nb = int((long["QID21"] == b).sum()) // 10  # respondents in bracket
        for it in ITEMS:
            s = per_item_bracket[(per_item_bracket["QID21"] == b)
                                 & (per_item_bracket["item"] == it)]
            assert s["n_respondents"].sum() == nb, (b, it, s["n_respondents"].sum())
    print("  per-bracket row counts add up for every (bracket, item) pair")

    # ---- 10. write outputs
    long_out = long[[
        "pid", "QID21", "item", "position", "answer_text", "yes",
        "price_dollars", "price_text", "relative_price", "free", "ladder_index",
        "ladder_max",
    ]]
    long_out.to_csv(OUT / "agent_real_items_long.csv", index=False)
    cells_sorted = cells.assign(
        bkey=cells["QID21"].map({b: i for i, b in enumerate(INCOME_ORDER)} | {"All brackets": 5})
    ).sort_values(["item", "bkey", "price_dollars"], ignore_index=True).drop(columns="bkey")
    cells_sorted.to_csv(OUT / "agent_real_demand_curves.csv", index=False)
    print(f"wrote {OUT / 'agent_real_items_long.csv'} ({len(long_out)} rows)")
    print(f"wrote {OUT / 'agent_real_demand_curves.csv'} ({len(cells_sorted)} rows)")


if __name__ == "__main__":
    main()
