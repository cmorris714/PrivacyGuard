"""Build the text conditions used by every classifier experiment.

Usage: python -m m3.data.build_conditions
Writes data/m3/conditions.jsonl.gz (one row per record x condition).

  C0 original : normalized original query, no contact sentence
  C1 raw      : contact sentence with the synthetic value (M2 variants.jsonl, raw)
  C2/C3/C4    : delete / generic / typed with the recorded spans (M2 variants.jsonl)
  D2/D3/D4    : delete / generic / typed with the selected detector's spans applied to C1
The selected detector variant is read from results/m3/detector/selection.json.
"""
from m3.common import DATA, RESULTS, DATA_M3, load_injections, load_splits, norm, read_json, read_jsonl, write_jsonl
from m3.detector import metrics as M
from m3.detector.evaluate import variant_spans

ORACLE = {"C1": "raw", "C2": "delete", "C3": "generic", "C4": "typed"}
DETECTED = {"D2": "delete", "D3": "generic", "D4": "typed"}
CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "D2", "D3", "D4")


def build(variant=None):
    variant = variant or read_json(RESULTS / "detector" / "selection.json")["selected"]
    inj = load_injections()
    splits = load_splits()
    recs = {r["record_id"]: r for r in read_jsonl(DATA / "banking_records.jsonl")}
    var = {(r["record_id"], r["mask_method"]): r["transformed_text"] for r in read_jsonl(DATA / "variants.jsonl")}
    det = variant_spans()[variant]["ledger"]
    rows = []
    for rid in sorted(inj):
        L = inj[rid]
        base = {"record_id": rid, "split": splits.get(rid, "excluded"), "category": recs[rid]["category"],
                "pii_type": L["pii_type"], "placement": L["template_id"]}
        clean = norm(recs[rid]["text"])
        rows.append({**base, "condition": "C0", "text": clean, "detector_variant": None, "n_spans_masked": 0,
                     "residual_leak": False, "partial_leak": False})
        for c, m in ORACLE.items():
            text = var[(rid, m)]
            rows.append({**base, "condition": c, "text": text, "detector_variant": None,
                         "n_spans_masked": 0 if m == "raw" else 1,
                         "residual_leak": m == "raw", "partial_leak": False})
        spans = M.merge_spans(M.filter_target(det[rid]["spans"]))
        for c, m in DETECTED.items():
            text = M.mask(L["raw_text"], spans, m)
            leak = M.is_residual_leak(text, L["pii_value"])
            rows.append({**base, "condition": c, "text": text, "detector_variant": variant,
                         "n_spans_masked": len(spans), "residual_leak": leak,
                         "partial_leak": M.is_partial_leak(text, L["pii_value"], clean)})
    check(rows, inj, var, recs)
    write_jsonl(DATA_M3 / "conditions.jsonl.gz", rows)
    return rows


def check(rows, inj, var, recs):
    by_c = {}
    for r in rows:
        by_c.setdefault(r["condition"], []).append(r)
    for c in CONDITIONS:
        assert len(by_c[c]) == len(inj) == 13083, c
    for r in rows:
        rid, c = r["record_id"], r["condition"]
        if c in ORACLE:
            assert r["text"] == var[(rid, ORACLE[c])]
        if c in ("C2", "C3", "C4"):
            assert inj[rid]["pii_value"].casefold() not in r["text"].casefold()
        if c == "C0":
            assert r["text"] == norm(recs[rid]["text"])


def load_conditions(conditions=None, splits=None):
    rows = read_jsonl(DATA_M3 / "conditions.jsonl.gz")
    return [r for r in rows if (conditions is None or r["condition"] in conditions)
            and (splits is None or r["split"] in splits)]


if __name__ == "__main__":
    rows = build()
    import collections
    leaks = collections.Counter((r["condition"], r["split"]) for r in rows if r["residual_leak"] and r["condition"][0] == "D")
    print("rows:", len(rows), "detector leaks by (condition, split):", dict(sorted(leaks.items())))
