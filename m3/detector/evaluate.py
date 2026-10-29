"""Score detector variants from the stored span files (standard library only).

Usage: python -m m3.detector.evaluate [--final VARIANT]
  without --final : selection metrics on train+val ledger, challenge_dev and clean train+val queries
  with --final V  : also scores ledger test and challenge_heldout for V and V1 (run once, after selection)

Outputs in results/m3/detector/: m2_reproduction.json, person_44_breakdown.csv, variants_summary.csv,
<variant>.json, over_redaction_examples.json, final_<variant>.json
"""
import argparse
import collections
import csv
import json
import statistics
import time

from m3.common import DATA, DATA_M3, RESULTS, load_injections, load_splits, norm, read_jsonl, write_json
from m3.detector import metrics as M

OUT = RESULTS / "detector"
ADDRESS_RULE = "CustomUSAddressRecognizer"


def load_spans(engine, corpus):
    path = DATA_M3 / f"detector_spans_{engine}_{corpus}.jsonl.gz"
    if not path.exists():
        return None
    return {r["id"]: r for r in read_jsonl(path)}


def variant_spans():
    """{variant: {corpus: {id: {"spans": [(s, e, type, score)], "ms": float}}}} for available variants."""
    lg = {c: load_spans("lg", c) for c in ("ledger", "clean", "challenge")}
    trf = {c: load_spans("trf", c) for c in ("ledger", "clean", "challenge")}
    hf = {c: load_spans("hfner", c) for c in ("ledger", "clean", "challenge")}

    def conv(rows, drop_rule=False, extra=None):
        out = {}
        for rid, r in rows.items():
            spans = [tuple(s[:4]) for s in r["spans"] if not (drop_rule and s[4] == ADDRESS_RULE)]
            ms = r["ms"]
            if extra is not None:
                spans += [tuple(s[:4]) for s in extra[rid]["spans"]]
                ms += extra[rid]["ms"]
            out[rid] = {"spans": spans, "ms": ms}
        return out

    v = {"V0": {c: conv(lg[c], drop_rule=True) for c in lg}, "V1": {c: conv(lg[c]) for c in lg}}
    if all(trf.values()):
        v["V2"] = {c: conv(trf[c]) for c in trf}
    if all(hf.values()):
        v["V3"] = {c: conv(lg[c], extra=hf[c]) for c in lg}
    return v


def ledger_metrics(spans_by_id, ids, inj, clean_text, target_only=True):
    tot, cnt = collections.Counter(), collections.defaultdict(collections.Counter)
    for rid in ids:
        L = inj[rid]
        spans = spans_by_id[rid]["spans"]
        if target_only:
            spans = M.filter_target(spans)
        r = M.evaluate_record(L["raw_text"], spans, L["pii_type"], L["start_char"], L["end_char"],
                              L["pii_value"], clean_text[rid])
        for t in (L["pii_type"], "ALL"):
            tot[t] += 1
            for k in ("found", "full_span", "residual_leak", "partial_leak"):
                cnt[t][k] += r[k]
    out = {}
    for t in ("PERSON", "EMAIL", "PHONE", "ADDRESS", "ALL"):
        n = tot[t]
        out[t] = {"n": n, "found": cnt[t]["found"], "full_span": cnt[t]["full_span"],
                  "residual_leak": cnt[t]["residual_leak"], "partial_leak": cnt[t]["partial_leak"],
                  "detection_recall": round(cnt[t]["found"] / n, 4) if n else None,
                  "full_span_recall": round(cnt[t]["full_span"] / n, 4) if n else None,
                  "residual_leak_rate": round(cnt[t]["residual_leak"] / n, 4) if n else None}
    return out


def over_redaction(spans_by_id, ids, clean_text, n_examples=10):
    by_type, examples = collections.Counter(), collections.defaultdict(list)
    flagged = 0
    for rid in ids:
        tgt = M.filter_target(spans_by_id[rid]["spans"])
        flagged += bool(tgt)
        for t in sorted({s[2] for s in tgt}):
            by_type[t] += 1
            if len(examples[t]) < n_examples:
                s = next(x for x in tgt if x[2] == t)
                examples[t].append({"id": rid, "text": clean_text[rid], "span": clean_text[rid][s[0]:s[1]]})
    n = len(ids)
    return ({"n": n, "flagged": flagged, "over_redaction_rate": round(flagged / n, 4),
             "by_entity": {t: {"flagged": c, "rate": round(c / n, 4)} for t, c in sorted(by_type.items())}},
            dict(examples))


def challenge_metrics(spans_by_id, items):
    pos = collections.defaultdict(collections.Counter)
    tag = collections.defaultdict(collections.Counter)
    neg = collections.Counter()
    for it in items:
        tgt = M.filter_target(spans_by_id[it["id"]]["spans"])
        if it["category"] == "negative":
            neg["n"] += 1
            neg["flagged"] += bool(tgt)
            continue
        sp = it["spans"][0]
        r = M.evaluate_record(it["text"], tgt, sp["type"], sp["start"], sp["end"], sp["value"])
        for key, d in (((sp["type"]), pos), (("ALL"), pos), ((sp["type"] + "/" + it["format_tag"]), tag)):
            d[key]["n"] += 1
            for k in ("found", "full_span", "residual_leak"):
                d[key][k] += r[k]

    def fin(d):
        return {k: {**v, "detection_recall": round(v["found"] / v["n"], 4),
                    "full_span_recall": round(v["full_span"] / v["n"], 4),
                    "residual_leak_rate": round(v["residual_leak"] / v["n"], 4)} for k, v in sorted(d.items())}

    return {"by_type": fin(pos), "by_format": fin(tag),
            "negatives": {**neg, "false_positive_rate": round(neg["flagged"] / neg["n"], 4) if neg["n"] else None}}


def latency(spans_by_id, ids):
    ms = sorted(spans_by_id[i]["ms"] for i in ids)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2), "mean_ms": round(statistics.fmean(ms), 2)}


def mask_overhead(spans_by_id, ids, inj):
    t0 = time.perf_counter()
    for rid in ids:
        M.mask(inj[rid]["raw_text"], M.filter_target(spans_by_id[rid]["spans"]), "typed")
    return round((time.perf_counter() - t0) * 1000 / len(ids), 4)


def m2_reproduction(v1, inj, clean_text, ids_all):
    """M2 final configuration: matching-type metrics, leaks after masking with ALL returned spans."""
    res = ledger_metrics(v1["ledger"], ids_all, inj, clean_text, target_only=False)
    write_json(OUT / "m2_reproduction.json", {
        "description": "Presidio en_core_web_lg + address rule v2 on all 13,083 raw texts; leaks after masking with every returned span (M2 pilot behaviour)",
        "expected_from_M2": {"found_total": 13039, "PERSON_found": 3191, "PERSON_full_span": 3101,
                             "PERSON_complete_value_left": 32, "ADDRESS_full_span": 3351},
        "result": res,
        "reproduced": (res["ALL"]["found"] == 13039 and res["PERSON"]["found"] == 3191
                       and res["PERSON"]["full_span"] == 3101 and res["PERSON"]["residual_leak"] == 32
                       and res["ADDRESS"]["full_span"] == 3351)})
    return res


def person_44(v1, inj, ids_all):
    rows = []
    for rid in ids_all:
        L = inj[rid]
        if L["pii_type"] != "PERSON":
            continue
        spans = v1["ledger"][rid]["spans"]
        s0, e0 = L["start_char"], L["end_char"]
        if M.is_found(spans, "PERSON", s0, e0):
            continue
        cover = [f"{t}:{L['raw_text'][s:e]}" for s, e, t, _ in spans if s < e0 and e > s0]
        rows.append({"record_id": rid, "value": L["pii_value"], "placement": L["template_id"],
                     "covering_spans": "; ".join(cover) or "none",
                     "left_after_all_span_mask": M.is_residual_leak(M.mask(L["raw_text"], spans, "generic"), L["pii_value"]),
                     "left_after_target_span_mask": M.is_residual_leak(
                         M.mask(L["raw_text"], M.filter_target(spans), "generic"), L["pii_value"])})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "person_44_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final")
    a = ap.parse_args()
    inj = load_injections()
    splits = load_splits()
    clean_text = {r["record_id"]: norm(r["text"]) for r in read_jsonl(DATA / "banking_records.jsonl")}
    ids_all = sorted(inj)
    trval = sorted(i for i, s in splits.items() if s in ("train", "val"))
    test = sorted(i for i, s in splits.items() if s == "test")
    items = read_jsonl(DATA_M3 / "challenge_set.jsonl")
    dev = [it for it in items if it["subset"] == "challenge_dev"]
    held = [it for it in items if it["subset"] == "challenge_heldout"]
    V = variant_spans()

    rep = m2_reproduction(V["V1"], inj, clean_text, ids_all)
    p44 = person_44(V["V1"], inj, ids_all)
    print("M2 reproduction:", {t: (rep[t]["found"], rep[t]["full_span"], rep[t]["residual_leak"]) for t in rep})
    print("PERSON misses:", len(p44), "left after all-span mask:", sum(r["left_after_all_span_mask"] for r in p44),
          "left after target-span mask:", sum(r["left_after_target_span_mask"] for r in p44))

    summary, examples = [], {}
    for name, vs in V.items():
        led = ledger_metrics(vs["ledger"], trval, inj, clean_text)
        ovr, ex = over_redaction(vs["clean"], trval, clean_text)
        ch = challenge_metrics(vs["challenge"], dev)
        lat = latency(vs["ledger"], trval)
        res = {"variant": name, "scope": "selection (train+val ledger, challenge_dev, clean train+val)",
               "ledger": led, "over_redaction": ovr, "challenge_dev": ch, "latency": lat,
               "mask_ms_per_record": mask_overhead(vs["ledger"], trval[:2000], inj)}
        write_json(OUT / f"{name}.json", res)
        examples[name] = ex
        summary.append({"variant": name, "n_ledger": led["ALL"]["n"],
                        "detection_recall": led["ALL"]["detection_recall"],
                        "full_span_recall": led["ALL"]["full_span_recall"],
                        "residual_leak_rate": led["ALL"]["residual_leak_rate"],
                        "residual_leaks": led["ALL"]["residual_leak"], "partial_leaks": led["ALL"]["partial_leak"],
                        "person_detection_recall": led["PERSON"]["detection_recall"],
                        "person_residual_leaks": led["PERSON"]["residual_leak"],
                        "over_redaction_rate": ovr["over_redaction_rate"],
                        "challenge_dev_detection_recall": ch["by_type"]["ALL"]["detection_recall"],
                        "challenge_dev_leak_rate": ch["by_type"]["ALL"]["residual_leak_rate"],
                        "challenge_dev_negative_fp_rate": ch["negatives"]["false_positive_rate"],
                        "median_ms": lat["median_ms"], "p95_ms": lat["p95_ms"]})
        print(name, json.dumps(summary[-1]))
    write_json(OUT / "over_redaction_examples.json", examples)
    with open(OUT / "variants_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    if a.final:
        for name in sorted({a.final, "V1"}):
            vs = V[name]
            res = {"variant": name, "scope": "final (after selection): test ledger, challenge_heldout, clean test",
                   "ledger_test": ledger_metrics(vs["ledger"], test, inj, clean_text),
                   "ledger_all_13083": ledger_metrics(vs["ledger"], ids_all, inj, clean_text),
                   "over_redaction_test": over_redaction(vs["clean"], test, clean_text)[0],
                   "over_redaction_all": over_redaction(vs["clean"], ids_all, clean_text)[0],
                   "challenge_heldout": challenge_metrics(vs["challenge"], held),
                   "latency_all": latency(vs["ledger"], ids_all)}
            write_json(OUT / f"final_{name}.json", res)
            print("FINAL", name, json.dumps({k: res["ledger_test"]["ALL"] if k == "ledger_test" else None
                                             for k in ("ledger_test",)}))


if __name__ == "__main__":
    main()
