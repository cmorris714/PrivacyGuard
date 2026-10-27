"""Test-set analysis of the frozen experiments, recomputed from the saved prediction files.

Usage: python -m m3.analysis.evaluate
Writes to results/m3/analysis/: main_table.csv, per_intent_recall.csv, small_intents.csv, terciles.csv,
slices.csv, confusion_<protocol>_<cond>.csv (selected model, C1 and best method), top_confusions.csv,
seeds.csv, findings.md, analysis.json
"""
import collections
import csv
import json

import numpy as np
import yaml

from m3.analysis import stats as S
from m3.common import DATA, DATA_M3, RESULTS, ROOT, load_injections, load_splits, read_json, read_jsonl, write_json
from m3.models.evaluation import read_predictions

OUT = RESULTS / "analysis"
RUNS = RESULTS / "runs"
METHODS = {"C2": "delete", "C3": "generic", "C4": "typed"}
COND_NAMES = {"C0": "original (no contact sentence)", "C1": "raw", "C2": "delete", "C3": "generic [REDACTED]",
              "C4": "typed [TYPE]", "D2": "delete (detector)", "D3": "generic (detector)", "D4": "typed (detector)"}


def protocols():
    """{protocol: {"model": name, "conds": {cond: (run_id, cond)}}} for the runs that exist."""
    fam = yaml.safe_load(open(ROOT / "configs" / "frozen_model.yaml", encoding="utf-8"))["model"]["family"]
    model = {"transformer": "DistilBERT", "embed_clf": "MiniLM+clf", "tfidf_svm": "TF-IDF+SVM"}[fam]
    P = {
        "frozen": {"model": model, "conds": {c: ("e2a_frozen_c1_s553", c) for c in
                                             ("C0", "C1", "C2", "C3", "C4", "D2", "D3", "D4")}},
        "retrained": {"model": model, "conds": {"C1": ("e2a_frozen_c1_s553", "C1"), "C0": ("e2b_frozen_c0_s553", "C0"),
                                                **{c: (f"e2b_frozen_{c.lower()}_s553", c) for c in METHODS},
                                                **{"D" + c[1]: (f"e2b_frozen_{c.lower()}_s553", "D" + c[1]) for c in METHODS}}},
        "frozen_tfidf": {"model": "TF-IDF+SVM", "conds": {c: ("e2c_tfidf_svm_c1_s553", c) for c in
                                                         ("C0", "C1", "C2", "C3", "C4", "D2", "D3", "D4")}},
        "retrained_tfidf": {"model": "TF-IDF+SVM", "conds": {"C1": ("e2c_tfidf_svm_c1_s553", "C1"),
                                                             "C0": ("e2c_tfidf_svm_c0_s553", "C0"),
                                                             **{c: (f"e2c_tfidf_svm_{c.lower()}_s553", c) for c in METHODS},
                                                             **{"D" + c[1]: (f"e2c_tfidf_svm_{c.lower()}_s553", "D" + c[1])
                                                                for c in METHODS}}},
    }
    for p in P.values():
        p["conds"] = {c: v for c, v in p["conds"].items()
                      if (RUNS / v[0] / f"predictions_test_{v[1]}.csv.gz").exists()}
    return {k: v for k, v in P.items() if "C1" in v["conds"]}


def load(run, cond, labels):
    rows = sorted(read_predictions(RUNS / run / f"predictions_test_{cond}.csv.gz"), key=lambda r: r["record_id"])
    ids = [r["record_id"] for r in rows]
    gold = np.array([labels[r["gold"]] for r in rows])
    pred = np.array([labels[r["pred"]] for r in rows])
    conf = np.array([float(r["prob_max"]) for r in rows])
    return ids, gold, pred, conf


def main_table(P, labels):
    out, cache = [], {}
    for proto, spec in P.items():
        data = {c: load(*v, labels) for c, v in spec["conds"].items()}
        ids0, gold = data["C1"][0], data["C1"][1]
        assert all(d[0] == ids0 for d in data.values()), "prediction files are not aligned"
        preds = {c: d[2] for c, d in data.items()}
        cache[proto] = (ids0, gold, preds, {c: d[3] for c, d in data.items()})
        bf = S.paired_bootstrap(gold, preds, "C1", "macro_f1")
        ba = S.paired_bootstrap(gold, preds, "C1", "accuracy")
        mc = {c: S.mcnemar_exact(preds[c] == gold, preds["C1"] == gold) for c in preds if c != "C1"}
        holm = {}
        for fam in (("C2", "C3", "C4"), ("D2", "D3", "D4")):
            present = [c for c in fam if c in mc]
            for c, p in zip(present, S.holm([mc[c][2] for c in present])):
                holm[c] = p
        if "C0" in mc:
            holm["C0"] = mc["C0"][2]
        for c in sorted(preds):
            out.append({"protocol": proto, "model": spec["model"], "condition": c, "condition_name": COND_NAMES[c],
                        "n": len(gold), "accuracy": round(ba[c]["value"], 6),
                        "accuracy_ci_low": round(ba[c]["ci"][0], 6), "accuracy_ci_high": round(ba[c]["ci"][1], 6),
                        "macro_f1": round(bf[c]["value"], 6), "macro_f1_ci_low": round(bf[c]["ci"][0], 6),
                        "macro_f1_ci_high": round(bf[c]["ci"][1], 6),
                        "delta_f1_vs_c1": round(bf[c]["delta"], 6), "delta_f1_ci_low": round(bf[c]["delta_ci"][0], 6),
                        "delta_f1_ci_high": round(bf[c]["delta_ci"][1], 6),
                        "delta_acc_vs_c1": round(ba[c]["delta"], 6), "delta_acc_ci_low": round(ba[c]["delta_ci"][0], 6),
                        "delta_acc_ci_high": round(ba[c]["delta_ci"][1], 6),
                        "mcnemar_b_better_c1": mc[c][1] if c in mc else "", "mcnemar_c_worse_c1": mc[c][0] if c in mc else "",
                        "mcnemar_p": round(mc[c][2], 6) if c in mc else "", "mcnemar_p_holm": round(holm[c], 6) if c in holm else ""})
    return out, cache


def train_counts():
    splits = load_splits()
    recs = read_jsonl(DATA / "banking_records.jsonl")
    return collections.Counter(r["category"] for r in recs if splits.get(r["record_id"]) == "train")


def per_intent(cache, labels):
    id2 = {v: k for k, v in labels.items()}
    tc = train_counts()
    order = sorted(tc, key=lambda k: (tc[k], k))
    third = {lab: ("low" if i < 26 else "mid" if i < 52 else "high") for i, lab in enumerate(order)}
    rows, terc = [], []
    for proto, (ids, gold, preds, _) in cache.items():
        for c, p in preds.items():
            for k in range(77):
                m = gold == k
                n, hit = int(m.sum()), int((p[m] == k).sum())
                lo, hi = S.wilson(hit, n)
                rows.append({"protocol": proto, "condition": c, "intent": id2[k], "train_count": tc[id2[k]],
                             "tercile": third[id2[k]], "n": n, "recall": round(hit / n, 4),
                             "wilson_low": round(lo, 4), "wilson_high": round(hi, 4)})
            for t in ("low", "mid", "high"):
                m = np.array([third[id2[g]] == t for g in gold])
                terc.append({"protocol": proto, "condition": c, "tercile": t, "n": int(m.sum()),
                             "accuracy": round(float((p[m] == gold[m]).mean()), 4),
                             "train_count_range": f"{min(tc[l] for l in third if third[l] == t)}-"
                                                  f"{max(tc[l] for l in third if third[l] == t)}"})
    small = order[:10]
    return rows, terc, small, third


def slices(cache, labels):
    inj = load_injections()
    cond = {(r["record_id"], r["condition"]): r for r in read_jsonl(DATA_M3 / "conditions.jsonl.gz")
            if r["split"] == "test"}
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    eng = {"V1": "lg", "V0": "lg", "V2": "trf"}.get(sel, "lg")
    from m3.detector import metrics as M
    clean = {r["id"]: r for r in read_jsonl(DATA_M3 / f"detector_spans_{eng}_clean.jsonl.gz")}
    over = {rid: M.over_redacted(clean[rid]["spans"]) for rid in clean}
    rows = []
    for proto, (ids, gold, preds, _) in cache.items():
        keys = {"placement": [inj[i]["template_id"] for i in ids], "pii_type": [inj[i]["pii_type"] for i in ids],
                "clean_query_over_redacted": [over[i] for i in ids]}
        for c, p in preds.items():
            corr = p == gold
            leak = [cond[(i, c)]["residual_leak"] for i in ids] if c.startswith("D") else None
            for name, vals in list(keys.items()) + ([("residual_leak", leak)] if leak else []):
                for v in sorted(set(vals), key=str):
                    m = np.array([x == v for x in vals])
                    rows.append({"protocol": proto, "condition": c, "slice": name, "value": str(v), "n": int(m.sum()),
                                 "accuracy": round(float(corr[m].mean()), 4),
                                 "macro_f1": round(S.macro_f1(gold[m], p[m]), 4)})
    return rows


def confusions(cache, labels, conds):
    id2 = {v: k for k, v in labels.items()}
    top = []
    proto = "frozen"
    ids, gold, preds, _ = cache[proto]
    for c in conds:
        if c not in preds:
            continue
        cm = np.zeros((77, 77), dtype=int)
        np.add.at(cm, (gold, preds[c]), 1)
        with open(OUT / f"confusion_{proto}_{c}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["gold\\pred"] + [id2[k] for k in range(77)])
            for k in range(77):
                w.writerow([id2[k]] + cm[k].tolist())
        off = [(cm[i, j], id2[i], id2[j]) for i in range(77) for j in range(77) if i != j and cm[i, j] > 0]
        for n, g, p in sorted(off, reverse=True)[:10]:
            top.append({"protocol": proto, "condition": c, "gold": g, "pred": p, "count": int(n)})
    return top


def seeds(labels):
    rows = []
    for d in sorted(RUNS.glob("e2d_*")) + [RUNS / "e2a_frozen_c1_s553"] + sorted(RUNS.glob("e2b_frozen_c*_s553")):
        if not (d / "metrics.json").exists():
            continue
        m = read_json(d / "metrics.json")
        man = read_json(d / "manifest.json")
        train = man["config"]["train"]
        if train not in m["test"]:
            continue
        rows.append({"run": d.name, "train_condition": train, "seed": man["seed"],
                     "test_macro_f1": m["test"][train]["macro_f1"], "test_accuracy": m["test"][train]["accuracy"]})
    summary = []
    for c in sorted({r["train_condition"] for r in rows}):
        v = [r["test_macro_f1"] for r in rows if r["train_condition"] == c]
        if len(v) >= 3:
            summary.append({"train_condition": c, "n_seeds": len(v), "mean_macro_f1": round(float(np.mean(v)), 6),
                            "std_macro_f1": round(float(np.std(v, ddof=1)), 6)})
    return rows, summary


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}" if p < 0.01 else f"= {p:.2f}"


def pp(x):
    return f"{x * 100:+.1f} pp"


def findings(table, terc, seed_summary, best):
    t = {(r["protocol"], r["condition"]): r for r in table}
    lines = ["# Findings (test set, n = 3,080; 95% CIs from 2,000 paired bootstrap resamples; McNemar exact, Holm-adjusted)", ""]
    for proto in ("frozen", "retrained", "frozen_tfidf", "retrained_tfidf"):
        if (proto, "C1") not in t:
            continue
        base = t[(proto, "C1")]
        lines.append(f"## {proto} ({base['model']}): C1 raw Macro-F1 {base['macro_f1']:.3f} "
                     f"(95% CI {base['macro_f1_ci_low']:.3f} to {base['macro_f1_ci_high']:.3f})")
        for c in ("C0", "C2", "C3", "C4", "D2", "D3", "D4"):
            r = t.get((proto, c))
            if not r:
                continue
            sig = r["mcnemar_p_holm"] != "" and float(r["mcnemar_p_holm"]) < 0.05
            lines.append(f"- {COND_NAMES[c]}: Macro-F1 {r['macro_f1']:.3f}, {pp(r['delta_f1_vs_c1'])} vs raw "
                         f"(95% CI {pp(r['delta_f1_ci_low'])} to {pp(r['delta_f1_ci_high'])}); accuracy "
                         f"{pp(r['delta_acc_vs_c1'])}; McNemar p {fmt_p(float(r['mcnemar_p_holm']))}"
                         f"{' (Holm)' if c != 'C0' else ''}: "
                         f"{'a detectable difference' if sig else 'not a detectable difference'} at n = {r['n']:,}.")
        top = max((c for c in ("C2", "C3", "C4") if (proto, c) in t), key=lambda c: t[(proto, c)]["macro_f1"], default=None)
        if top:
            lines.append(f"- Highest test Macro-F1 among methods (descriptive): {METHODS[top]} "
                         f"({pp(t[(proto, top)]['delta_f1_vs_c1'])} vs raw). Method chosen on validation: {METHODS[best]}.")
        if best and (proto, best) in t:
            dk = "D" + best[1]
            if (proto, dk) in t:
                lines.append(f"- Cost of detector errors for {METHODS[best]}: "
                             f"{(t[(proto, dk)]['macro_f1'] - t[(proto, best)]['macro_f1']) * 100:+.2f} pp Macro-F1 (D vs C).")
        lines.append("")
    if seed_summary:
        lines.append("## Seeds (553, 554, 555)")
        for s in seed_summary:
            lines.append(f"- {s['train_condition']}: mean Macro-F1 {s['mean_macro_f1']:.3f} +/- {s['std_macro_f1']:.3f} (sd).")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = read_json(DATA_M3 / "label_map.json")
    P = protocols()
    table, cache = main_table(P, labels)
    write_csv(OUT / "main_table.csv", table)
    pi, terc, small, third = per_intent(cache, labels)
    write_csv(OUT / "per_intent_recall.csv", pi)
    write_csv(OUT / "terciles.csv", terc)
    write_csv(OUT / "small_intents.csv", [r for r in pi if r["intent"] in small])
    write_csv(OUT / "slices.csv", slices(cache, labels))
    tbl = {(r["protocol"], r["condition"]): r for r in table}
    from m3.run_experiments import best_sanitized
    best = best_sanitized()  # chosen on validation Macro-F1 of the matched runs, never on test
    write_csv(OUT / "top_confusions.csv", confusions(cache, labels, ("C1", best, "D" + best[1])))
    seed_rows, seed_summary = seeds(labels)
    if seed_rows:
        write_csv(OUT / "seeds.csv", seed_rows)
    (OUT / "findings.md").write_text(findings(table, terc, seed_summary, best), encoding="utf-8", newline="\n")
    write_json(OUT / "analysis.json", {"protocols": {k: {"model": v["model"], "conds": v["conds"]} for k, v in P.items()},
                                       "best_method_retrained": best, "smallest_intents": small,
                                       "seed_summary": seed_summary, "n_bootstrap": S.N_BOOT, "seed": S.SEED})
    print(json.dumps({k: len(v["conds"]) for k, v in P.items()}), "best:", best)


if __name__ == "__main__":
    main()
