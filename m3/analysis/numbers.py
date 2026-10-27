"""Collect every number shown in the M3 deck into results/m3/deck_numbers.json.

Usage: python -m m3.analysis.numbers
Each key: {"value": raw value, "display": string printed on the slide, "source": file + key, "note": definition}.
M2 facts are read from data/*.json (never typed); M3 numbers from results/m3.
Conventions: Macro-F1 with 3 decimals; rates and shares as percentages with 1 decimal; differences in
percentage points (pp) with sign and 1 decimal; counts with thousands separators.
"""
import csv
import hashlib
import json

import yaml

from m3.common import DATA, DATA_M3, RESULTS, ROOT, norm, read_json, read_jsonl, write_json

N = {}
AN = RESULTS / "analysis"
DET = RESULTS / "detector"


def put(key, value, display, source, note=""):
    assert key not in N, f"duplicate key {key}"
    N[key] = {"value": value, "display": display, "source": source, "note": note}


def cnt(key, v, src, note=""):
    put(key, v, f"{v:,}", src, note)


def pct(key, v, src, note="", nd=1):
    put(key, v, f"{v * 100:.{nd}f}%", src, note)


def f1(key, v, src, note=""):
    put(key, v, f"{v:.3f}", src, note)


def dpp(key, v, src, note=""):
    s = f"{v * 100:+.1f}"
    put(key, v, ("+0.0" if s in ("-0.0", "+0.0") else s) + " pp", src, note)


def ms(key, v, src, note=""):
    put(key, v, f"{v:.1f} ms" if v >= 1 else f"{v:.2f} ms", src, note)


def m2_facts():
    ev = read_json(DATA / "evidence_summary.json")
    prof = read_json(DATA / "profile.json")
    man = read_json(DATA / "source_manifest.json")
    s = "data/evidence_summary.json"
    for k in ("source_rows", "train_rows", "test_rows", "injection_rows", "variant_rows", "sanitized_rows",
              "eligible_train_rows", "modeling_version_rows", "human_review_records"):
        cnt(f"m2_{k}", ev[k], f"{s}:{k}")
    cnt("m2_intents", ev["intent_classes"], f"{s}:intent_classes")
    for t, v in prof["type_counts"].items():
        cnt(f"m2_type_{t}", v, f"data/profile.json:type_counts.{t}")
    put("faker_version", prof["faker_version"], prof["faker_version"], "data/profile.json:faker_version")
    for sp in ("train", "test"):
        cnt(f"b77_{sp}_bytes", man["banking77"][sp]["bytes"], f"data/source_manifest.json:banking77.{sp}.bytes")
        put(f"b77_{sp}_sha_prefix", man["banking77"][sp]["sha256"][:12], man["banking77"][sp]["sha256"][:12],
            f"data/source_manifest.json:banking77.{sp}.sha256", "first 12 hex characters")
    # full hash recomputed from the record text (M2 feedback C)
    rec = next(r for r in read_jsonl(DATA / "banking_records.jsonl") if r["record_id"] == "B77_train_00000")
    h = hashlib.sha256(norm(rec["text"]).casefold().encode("utf-8")).hexdigest()
    assert h == rec["text_sha256"], "recomputed hash differs from banking_records.jsonl"
    put("hash_example_id", rec["record_id"], rec["record_id"], "data/banking_records.jsonl")
    put("hash_example_text", rec["text"], rec["text"], "data/banking_records.jsonl")
    put("hash_example_sha256", h, h, "recomputed: sha256(casefold(NFC + whitespace-normalized text))")
    cnt("excluded_rows", len(prof["exclusion_reasons"]) and sum(prof["exclusion_reasons"].values()),
        "data/profile.json:exclusion_reasons")


def splits_and_data():
    s = read_json(DATA_M3 / "splits.json")
    for k, v in s["counts"].items():
        cnt(f"n_{k}", v, f"data/m3/splits.json:counts.{k}")
    for sp in ("train", "val", "test"):
        b = s["balance"][sp]
        for t, v in b["pii_type"].items():
            pct(f"share_{sp}_{t}", v / b["n"], f"data/m3/splits.json:balance.{sp}.pii_type.{t}")
        pct(f"share_{sp}_prefix", b["placement"]["prefix"] / b["n"], f"data/m3/splits.json:balance.{sp}.placement")
        cnt(f"{sp}_min_per_intent", b["min_per_intent"], f"data/m3/splits.json:balance.{sp}.min_per_intent")
        cnt(f"{sp}_max_per_intent", b["max_per_intent"], f"data/m3/splits.json:balance.{sp}.max_per_intent")
    ml = read_json(RESULTS / "data" / "max_len.json")
    cnt("max_len", ml["distilbert-base-uncased"]["max_len"], "results/m3/data/max_len.json")
    cnt("test_truncated", ml["distilbert-base-uncased"]["test_inputs_truncated_all_conditions"], "results/m3/data/max_len.json")
    tl = list(csv.DictReader(open(RESULTS / "data" / "token_lengths.csv", encoding="utf-8")))
    mx = max(float(r["p99_9"]) for r in tl if r["tokenizer"] == "distilbert-base-uncased" and r["part"] == "train+val")
    put("tokens_p999_max", mx, f"{mx:.0f}", "results/m3/data/token_lengths.csv", "largest p99.9 token count over conditions")
    over = max(int(r["over_96"]) for r in tl if r["tokenizer"] == "distilbert-base-uncased" and r["part"] == "train+val")
    cnt("trainval_over_maxlen", over, "results/m3/data/token_lengths.csv:over_96",
        "most train+val inputs above max_len in any one condition")
    tmax = max(int(r["max"]) for r in tl if r["tokenizer"] == "distilbert-base-uncased")
    cnt("tokens_max", tmax, "results/m3/data/token_lengths.csv", "longest input in tokens, any condition/split")
    ch = read_jsonl(DATA_M3 / "challenge_set.jsonl")
    cnt("challenge_n", len(ch), "data/m3/challenge_set.jsonl")
    cnt("challenge_pos", sum(1 for x in ch if x["spans"]), "data/m3/challenge_set.jsonl")
    cnt("challenge_neg", sum(1 for x in ch if not x["spans"]), "data/m3/challenge_set.jsonl")
    cnt("challenge_formats", len({x["format_tag"] for x in ch if x["spans"]}), "data/m3/challenge_set.jsonl",
        "positive format groups")
    cnt("challenge_neg_groups", len({x["format_tag"] for x in ch if not x["spans"]}), "data/m3/challenge_set.jsonl")
    for sub in ("challenge_dev", "challenge_heldout"):
        cnt(f"n_{sub}", sum(1 for x in ch if x["subset"] == sub), "data/m3/challenge_set.jsonl:subset")
    for t in ("PERSON", "EMAIL", "PHONE", "ADDRESS"):
        cnt(f"challenge_{t}", sum(1 for x in ch if x["category"] == t), "data/m3/challenge_set.jsonl:category")
    cond = read_jsonl(DATA_M3 / "conditions.jsonl.gz")
    cnt("n_condition_rows", len(cond), "data/m3/conditions.jsonl.gz")
    cnt("n_condition_rows_modeled", sum(1 for r in cond if r["split"] != "excluded"), "data/m3/conditions.jsonl.gz")
    rv = list(csv.DictReader(open(DATA_M3 / "detector_review_24.csv", encoding="utf-8")))
    cnt("review24_pass", sum(1 for r in rv if r["review_status"].startswith("PASS")), "data/m3/detector_review_24.csv")


def detector():
    sel = read_json(DET / "selection.json")["selected"]
    put("detector_variant", sel, sel, "results/m3/detector/selection.json")
    rep = read_json(DET / "m2_reproduction.json")["result"]
    cnt("m2_found_total", rep["ALL"]["found"], "results/m3/detector/m2_reproduction.json:result.ALL.found")
    cnt("person_found", rep["PERSON"]["found"], "results/m3/detector/m2_reproduction.json")
    cnt("person_full", rep["PERSON"]["full_span"], "results/m3/detector/m2_reproduction.json")
    cnt("person_left_m2", rep["PERSON"]["residual_leak"], "results/m3/detector/m2_reproduction.json")
    cnt("person_partial_90", rep["PERSON"]["found"] - rep["PERSON"]["full_span"], "results/m3/detector/m2_reproduction.json",
        "found but only partly covered")
    p44 = list(csv.DictReader(open(DET / "person_44_breakdown.csv", encoding="utf-8")))
    cnt("person_missed", len(p44), "results/m3/detector/person_44_breakdown.csv")
    left = sum(r["left_after_all_span_mask"] == "True" for r in p44)
    cnt("person_missed_left", left, "results/m3/detector/person_44_breakdown.csv")
    cnt("person_missed_other_type", len(p44) - left, "results/m3/detector/person_44_breakdown.csv")
    for v in ("V0", "V1", "V2", "V3"):
        d = read_json(DET / f"{v}.json")
        pct(f"{v}_sel_recall", d["ledger"]["ALL"]["detection_recall"], f"results/m3/detector/{v}.json:ledger.ALL")
        pct(f"{v}_sel_full", d["ledger"]["ALL"]["full_span_recall"], f"results/m3/detector/{v}.json:ledger.ALL")
        pct(f"{v}_sel_leak", d["ledger"]["ALL"]["residual_leak_rate"], f"results/m3/detector/{v}.json:ledger.ALL", nd=2)
        cnt(f"{v}_sel_leaks", d["ledger"]["ALL"]["residual_leak"], f"results/m3/detector/{v}.json:ledger.ALL")
        pct(f"{v}_sel_over", d["over_redaction"]["over_redaction_rate"], f"results/m3/detector/{v}.json:over_redaction")
        pct(f"{v}_dev_recall", d["challenge_dev"]["by_type"]["ALL"]["detection_recall"], f"results/m3/detector/{v}.json:challenge_dev")
        ms(f"{v}_median_ms", d["latency"]["median_ms"], f"results/m3/detector/{v}.json:latency")
    cnt("n_trainval_ledger", read_json(DET / "V1.json")["ledger"]["ALL"]["n"], "results/m3/detector/V1.json")
    for v in ("V1", sel):
        d = read_json(DET / f"final_{v}.json")
        t, a = d["ledger_test"], d["ledger_all_13083"]
        src = f"results/m3/detector/final_{v}.json"
        pct(f"{v}_test_recall", t["ALL"]["detection_recall"], src + ":ledger_test.ALL")
        pct(f"{v}_test_full", t["ALL"]["full_span_recall"], src + ":ledger_test.ALL")
        cnt(f"{v}_test_leaks", t["ALL"]["residual_leak"], src + ":ledger_test.ALL")
        pct(f"{v}_test_leak_rate", t["ALL"]["residual_leak_rate"], src + ":ledger_test.ALL", nd=2)
        cnt(f"{v}_test_partial", t["ALL"]["partial_leak"], src + ":ledger_test.ALL")
        for ty in ("EMAIL", "PHONE", "PERSON", "ADDRESS"):
            cnt(f"{v}_test_leaks_{ty}", t[ty]["residual_leak"], src + f":ledger_test.{ty}")
            pct(f"{v}_test_recall_{ty}", t[ty]["detection_recall"], src + f":ledger_test.{ty}")
        cnt(f"{v}_all_leaks", a["ALL"]["residual_leak"], src + ":ledger_all_13083.ALL")
        pct(f"{v}_all_recall", a["ALL"]["detection_recall"], src + ":ledger_all_13083.ALL")
        pct(f"{v}_test_over", d["over_redaction_test"]["over_redaction_rate"], src + ":over_redaction_test")
        h = d["challenge_heldout"]
        pct(f"{v}_held_recall", h["by_type"]["ALL"]["detection_recall"], src + ":challenge_heldout.by_type.ALL")
        pct(f"{v}_held_leak", h["by_type"]["ALL"]["residual_leak_rate"], src + ":challenge_heldout.by_type.ALL")
        pct(f"{v}_held_neg_fp", h["negatives"]["false_positive_rate"], src + ":challenge_heldout.negatives")
        for ty in ("PERSON", "EMAIL", "PHONE", "ADDRESS"):
            pct(f"{v}_held_recall_{ty}", h["by_type"][ty]["detection_recall"], src + f":challenge_heldout.by_type.{ty}")
        ms(f"{v}_all_median_ms", d["latency_all"]["median_ms"], src + ":latency_all")
    ov = read_json(DET / "over_redaction_examples.json")[sel]
    put("over_example_entities", sorted(ov), ", ".join(sorted(ov)), "results/m3/detector/over_redaction_examples.json")


def models():
    fz = yaml.safe_load(open(ROOT / "configs" / "frozen_model.yaml", encoding="utf-8"))
    names = {"tfidf_svm": "TF-IDF + LinearSVC", "embed_clf": "MiniLM embeddings + logistic regression",
             "transformer": "DistilBERT (fine-tuned)"}
    put("model_name", names[fz["model"]["family"]], names[fz["model"]["family"]], "configs/frozen_model.yaml:model.family")
    lat = read_json(RESULTS / "latency.json")
    for d in sorted((RESULTS / "runs").glob("e1_*")):
        m = read_json(d / "metrics.json")
        key = {"tfidf_svm": "b1", "embed_clf": "b2", "transformer": "t1"}[read_json(d / "manifest.json")["config"]["model_cfg"]["family"]]
        v = m["val"]["C1"]
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            f1(f"e1_{key}_{k}", v[k], f"results/m3/runs/{d.name}/metrics.json:val.C1.{k}")
        c = lat["classifiers"][d.name]
        ms(f"e1_{key}_ms", c["batch1"]["median_ms"], "results/m3/latency.json:classifiers." + d.name)
        ms(f"e1_{key}_p95", c["batch1"]["p95_ms"], "results/m3/latency.json:classifiers." + d.name)
        put(f"e1_{key}_params", c["n_parameters"], f"{c['n_parameters'] / 1e6:.1f}M", "results/m3/latency.json:n_parameters")
        cnt(f"e1_{key}_train_s", round(m["train_seconds"]), f"results/m3/runs/{d.name}/metrics.json:train_seconds")
        put(f"e1_{key}_selected", m["selected_params"], json.dumps(m["selected_params"]), f"results/m3/runs/{d.name}/metrics.json")
        if key == "t1":
            curve = list(csv.DictReader(open(d / "train_curve.csv", encoding="utf-8")))
            cnt("t1_epochs", len(curve), f"results/m3/runs/{d.name}/train_curve.csv")
            f1("t1_epoch1_f1", float(curve[0]["val_macro_f1"]), f"results/m3/runs/{d.name}/train_curve.csv")
            dpp("t1_last_epoch_gain", float(curve[-1]["val_macro_f1"]) - float(curve[-2]["val_macro_f1"]),
                f"results/m3/runs/{d.name}/train_curve.csv", "val Macro-F1 gain in the last epoch")
    for k, other in (("gap_b2_b1", "e1_b1_macro_f1"), ("gap_b2_t1", "e1_t1_macro_f1")):
        g = round(N["e1_b2_macro_f1"]["value"], 3) - round(N[other]["value"], 3)  # matches the 3-decimal values shown
        put(k, g, f"{abs(g) * 100:.1f} pp", f"derived: e1_b2_macro_f1 - {other}", "unsigned gap in validation Macro-F1")
    put("speedup_b1_vs_b2", N["e1_b2_ms"]["value"] / N["e1_b1_ms"]["value"],
        f"{N['e1_b2_ms']['value'] / N['e1_b1_ms']['value']:.0f}x", "derived: median ms b2 / b1")
    put("frozen_C", fz["model"].get("params", {}).get("C"), str(fz["model"].get("params", {}).get("C")), "configs/frozen_model.yaml")
    put("b1_params_C", fz["cross_check_model"]["params"]["C"], str(fz["cross_check_model"]["params"]["C"]), "configs/frozen_model.yaml")
    walls = {}
    for d in (RESULTS / "runs").glob("*/manifest.json"):
        man = read_json(d)
        st = {"e2a": "e2", "e2b": "e2", "e2c": "e4", "e2d": "e4", "e3c": "e3"}.get(man["stage"], man["stage"])
        walls[st] = walls.get(st, 0) + man["wall_seconds"]
    for st, w in sorted(walls.items()):
        put(f"wall_{st}", w, f"{w / 60:.1f} min", "results/m3/runs/*/manifest.json:wall_seconds", "summed wall time per stage")
    log = list(csv.DictReader(open(RESULTS / "test_eval_log.csv", encoding="utf-8")))
    cnt("n_test_evals", sum(1 for r in log if r["run_id"] != "FROZEN_CONFIG_REGISTERED"), "results/m3/test_eval_log.csv")
    cnt("n_runs", len(list((RESULTS / "runs").glob("*/metrics.json"))), "results/m3/runs/*/metrics.json")
    # pipeline latency
    src = fz["source_run"]
    p = lat["pipeline"][src]
    ms("e2e_median_ms", p["median_ms"], "results/m3/latency.json:pipeline")
    ms("e2e_p95_ms", p["p95_ms"], "results/m3/latency.json:pipeline")
    cnt("e2e_records_per_hour", p["records_per_hour_batch1"], "results/m3/latency.json:pipeline")
    ms("det_median_ms_lat", lat["detector"]["batch1"]["median_ms"], "results/m3/latency.json:detector.batch1")
    ms("det_p95_ms_lat", lat["detector"]["batch1"]["p95_ms"], "results/m3/latency.json:detector.batch1")
    cnt("det_records_per_hour", lat["detector"]["records_per_hour"], "results/m3/latency.json:detector")
    c = lat["classifiers"][src]
    ms("cls_median_ms_lat", c["batch1"]["median_ms"], "results/m3/latency.json:classifiers")
    ms("cls_p95_ms_lat", c["batch1"]["p95_ms"], "results/m3/latency.json:classifiers")
    cnt("cls_records_per_hour_b64", c["records_per_hour_batch64"], "results/m3/latency.json:classifiers")
    put("cpu_name", lat["cpu"], lat["cpu"], "results/m3/latency.json:cpu")


def experiments():
    rows = list(csv.DictReader(open(AN / "main_table.csv", encoding="utf-8")))
    for r in rows:
        k = f"{r['protocol']}_{r['condition']}"
        src = f"results/m3/analysis/main_table.csv:{r['protocol']},{r['condition']}"
        f1(f"{k}_f1", float(r["macro_f1"]), src)
        put(f"{k}_f1_ci", [float(r["macro_f1_ci_low"]), float(r["macro_f1_ci_high"])],
            f"{float(r['macro_f1_ci_low']):.3f}-{float(r['macro_f1_ci_high']):.3f}", src)
        pct(f"{k}_acc", float(r["accuracy"]), src)
        if r["condition"] != "C1":
            dpp(f"{k}_d", float(r["delta_f1_vs_c1"]), src, "Macro-F1 difference vs C1 raw")
            put(f"{k}_d_ci", [float(r["delta_f1_ci_low"]), float(r["delta_f1_ci_high"])],
                f"{float(r['delta_f1_ci_low']) * 100:+.1f} to {float(r['delta_f1_ci_high']) * 100:+.1f}", src)
            p = float(r["mcnemar_p_holm"])
            put(f"{k}_p", p, "< 0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}", src,
                "exact McNemar vs C1 (Holm within method family)")
    an = read_json(AN / "analysis.json")
    best = an["best_method_retrained"]
    names = {"C2": "delete", "C3": "generic", "C4": "typed"}
    put("best_method", names[best], names[best], "results/m3/analysis/analysis.json:best_method_retrained")
    put("best_cond", best, best, "results/m3/analysis/analysis.json")
    bf = min(("C2", "C3", "C4"), key=lambda c: -float(next(r["macro_f1"] for r in rows
                                                         if r["protocol"] == "frozen" and r["condition"] == c)))
    put("best_method_frozen", names[bf], names[bf], "results/m3/analysis/main_table.csv (frozen, highest Macro-F1)")
    worst = min(("C2", "C3", "C4"), key=lambda c: float(next(r["macro_f1"] for r in rows
                                                            if r["protocol"] == "frozen" and r["condition"] == c)))
    put("worst_method_frozen", names[worst], names[worst], "results/m3/analysis/main_table.csv (frozen, lowest Macro-F1)")
    for proto in ("frozen", "retrained"):
        dk = "D" + best[1]
        a, b = f"{proto}_{dk}", f"{proto}_{best}"
        if a + "_f1" in N:
            dpp(f"{proto}_detector_cost", N[a + "_f1"]["value"] - N[b + "_f1"]["value"],
                "derived: Macro-F1 D_k - C_k (" + proto + ")", "cost of detector errors")
    # bootstrap CI half-width for accuracy at n = 3,080 (raw, frozen)
    lo, hi = N["frozen_C1_f1_ci"]["value"]
    r = next(x for x in rows if x["protocol"] == "frozen" and x["condition"] == "C1")
    put("ci_halfwidth_acc", (float(r["accuracy_ci_high"]) - float(r["accuracy_ci_low"])) / 2,
        f"{(float(r['accuracy_ci_high']) - float(r['accuracy_ci_low'])) / 2 * 100:.1f}", "results/m3/analysis/main_table.csv",
        "half-width of the 95% bootstrap CI for accuracy, pp")
    from m3.analysis.stats import wilson
    w = wilson(36, 40)
    put("wilson_halfwidth_90", (w[1] - w[0]) / 2, f"{(w[1] - w[0]) / 2 * 100:.0f}", "m3/analysis/stats.py:wilson(36, 40)",
        "Wilson 95% half-width at 90% recall with n = 40, pp")
    for s in an.get("seed_summary", []):
        put(f"seed_sd_{s['train_condition']}", s["std_macro_f1"], f"{s['std_macro_f1']:.3f}", "results/m3/analysis/analysis.json")
    # per-intent and terciles
    terc = list(csv.DictReader(open(AN / "terciles.csv", encoding="utf-8")))
    for t in terc:
        if t["protocol"] in ("frozen", "retrained") and t["condition"] in ("C1", best):
            pct(f"terc_{t['protocol']}_{t['condition']}_{t['tercile']}", float(t["accuracy"]), "results/m3/analysis/terciles.csv")
            if f"terc_range_{t['tercile']}" not in N:
                put(f"terc_range_{t['tercile']}", t["train_count_range"], t["train_count_range"], "results/m3/analysis/terciles.csv")
    sm = list(csv.DictReader(open(AN / "small_intents.csv", encoding="utf-8")))
    low = [r for r in sm if r["protocol"] == "frozen" and r["condition"] == "C1"]
    worst_small = min(low, key=lambda r: float(r["recall"]))
    put("small_worst_intent", worst_small["intent"], worst_small["intent"].replace("_", " "), "results/m3/analysis/small_intents.csv")
    pct("small_worst_recall", float(worst_small["recall"]), "results/m3/analysis/small_intents.csv")
    # slices: placement
    sl = list(csv.DictReader(open(AN / "slices.csv", encoding="utf-8")))
    for r in sl:
        if r["slice"] == "placement" and r["protocol"] in ("frozen", "retrained"):
            pct(f"place_{r['protocol']}_{r['condition']}_{r['value']}", float(r["accuracy"]), "results/m3/analysis/slices.csv")
    for proto in ("frozen", "retrained"):
        a = N.get(f"place_{proto}_C1_prefix")
        b = N.get(f"place_{proto}_C1_suffix")
        if a and b:
            dpp(f"place_gap_{proto}", a["value"] - b["value"], "derived: accuracy prefix - suffix on C1")
    over = [r for r in sl if r["slice"] == "clean_query_over_redacted" and r["protocol"] == "frozen" and r["condition"] == "D" + best[1]]
    for r in over:
        pct(f"over_slice_acc_{r['value']}", float(r["accuracy"]), "results/m3/analysis/slices.csv")
        cnt(f"over_slice_n_{r['value']}", int(r["n"]), "results/m3/analysis/slices.csv")
    # risk-coverage
    rc = read_json(AN / "risk_coverage.json")
    put("tau", rc["tau"], f"{rc['tau']:.2f}", "results/m3/analysis/risk_coverage.json:tau")
    for c in ("C1", best, "D" + best[1]):
        t = rc["test"].get(c)
        if t:
            pct(f"rc_cov_{c}", t["coverage"], "results/m3/analysis/risk_coverage.json:test." + c)
            pct(f"rc_prec_{c}", t["auto_precision"], "results/m3/analysis/risk_coverage.json:test." + c)
            pct(f"rc_queue_{c}", t["review_queue_share"], "results/m3/analysis/risk_coverage.json:test." + c)
    # business
    bz = read_json(AN / "business.json")
    for k, v in bz["misroutes_per_1000"].items():
        put(f"mis_{k}", v["value"], f"{v['value']:.0f}", "results/m3/analysis/business.json:misroutes_per_1000." + k)
        put(f"mis_add_{k}", v["added_vs_raw"], f"{v['added_vs_raw']:+.0f}", "results/m3/analysis/business.json")
    sel = N["detector_variant"]["value"]
    for v in ("V1", sel):
        put(f"leaks10k_{v}", bz["leaks_per_10000"][v], f"{bz['leaks_per_10000'][v]:.0f}", "results/m3/analysis/business.json")
        put(f"over1k_{v}", bz["over_redactions_per_1000_clean"][v], f"{bz['over_redactions_per_1000_clean'][v]:.1f}",
            "results/m3/analysis/business.json")
    put("hours_per_million", bz["throughput"]["hours_per_million_records"], f"{bz['throughput']['hours_per_million_records']:.1f}",
        "results/m3/analysis/business.json:throughput")
    put("mean_bytes", bz["mean_utf8_bytes_per_record"], f"{bz['mean_utf8_bytes_per_record']:.0f}", "results/m3/analysis/business.json")
    msc = bz["managed_service_comparison"]
    if "estimated_usd" in msc:
        put("managed_usd_1m", msc["estimated_usd"], f"${msc['estimated_usd']:,.2f}", "results/m3/analysis/business.json")
        put("managed_usd_per_gb", msc["usd_per_gb"], f"${msc['usd_per_gb']:,.2f}", "results/m3/analysis/business.json")
        put("managed_gb_1m", msc["volume_gb"], f"{msc['volume_gb']:.2f}", "results/m3/analysis/business.json")
    # error patterns
    ep = list(csv.DictReader(open(AN / "error_patterns.csv", encoding="utf-8")))
    for i, p in enumerate(ep, 1):
        cnt(f"ep{i}_count", int(p["count"]), "results/m3/analysis/error_patterns.csv")
        pct(f"ep{i}_share", float(p["share"]), "results/m3/analysis/error_patterns.csv")
        cnt(f"ep{i}_pop", int(p["population"]), "results/m3/analysis/error_patterns.csv")


def examples():
    """Values shown in example rows and the output-record excerpt (read from the files, not typed)."""
    ex = read_json(AN / "output_record_example.json")
    for k in ("record_id", "sanitized_text", "method", "residual_pii_check", "predicted_intent", "route"):
        put(f"ex_{k}", ex[k], str(ex[k]), "results/m3/analysis/output_record_example.json:" + k)
    put("ex_text_short", ex["sanitized_text"], ex["sanitized_text"][:22] + "...",
        "results/m3/analysis/output_record_example.json:sanitized_text", "first 22 characters")
    put("ex_confidence", ex["confidence"], f"{ex['confidence']:.2f}", "results/m3/analysis/output_record_example.json")
    e = ex["entities"][0]
    put("ex_entity", e, f'{{"type": "{e["type"]}", "start": {e["start"]}, "end": {e["end"]}}}',
        "results/m3/analysis/output_record_example.json:entities[0]")
    for k in ("type", "start", "end"):
        put(f"ex_entity_{k}", e[k], str(e[k]), "results/m3/analysis/output_record_example.json:entities[0]." + k)
    put("ex_top2", ex["top3"][1], ex["top3"][1], "results/m3/analysis/output_record_example.json:top3")
    sr = read_json(DATA / "evidence_summary.json")["sample_record"]["injection"]
    for k in ("record_id", "pii_type", "pii_value", "start_char", "end_char", "template_id", "seed"):
        put(f"m2ex_{k}", sr[k], f"{sr[k]}", "data/evidence_summary.json:sample_record.injection." + k)
    rec = next(r for r in read_jsonl(DATA / "banking_records.jsonl") if r["record_id"] == "B77_train_00000")
    for k in ("category", "split", "source_row", "source_file", "acquired_date", "char_count"):
        put(f"b77ex_{k}", rec[k], str(rec[k]), "data/banking_records.jsonl:B77_train_00000." + k)
    ch = next(x for x in read_jsonl(DATA_M3 / "challenge_set.jsonl") if x["id"] == "CH_0106")
    sp = ch["spans"][0]
    for k, v in (("id", ch["id"]), ("subset", ch["subset"]), ("text", ch["text"]), ("type", sp["type"]),
                 ("start", sp["start"]), ("end", sp["end"]), ("value", sp["value"]), ("format_tag", ch["format_tag"]),
                 ("carrier", ch["carrier_record_id"]), ("template", ch["template_id"])):
        put(f"chex_{k}", v, str(v), "data/m3/challenge_set.jsonl:CH_0106")
    put("chex_text_short", ch["text"], ch["text"][:24] + "...", "data/m3/challenge_set.jsonl:CH_0106")


def thresholds():
    t = {}
    t["T1"] = N["e1_b2_macro_f1"]["value"] >= 0.90 if N["model_name"]["value"].startswith("MiniLM") else None
    sel_key = {"MiniLM": "e1_b2_macro_f1", "DistilBERT": "e1_t1_macro_f1", "TF-IDF": "e1_b1_macro_f1"}
    k = next(v for p, v in sel_key.items() if N["model_name"]["value"].startswith(p))
    t["T1"] = N[k]["value"] >= 0.90
    best = N["best_cond"]["value"]
    t["T2"] = N[f"retrained_{best}_d"]["value"] >= -0.010
    bf = {"delete": "C2", "generic": "C3", "typed": "C4"}[N["best_method_frozen"]["value"]]
    t["T3"] = N[f"frozen_{bf}_d"]["value"] >= -0.020
    sel = N["detector_variant"]["value"]
    t["T4"] = (N[f"{sel}_test_leak_rate"]["value"] <= 0.0025 and N[f"{sel}_test_leaks_EMAIL"]["value"] == 0
               and N[f"{sel}_test_leaks_PHONE"]["value"] == 0 and N[f"{sel}_test_recall"]["value"] >= 0.995)
    t["T5"] = N[f"{sel}_held_recall"]["value"] >= 0.90
    t["T6"] = N[f"{sel}_test_over"]["value"] <= 0.05
    t["T7"] = N["e2e_median_ms"]["value"] <= 100
    for k2, v in t.items():
        put(f"thr_{k2}", v, "PASS" if v else "FAIL", "derived from deck_numbers + results/m3/preregistration.md")
    cnt("thr_pass_count", sum(t.values()), "derived")


def main():
    m2_facts()
    splits_and_data()
    detector()
    models()
    experiments()
    examples()
    thresholds()
    N["_conventions"] = {"value": None, "display": "Macro-F1: 3 decimals; rates: % with 1 decimal (leak rates 2); "
                         "differences: signed pp with 1 decimal; counts: thousands separators", "source": "m3/analysis/numbers.py",
                         "note": ""}
    write_json(RESULTS / "deck_numbers.json", N)
    print(len(N), "numbers;", {k: N[k]["display"] for k in N if k.startswith("thr_")})


if __name__ == "__main__":
    main()
