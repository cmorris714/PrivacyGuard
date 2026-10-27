"""Assemble one pipeline output record from saved results (no new model calls).

Usage: python -m m3.analysis.example_record
Picks one test record (seed 553) and combines the selected detector's spans, the masked text of the
validation-chosen method, and the matched classifier's saved prediction into the output format.
Writes results/m3/analysis/output_record_example.json.
"""
import random

import yaml

from m3.common import DATA_M3, RESULTS, ROOT, SEED, load_injections, read_json, read_jsonl, write_json
from m3.detector import metrics as M
from m3.models.evaluation import read_predictions


def main():
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    eng = {"V1": "lg", "V0": "lg", "V2": "trf"}[sel]
    best = read_json(RESULTS / "analysis" / "analysis.json")["best_method_retrained"]
    dk = "D" + best[1]
    method = {"C2": "delete", "C3": "generic", "C4": "typed"}[best]
    run = f"e2b_frozen_{best.lower()}_s553"
    tau = read_json(RESULTS / "analysis" / "risk_coverage.json")["tau"]
    preds = {r["record_id"]: r for r in read_predictions(RESULTS / "runs" / run / f"predictions_test_{dk}.csv.gz")}
    cond = {r["record_id"]: r for r in read_jsonl(DATA_M3 / "conditions.jsonl.gz") if r["condition"] == dk and r["split"] == "test"}
    spans = {r["id"]: r for r in read_jsonl(DATA_M3 / f"detector_spans_{eng}_ledger.jsonl.gz")}
    inj = load_injections()
    ids = sorted(i for i in preds if inj[i]["pii_type"] == "EMAIL" and preds[i]["gold"] == preds[i]["pred"])
    rid = random.Random(SEED).choice(ids)
    p, raw = preds[rid], inj[rid]["raw_text"]
    ents = [{"type": M.ENTITY_TO_TYPE[t], "start": s, "end": e}
            for s, e, t in M.merge_spans(M.filter_target(spans[rid]["spans"]))]
    fz = yaml.safe_load(open(ROOT / "configs" / "frozen_model.yaml", encoding="utf-8"))
    rec = {"record_id": rid, "sanitized_text": cond[rid]["text"], "method": method, "entities": ents,
           "residual_pii_check": "FAIL" if cond[rid]["residual_leak"] else "PASS",
           "predicted_intent": p["pred"], "confidence": round(float(p["prob_max"]), 2),
           "top3": [p["top1"], p["top2"], p["top3"]],
           "route": "auto" if float(p["prob_max"]) >= tau else "human_review",
           "model_version": f"{fz['source_run'].split('_c1')[0]}+retrained_{best}", "detector_version": f"{sel} (Presidio 2.2.364)"}
    write_json(RESULTS / "analysis" / "output_record_example.json", rec)
    print(rec)


if __name__ == "__main__":
    main()
