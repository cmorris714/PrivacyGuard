"""Confidence threshold for the human-review queue (risk-coverage analysis).

Usage: python -m m3.analysis.risk
The threshold tau is the smallest max-probability cut-off at which predictions above it reach >= 98%
precision on the C1 validation split (frozen model). It is then applied unchanged to the test conditions.
Writes results/m3/analysis/risk_coverage.json and risk_curve.csv.
"""
import csv

import numpy as np

from m3.analysis.evaluate import OUT, RUNS, load, protocols
from m3.common import DATA_M3, read_json
from m3.models.evaluation import read_predictions

TARGET = 0.98


def curve(conf, correct):
    order = np.argsort(-conf)
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    return conf[order], k / len(c), np.cumsum(c) / k   # threshold, coverage, precision above threshold


def main():
    labels = read_json(DATA_M3 / "label_map.json")
    P = protocols()
    run = P["frozen"]["conds"]["C1"][0]
    val = read_predictions(RUNS / run / "predictions_val_C1.csv.gz")
    conf = np.array([float(r["prob_max"]) for r in val])
    corr = np.array([r["gold"] == r["pred"] for r in val])
    th, cov, prec = curve(conf, corr)
    ok = np.where(prec >= TARGET)[0]
    # largest coverage whose precision stays >= target; tau = confidence at that point
    i = ok.max() if len(ok) else 0
    tau = float(th[i])
    res = {"target_precision": TARGET, "tau": round(tau, 4), "chosen_on": f"{run} validation C1",
           "val_coverage": round(float(cov[i]), 4), "val_precision": round(float(prec[i]), 4), "test": {}}
    rows = []
    for c, spec in P["frozen"]["conds"].items():
        ids, gold, pred, cf = load(*spec, labels)
        auto = cf >= tau
        res["test"][c] = {"coverage": round(float(auto.mean()), 4),
                          "auto_precision": round(float((pred[auto] == gold[auto]).mean()), 4),
                          "review_queue_share": round(float(1 - auto.mean()), 4),
                          "overall_accuracy": round(float((pred == gold).mean()), 4)}
        t, cv, pr = curve(cf, pred == gold)
        for q in np.linspace(0.05, 1.0, 20):
            j = min(len(cv) - 1, int(q * len(cv)) - 1)
            rows.append({"condition": c, "coverage": round(float(cv[j]), 4), "precision": round(float(pr[j]), 4),
                         "threshold": round(float(t[j]), 4)})
    from m3.common import write_json
    # Final pipeline (selected detector + generic masking + classifier retrained on generic text) at the same
    # threshold. Descriptive only: tau is not re-tuned on test.
    run, cond = P["retrained"]["conds"]["D3"]
    ids, gold, pred, cf = load(run, cond, labels)
    auto = cf >= tau
    ok = pred == gold
    res["final_pipeline"] = {"run": run, "condition": cond, "n": int(len(gold)), "n_auto": int(auto.sum()),
                             "n_auto_correct": int((auto & ok).sum()),
                             "coverage": round(float(auto.mean()), 4),
                             "auto_precision": round(float(ok[auto].mean()), 4),
                             "review_queue_share": round(float(1 - auto.mean()), 4),
                             "overall_accuracy": round(float(ok.mean()), 4)}
    write_json(OUT / "risk_coverage.json", res)
    with open(OUT / "risk_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("tau", res["tau"], {c: v["coverage"] for c, v in res["test"].items()})


if __name__ == "__main__":
    main()
