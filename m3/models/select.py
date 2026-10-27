"""Apply the pre-registered model selection rule to the Exp 1 runs and freeze the chosen configuration.

Usage: python -m m3.models.select   (after Exp 1 and `python -m m3.latency --part classifiers`)
Rule: highest validation Macro-F1 on C1; if a faster model (lower p95 ms/query at batch 1) is within
0.5 percentage points, the faster model is chosen.
Writes results/m3/selection.md and configs/frozen_model.yaml, and registers the frozen config's SHA-256
in results/m3/test_eval_log.csv before any test evaluation.
"""
import datetime as dt

import yaml

from m3.common import ARTIFACTS, ROOT, RESULTS, read_json, sha256_file
from m3.run_experiments import TEST_LOG, log_test

NAMES = {"tfidf_svm": "B1 TF-IDF + LinearSVC", "embed_clf": "B2 MiniLM embeddings + classifier",
         "transformer": "T1 DistilBERT (fine-tuned)"}


def main():
    lat = read_json(ARTIFACTS / "latency_parts" / "classifiers.json")["models"]
    rows = []
    for d in sorted((RESULTS / "runs").glob("e1_*")):
        m = read_json(d / "metrics.json")
        man = read_json(d / "manifest.json")
        fam = man["config"]["model_cfg"]["family"]
        rows.append({"run": d.name, "family": fam, "cfg": man["config"]["model_cfg"], "params": m["selected_params"],
                     "val": m["val"]["C1"], "p95": lat[d.name]["batch1"]["p95_ms"],
                     "median": lat[d.name]["batch1"]["median_ms"], "n_par": lat[d.name]["n_parameters"], "train_s": m["train_seconds"]})
    best = max(rows, key=lambda r: r["val"]["macro_f1"])
    close = [r for r in rows if best["val"]["macro_f1"] - r["val"]["macro_f1"] <= 0.005]
    chosen = min(close, key=lambda r: r["p95"])
    b1 = next(r for r in rows if r["family"] == "tfidf_svm")

    model_cfg = dict(chosen["cfg"])
    model_cfg.pop("seed", None)
    if chosen["family"] != "transformer":
        model_cfg["params"] = chosen["params"]
    b1_cfg = {"family": "tfidf_svm", "params": b1["params"]}
    revisions = read_json(RESULTS / "env" / "models.json")
    frozen = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "source_run": chosen["run"], "selection": "results/m3/selection.md",
              "model": model_cfg, "seed": 553,
              "model_revision": revisions.get(model_cfg.get("model_id", model_cfg.get("encoder", "")), {}).get("revision"),
              "cross_check_model": b1_cfg}
    path = ROOT / "configs" / "frozen_model.yaml"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Frozen classifier configuration (selected on the C1 validation split; see results/m3/selection.md)\n")
        yaml.safe_dump(frozen, f, sort_keys=False)
    sha = sha256_file(path)
    if not TEST_LOG.exists() or sha not in TEST_LOG.read_text(encoding="utf-8"):
        log_test("FROZEN_CONFIG_REGISTERED", chosen["run"], sha, [], 0)

    lines = ["# Model selection (Exp 1)", "",
             "All models trained on the C1 raw training split (7,993 queries) and compared on the C1 validation split (1,999). "
             "Latency: CPU, batch 1, 500 validation records (`results/m3/latency.json`).", "",
             "| Model | Parameters | Val accuracy | Val macro-P | Val macro-R | Val Macro-F1 | p95 ms/query | Training s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["val"]
        lines.append(f"| {NAMES[r['family']]} | {r['n_par'] / 1e6:.1f}M | {v['accuracy']:.4f} | {v['macro_precision']:.4f} | "
                     f"{v['macro_recall']:.4f} | {v['macro_f1']:.4f} | {r['p95']:.2f} | {r['train_s']:.0f} |")
    lines += ["", f"Highest validation Macro-F1: {NAMES[best['family']]} ({best['val']['macro_f1']:.4f}).",
              f"Models within 0.5 pp: {', '.join(NAMES[r['family']] for r in close)}.",
              f"**Selected: {NAMES[chosen['family']]}** (`{chosen['run']}`), frozen to `configs/frozen_model.yaml` "
              f"(SHA-256 `{sha}`).", "",
              "Grid choices: " + "; ".join(f"{NAMES[r['family']]}: {r['params']}" for r in rows), "",
              "Considered but not run (scope decision): majority-class baseline, TextCNN, BiLSTM, BERT-base, RoBERTa-base, "
              "ModernBERT-base, zero-shot Qwen2.5 instruction model, a second DistilBERT learning rate (5e-5), all-mpnet-base-v2."]
    open(RESULTS / "selection.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("selected", chosen["run"], sha[:12])


if __name__ == "__main__":
    main()
