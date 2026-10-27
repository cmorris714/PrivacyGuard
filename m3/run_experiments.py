"""Resumable experiment runner for the M3 classifier runs listed in m3/experiments.yaml.

Usage:
  python -m m3.run_experiments --stage e1        # one stage
  python -m m3.run_experiments --run <run_id>    # one run
  python -m m3.run_experiments --all             # everything, in file order
  python -m m3.run_experiments --dry-run --all   # resolve configs and print hashes only
  python -m m3.run_experiments --smoke --all     # 300 training rows, 1 epoch, no test data; output in artifacts/smoke

A run is skipped when results/m3/runs/<run_id>/metrics.json exists with the same config hash.
Per-run outputs: manifest.json, metrics.json, predictions_val_<cond>.csv.gz, predictions_test_<cond>.csv.gz,
train_curve.csv (transformers), grid.json (grid searches). Checkpoints go to artifacts/models/<run_id>/.
Every test-set evaluation is appended to results/m3/test_eval_log.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import time
import traceback

import yaml

from m3.common import ARTIFACTS, ROOT, RESULTS, config_hash, git_commit, read_json, seed_everything, sha256_file, write_json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RUNS = RESULTS / "runs"
FROZEN = ROOT / "configs" / "frozen_model.yaml"
SMOKE_FROZEN = ARTIFACTS / "smoke" / "frozen_model_smoke.yaml"  # used only with --smoke
TEST_LOG = RESULTS / "test_eval_log.csv"
STATUS = ARTIFACTS / "run_status.json"
METHOD = {"C2": "delete", "C3": "generic", "C4": "typed"}


def load_spec():
    return yaml.safe_load(open(ROOT / "m3" / "experiments.yaml", encoding="utf-8"))


def max_len():
    return read_json(RESULTS / "data" / "max_len.json")["distilbert-base-uncased"]["max_len"]


def frozen():
    return yaml.safe_load(open(FROZEN, encoding="utf-8"))


def best_sanitized():
    """Best of C2/C3/C4 by validation Macro-F1 of the matched (2b) runs."""
    scores = {}
    base = ARTIFACTS / "smoke" if FROZEN == SMOKE_FROZEN else RUNS
    for c in METHOD:
        m = base / f"e2b_frozen_{c.lower()}_s553" / "metrics.json"
        if not m.exists():
            raise RuntimeError("e2b runs must finish before C_best/D_best runs")
        scores[c] = read_json(m)["val"][c]["macro_f1"]
    return max(scores, key=scores.get)


def resolve(run, spec):
    """Fill model config, seed and symbolic conditions; returns a fully explicit run config."""
    r = dict(run)
    r["seed"] = run.get("seed", spec["seed"])
    if run["model"] == "frozen":
        fz = frozen()
        model = dict(fz["model"])
        r["frozen_config_sha256"] = sha256_file(FROZEN)
        if run.get("reuse") == "selected_e1_run":
            r["reuse"] = fz["source_run"]
    elif run["model"] == "b1_frozen":
        fz = frozen()
        model = dict(fz["cross_check_model"])
        r["frozen_config_sha256"] = sha256_file(FROZEN)
    else:
        model = dict(spec["models"][run["model"]])
        if model.get("max_len") == "from_token_lengths":
            model["max_len"] = max_len()
    model["seed"] = r["seed"]
    r["model_cfg"] = model
    sym = {}
    if "C_best" in str(run) or "D_best" in str(run):
        cb = best_sanitized()
        sym = {"C_best": cb, "D_best": "D" + cb[1]}
    r["train"] = sym.get(run["train"], run["train"])
    r["test"] = [sym.get(c, c) for c in run.get("test", [])]
    return r


def texts(rows, cond, split, limit=None):
    sel = [x for x in rows if x["condition"] == cond and x["split"] == split]
    sel.sort(key=lambda x: x["record_id"])
    if limit:  # smoke tests: seeded random subset (records are ordered by intent)
        import random
        sel = random.Random(553).sample(sel, min(limit, len(sel)))
    return sel


def log_test(run_id, chash, frozen_sha, conds, n):
    new = not TEST_LOG.exists()
    with open(TEST_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(["timestamp_utc", "run_id", "config_hash", "frozen_config_sha256", "conditions", "n_records"])
        w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), run_id, chash, frozen_sha,
                    " ".join(conds), n])


def versions():
    import numpy
    import sklearn
    import torch
    import transformers
    v = {"python": platform.python_version(), "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
         "torch": torch.__version__, "transformers": transformers.__version__}
    try:
        import sentence_transformers
        v["sentence-transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
    return v


def set_status(run_id, **kw):
    s = read_json(STATUS) if STATUS.exists() else {}
    s.setdefault(run_id, {}).update(kw)
    write_json(STATUS, s)


def execute(r, rows, smoke=False):
    import numpy as np
    import torch
    from m3.models import classifiers
    from m3.models.evaluation import classification_metrics, write_predictions

    labels = read_json(ROOT / "data" / "m3" / "label_map.json")
    id2label = {v: k for k, v in labels.items()}
    out = (ARTIFACTS / "smoke" / r["id"]) if smoke else (RUNS / r["id"])
    ckpt = ARTIFACTS / ("smoke_models" if smoke else "models") / r["id"]
    cfg = dict(r["model_cfg"])
    if smoke:
        cfg["max_epochs"] = 1
    seed_everything(r["seed"])
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    lim = 300 if smoke else None
    tr = texts(rows, r["train"], "train", lim)
    va = texts(rows, r["train"], "val", lim)
    Xtr, ytr = [x["text"] for x in tr], np.array([labels[x["category"]] for x in tr])
    Xva, yva = [x["text"] for x in va], np.array([labels[x["category"]] for x in va])

    model = classifiers.build(cfg)
    t0 = time.perf_counter()
    if r.get("reuse"):
        src = ARTIFACTS / "models" / r["reuse"]
        model.load(str(src))
        if cfg["family"] == "tfidf_svm" and not hasattr(model, "params"):
            model.params = cfg.get("params")
        info = {"reused_checkpoint_of": r["reuse"]}
    else:
        if smoke and cfg["family"] == "tfidf_svm" and cfg.get("params") is None:
            cfg["params"] = {"analyzer": "word", "ngram": [1, 2], "C": 1, "min_df": 1, "class_weight": None}
            model = classifiers.build(cfg)
        info = model.fit(Xtr, ytr, Xva, yva)
        model.save(str(ckpt))
    train_s = time.perf_counter() - t0

    metrics = {"run_id": r["id"], "config_hash": r["hash"], "val": {}, "test": {}}
    pv = model.predict_proba(Xva)
    metrics["val"][r["train"]] = classification_metrics(yva, pv.argmax(1))
    write_predictions(out / f"predictions_val_{r['train']}.csv.gz", [x["record_id"] for x in va], yva, pv, id2label)

    test_conds = r["test"]
    if test_conds and not smoke:
        assert r.get("frozen_config_sha256"), "test evaluation is only allowed for frozen configurations"
        assert r["frozen_config_sha256"] == sha256_file(FROZEN), "frozen config changed after registration"
    n_test = 0
    for c in test_conds:
        te = texts(rows, c, "val" if smoke else "test", lim)  # smoke never touches the test split
        Xte, yte = [x["text"] for x in te], np.array([labels[x["category"]] for x in te])
        t1 = time.perf_counter()
        pt = model.predict_proba(Xte)
        metrics["test"][c] = {**classification_metrics(yte, pt.argmax(1)),
                              "predict_seconds": round(time.perf_counter() - t1, 2)}
        write_predictions(out / f"predictions_{'smoketest' if smoke else 'test'}_{c}.csv.gz",
                          [x["record_id"] for x in te], yte, pt, id2label)
        n_test += len(te)
    if test_conds and not smoke:
        log_test(r["id"], r["hash"], r["frozen_config_sha256"], test_conds, n_test)

    if "train_curve" in info:
        with open(out / "train_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(info["train_curve"][0]), lineterminator="\n")
            w.writeheader()
            w.writerows(info["train_curve"])
    if info.get("grid"):
        write_json(out / "grid.json", info["grid"])
    metrics["selected_params"] = info.get("selected_params")
    metrics["train_seconds"] = round(train_s, 1)
    metrics["n_train"], metrics["n_val"] = len(tr), len(va)
    write_json(out / "metrics.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage")
    g.add_argument("--run")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global FROZEN
    if a.smoke:
        FROZEN = SMOKE_FROZEN
    spec = load_spec()
    runs = [x for x in spec["runs"] if a.all or x["id"] == a.run or x["stage"] == a.stage]
    rows = None
    for run in runs:
        try:
            r = resolve(run, spec)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[wait] {run['id']}: {e}")
            continue
        if a.smoke and r.get("reuse"):
            r["reuse"] = None  # smoke: train a tiny model instead of loading a checkpoint
        r["hash"] = config_hash({k: v for k, v in r.items() if k not in ("hash",)})
        out = (ARTIFACTS / "smoke" if a.smoke else RUNS) / r["id"]
        if a.dry_run:
            print(r["id"], r["hash"], r["train"], r["test"])
            continue
        m = out / "metrics.json"
        if m.exists() and read_json(m).get("config_hash") == r["hash"]:
            print(f"[skip] {r['id']} (done, hash {r['hash']})")
            continue
        if rows is None:
            from m3.data.build_conditions import load_conditions
            rows = load_conditions()
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        manifest = {"run_id": r["id"], "stage": r["stage"], "config": r, "config_hash": r["hash"],
                    "seed": r["seed"], "device": dev,
                    "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
                    "versions": versions(), "git_commit": git_commit(),
                    "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        set_status(r["id"], state="running", start=manifest["start_utc"], smoke=a.smoke)
        print(f"[run] {r['id']} train={r['train']} test={r['test']} device={dev}", flush=True)
        t0 = time.perf_counter()
        try:
            metrics = execute(r, rows, smoke=a.smoke)
        except Exception:
            set_status(r["id"], state="failed", error=traceback.format_exc()[-1500:])
            print(traceback.format_exc(), flush=True)
            continue
        manifest["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        manifest["wall_seconds"] = round(time.perf_counter() - t0, 1)
        write_json(out / "manifest.json", manifest)
        set_status(r["id"], state="done", end=manifest["end_utc"], wall_seconds=manifest["wall_seconds"])
        v = next(iter(metrics["val"].values()))
        print(f"[done] {r['id']} val macro-F1 {v['macro_f1']:.4f} in {manifest['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
