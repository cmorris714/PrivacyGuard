"""Latency and throughput on the project PC (CPU), warm models, 3 warm-up calls, 500 records.

Usage:
  python -m m3.latency --part classifiers   # modeling environment (CPU PyTorch)
  python -m m3.latency --part detector      # detector environment of the selected variant
  python -m m3.latency --part combine       # any environment; writes results/m3/latency.json

The same 500 validation records (C1 raw text, seed 553) are used for every part. Classifiers are timed
on the masked text (D3 condition) that the pipeline would feed them. End-to-end time per record is the
sum of the detector (analyze + mask) and classifier times measured on that record at batch size 1.
"""
import argparse
import os
import platform
import random
import statistics
import sys
import time

from m3.common import ARTIFACTS, RESULTS, ROOT, SEED, read_json, write_json

N, WARMUP = 500, 3
PARTS = ARTIFACTS / "latency_parts"  # per-record timings stay local; the summary is results/m3/latency.json


def sample_records():
    from m3.common import DATA, load_splits, read_jsonl
    splits = load_splits()
    ids = sorted(i for i, s in splits.items() if s == "val")
    ids = random.Random(SEED).sample(ids, N)
    inj = {r["record_id"]: r for r in read_jsonl(DATA / "pii_injections.jsonl")}
    return ids, [inj[i]["raw_text"] for i in ids]


def stats(ms):
    ms = sorted(ms)
    return {"n": len(ms), "median_ms": round(statistics.median(ms), 3),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 3), "mean_ms": round(statistics.fmean(ms), 3)}


def cpu_info():
    return {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}


def n_parameters(m):
    """Trainable weights for the transformer; encoder + head for B2; linear weights + biases for B1."""
    fam = m.cfg["family"]
    if fam == "transformer":
        return int(sum(p.numel() for p in m.model.parameters()))
    if fam == "embed_clf":
        head = m.clf
        n_head = (sum(w.size for w in head.coefs_) + sum(b.size for b in head.intercepts_)) if hasattr(head, "coefs_")             else head.coef_.size + head.intercept_.size
        return int(sum(p.numel() for p in m.encoder().parameters()) + n_head)
    svc = m.pipe[-1]
    return int(svc.coef_.size + svc.intercept_.size)


def part_classifiers():
    import numpy as np
    import torch
    from m3.data.build_conditions import load_conditions
    from m3.models import classifiers
    threads = max(1, (os.cpu_count() or 2) // 2)
    torch.set_num_threads(threads)
    ids, _ = sample_records()
    d3 = {r["record_id"]: r["text"] for r in load_conditions(("D3",), ("val",))}
    texts = [d3[i] for i in ids]
    out = {"device": "cpu", "torch_threads": threads, **cpu_info(), "models": {}}
    for run_dir in sorted((RESULTS / "runs").glob("e1_*")):
        cfg = read_json(run_dir / "manifest.json")["config"]["model_cfg"]
        m = classifiers.build(cfg).load(str(ARTIFACTS / "models" / run_dir.name))
        for t in texts[:WARMUP]:
            m.predict_proba([t])
        per = []
        for t in texts:
            t0 = time.perf_counter()
            m.predict_proba([t])
            per.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        for i in range(0, len(texts), 64):
            m.predict_proba(texts[i:i + 64])
        b64 = (time.perf_counter() - t0) * 1000 / len(texts)
        out["models"][run_dir.name] = {"family": cfg["family"], "n_parameters": n_parameters(m), "batch1": stats(per),
                                       "batch64_ms_per_record": round(b64, 3),
                                       "records_per_hour_batch64": int(3_600_000 / b64), "per_record_ms": per}
        print(run_dir.name, out["models"][run_dir.name]["batch1"], round(b64, 3))
    write_json(PARTS / "classifiers.json", out)


def part_detector():
    sys.path.insert(0, str(ROOT / "evidence"))
    from m3.detector import metrics as M
    from m3.detector.run_detector import build_engine
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    engine = {"V1": "lg", "V0": "lg", "V2": "trf"}[sel]
    eng = build_engine(engine)
    ids, texts = sample_records()
    for t in texts[:WARMUP]:
        eng.analyze(text=t, language="en")
    per = []
    for t in texts:
        t0 = time.perf_counter()
        res = eng.analyze(text=t, language="en")
        M.mask(t, M.filter_target([(r.start, r.end, r.entity_type, r.score) for r in res]), "generic")
        per.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    for i in range(0, len(texts), 64):  # batch 64: one call per record, measured as a block
        for t in texts[i:i + 64]:
            res = eng.analyze(text=t, language="en")
            M.mask(t, M.filter_target([(r.start, r.end, r.entity_type, r.score) for r in res]), "generic")
    b64 = (time.perf_counter() - t0) * 1000 / len(texts)
    out = {"variant": sel, "engine": engine, **cpu_info(), "batch1": stats(per), "batch64_ms_per_record": round(b64, 3),
           "records_per_hour": int(3_600_000 / b64), "per_record_ms": per}
    print(sel, out["batch1"])
    write_json(PARTS / "detector.json", out)


def part_combine():
    det = read_json(PARTS / "detector.json")
    cls = read_json(PARTS / "classifiers.json")
    out = {"records": N, "warmup_calls": WARMUP, "sample": "500 validation records, seed 553",
           "device": "CPU", "cpu": read_json(RESULTS / "env" / "machine.json")["cpu"],
           "torch_threads": cls["torch_threads"],
           "detector": {k: v for k, v in det.items() if k != "per_record_ms"}, "classifiers": {}, "pipeline": {}}
    for run, c in cls["models"].items():
        out["classifiers"][run] = {k: v for k, v in c.items() if k != "per_record_ms"}
        e2e = [a + b for a, b in zip(det["per_record_ms"], c["per_record_ms"])]
        s = stats(e2e)
        out["pipeline"][run] = {**s, "records_per_hour_batch1": int(3_600_000 / s["mean_ms"]),
                                "note": "per-record sum of detector (analyze + mask) and classifier times"}
    write_json(RESULTS / "latency.json", out)
    print({k: v["median_ms"] for k, v in out["pipeline"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["classifiers", "detector", "combine"], required=True)
    p = ap.parse_args().part
    {"classifiers": part_classifiers, "detector": part_detector, "combine": part_combine}[p]()
