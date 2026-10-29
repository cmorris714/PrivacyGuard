"""PERSON spans from a Hugging Face token-classification model (detector variant V3 = V1 + these spans).

Usage: python -m m3.detector.hf_ner [--model dslim/bert-base-NER] [--device cpu|cuda]
Output: data/m3/detector_spans_hfner_<corpus>.jsonl.gz (PER entities only, mapped to PERSON).
Timing is per record at batch size 1 with a warm model, like the Presidio runs.
"""
import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from m3.common import DATA_M3, write_jsonl  # noqa: E402
from m3.detector.run_detector import corpus_texts  # noqa: E402

MODEL = "dslim/bert-base-NER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    import torch
    from transformers import pipeline
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    ner = pipeline("token-classification", model=a.model, aggregation_strategy="simple",
                   device=0 if a.device == "cuda" else -1)
    ner("My name is Warm Up.")
    for corpus in ("challenge", "ledger", "clean"):
        rows = []
        t_all = time.perf_counter()
        for rid, text in corpus_texts(corpus)[:a.limit]:
            t0 = time.perf_counter()
            ents = ner(text)
            ms = (time.perf_counter() - t0) * 1000
            spans = [[int(e["start"]), int(e["end"]), "PERSON", round(float(e["score"]), 4), "hf:" + a.model]
                     for e in ents if e["entity_group"] == "PER"]
            rows.append({"id": rid, "spans": spans, "ms": round(ms, 3)})
        if a.limit is None:
            write_jsonl(DATA_M3 / f"detector_spans_hfner_{corpus}.jsonl.gz", rows)
        print(f"hfner/{corpus}: {len(rows)} texts in {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
