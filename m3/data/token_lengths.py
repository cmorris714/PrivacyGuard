"""Token-length percentiles per tokenizer and condition; picks max_len.

Usage: python -m m3.data.token_lengths
Writes results/m3/data/token_lengths.csv and results/m3/data/max_len.json.
max_len = smallest of 64/96/128 that covers >= 99.9% of train+val inputs in every condition.
"""
import csv

import numpy as np
from transformers import AutoTokenizer

from m3.common import RESULTS, write_json
from m3.data.build_conditions import CONDITIONS, load_conditions

TOKENIZERS = ("distilbert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2")
CANDIDATES = (64, 96, 128)


def main():
    rows = load_conditions()
    out, choice = [], {}
    for name in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(name)
        cover_ok = {L: True for L in CANDIDATES}
        test_trunc = {L: 0 for L in CANDIDATES}
        for c in CONDITIONS:
            for part, splits in (("train+val", ("train", "val")), ("test", ("test",))):
                texts = [r["text"] for r in rows if r["condition"] == c and r["split"] in splits]
                n = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
                rec = {"tokenizer": name, "condition": c, "part": part, "n": len(n),
                       "p50": int(np.percentile(n, 50)), "p95": int(np.percentile(n, 95)),
                       "p99": int(np.percentile(n, 99)), "p99_9": float(np.percentile(n, 99.9)), "max": int(n.max())}
                for L in CANDIDATES:
                    rec[f"over_{L}"] = int((n > L).sum())
                    if part == "train+val" and (n <= L).mean() < 0.999:
                        cover_ok[L] = False
                    if part == "test":
                        test_trunc[L] += int((n > L).sum())
                out.append(rec)
        L = next(L for L in CANDIDATES if cover_ok[L])
        choice[name] = {"max_len": L, "test_inputs_truncated_all_conditions": test_trunc[L]}
    RESULTS.joinpath("data").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "data" / "token_lengths.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    write_json(RESULTS / "data" / "max_len.json", {"rule": "smallest of 64/96/128 covering >= 99.9% of train+val inputs in every condition", **choice})
    print(choice)


if __name__ == "__main__":
    main()
