"""Draw the 24-record review sample of detector-sanitized text (4 PII types x 2 placements x 3, train split).

Usage: python -m m3.data.review_gate
Writes data/m3/detector_review_24.csv. The review columns are filled after reading each row.
"""
import csv
import random

from m3.common import DATA_M3, PII_TYPES, PLACEMENTS, SEED
from m3.data.build_conditions import load_conditions

FIELDS = ["record_id", "pii_type", "placement", "category", "raw", "D2_delete", "D3_generic", "D4_typed",
          "n_spans_masked", "residual_leak", "review_status", "reviewer", "review_date", "note", "carson_signoff"]


def main():
    rows = load_conditions(("C1", "D2", "D3", "D4"), ("train",))
    by = {}
    for r in rows:
        by.setdefault(r["record_id"], {})[r["condition"]] = r
    rng = random.Random(SEED)
    out = []
    for t in PII_TYPES:
        for p in PLACEMENTS:
            ids = sorted(i for i, d in by.items() if d["C1"]["pii_type"] == t and d["C1"]["placement"] == p)
            for rid in rng.sample(ids, 3):
                d = by[rid]
                out.append({"record_id": rid, "pii_type": t, "placement": p, "category": d["C1"]["category"],
                            "raw": d["C1"]["text"], "D2_delete": d["D2"]["text"], "D3_generic": d["D3"]["text"],
                            "D4_typed": d["D4"]["text"], "n_spans_masked": d["D3"]["n_spans_masked"],
                            "residual_leak": d["D3"]["residual_leak"], "review_status": "", "reviewer": "",
                            "review_date": "", "note": "", "carson_signoff": ""})
    path = DATA_M3 / "detector_review_24.csv"
    if path.exists():  # keep review columns already filled in
        old = {r["record_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
        for r in out:
            if r["record_id"] in old:
                for k in ("review_status", "reviewer", "review_date", "note", "carson_signoff"):
                    r[k] = old[r["record_id"]][k]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print("wrote", len(out), "rows")


if __name__ == "__main__":
    main()
