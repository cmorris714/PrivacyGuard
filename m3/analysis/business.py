"""Business-value numbers from measured results only.

Usage: python -m m3.analysis.business
Writes results/m3/analysis/business.json. Every value carries the formula and inputs used.
A managed-service price is included only if results/m3/analysis/pricing_source.json exists (written from the
provider's pricing page with its access date); otherwise the comparison stays qualitative.
"""
import csv

import yaml

from m3.common import DATA_M3, RESULTS, ROOT, read_json, read_jsonl, write_json

OUT = RESULTS / "analysis"


def main():
    table = list(csv.DictReader(open(OUT / "main_table.csv", encoding="utf-8")))
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    det_sel = read_json(RESULTS / "detector" / f"final_{sel}.json")
    det_v1 = read_json(RESULTS / "detector" / "final_V1.json")
    lat = read_json(RESULTS / "latency.json")
    src = yaml.safe_load(open(ROOT / "configs" / "frozen_model.yaml", encoding="utf-8"))["source_run"]
    pipe = lat["pipeline"][src]
    out = {"misroutes_per_1000": {}, "assumptions": []}
    for r in table:
        if r["protocol"] in ("frozen", "retrained"):
            mis = (1 - float(r["accuracy"])) * 1000
            raw = (1 - float(next(x["accuracy"] for x in table if x["protocol"] == r["protocol"] and x["condition"] == "C1"))) * 1000
            out["misroutes_per_1000"][f"{r['protocol']}_{r['condition']}"] = {
                "value": round(mis, 1), "added_vs_raw": round(mis - raw, 1), "formula": "(1 - test accuracy) x 1,000"}
    out["leaks_per_10000"] = {
        "V1": round(det_v1["ledger_test"]["ALL"]["residual_leak_rate"] * 10000, 1),
        sel: round(det_sel["ledger_test"]["ALL"]["residual_leak_rate"] * 10000, 1),
        "formula": "residual leak rate on the 3,080 test ledger records x 10,000"}
    out["over_redactions_per_1000_clean"] = {
        "V1": round(det_v1["over_redaction_test"]["over_redaction_rate"] * 1000, 1),
        sel: round(det_sel["over_redaction_test"]["over_redaction_rate"] * 1000, 1),
        "formula": "share of clean test queries with a target-type detection x 1,000"}
    rph = pipe["records_per_hour_batch1"]
    out["throughput"] = {"records_per_hour_end_to_end": rph,
                         "hours_per_million_records": round(1_000_000 / rph, 1),
                         "median_ms_end_to_end": pipe["median_ms"], "p95_ms_end_to_end": pipe["p95_ms"],
                         "device": f"CPU ({lat['cpu']}), batch 1, one process",
                         "formula": "3,600,000 ms / mean end-to-end ms per record"}
    c1 = [r["text"] for r in read_jsonl(DATA_M3 / "conditions.jsonl.gz") if r["condition"] == "C1"]
    mean_bytes = sum(len(t.encode("utf-8")) for t in c1) / len(c1)
    out["mean_utf8_bytes_per_record"] = round(mean_bytes, 1)
    out["local_api_cost_usd"] = {"value": 0, "note": "runs on hardware already owned; electricity not estimated"}
    price = OUT / "pricing_source.json"
    if price.exists():
        p = read_json(price)
        gb = 1_000_000 * mean_bytes / 1e9
        billable = max(0.0, gb - p.get("free_gb_per_month", 0.0))
        out["managed_service_comparison"] = {
            "volume_records": 1_000_000, "volume_gb": round(gb, 3), "billable_gb": round(billable, 3),
            "usd_per_gb": p["usd_per_gb"], "estimated_usd": round(billable * p["usd_per_gb"], 2),
            "source": p["url"], "accessed": p["accessed"], "unit": p["unit"]}
    else:
        out["managed_service_comparison"] = {"note": "usage-based pricing per GB inspected/transformed; no dollar figure shown"}
    out["assumptions"] = ["accuracy and leak rates measured on the BANKING77 test split with synthetic PII",
                          "throughput measured on one PC at batch size 1; batching would raise it",
                          "no electricity, hardware or staff costs included"]
    write_json(OUT / "business.json", out)
    print({k: out[k] for k in ("leaks_per_10000", "throughput")})


if __name__ == "__main__":
    main()
