"""Stratified 80/20 train/validation split of the eligible BANKING77 training rows (seed 553).

Usage: python -m m3.data.splits
Writes data/m3/splits.json and data/m3/label_map.json.
"""
import collections

from sklearn.model_selection import train_test_split

from m3.common import DATA, DATA_M3, SEED, read_jsonl, write_json


def build():
    records = {r["record_id"]: r for r in read_jsonl(DATA / "banking_records.jsonl")}
    elig = read_jsonl(DATA / "eligibility.jsonl")
    inj = {r["record_id"]: r for r in read_jsonl(DATA / "pii_injections.jsonl")}
    train_ids = sorted(e["record_id"] for e in elig if e["split"] == "train" and e["eligible"])
    test_ids = sorted(e["record_id"] for e in elig if e["split"] == "test" and e["eligible"])
    excluded = sorted(e["record_id"] for e in elig if not e["eligible"])
    y = [records[i]["category"] for i in train_ids]
    tr, va = train_test_split(train_ids, test_size=0.2, stratify=y, random_state=SEED)
    assignment = {**{i: "train" for i in tr}, **{i: "val" for i in va}, **{i: "test" for i in test_ids}}

    # integrity: disjoint ids, every intent everywhere, no normalized-text overlap across splits
    sets = {s: {i for i, v in assignment.items() if v == s} for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    labels = sorted({r["category"] for r in records.values()})
    assert len(labels) == 77
    for s, ids in sets.items():
        assert {records[i]["category"] for i in ids} == set(labels), s
    hashes = {s: {records[i]["text_sha256"] for i in ids} for s, ids in sets.items()}
    overlap = {f"{a}-{b}": len(hashes[a] & hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in overlap.values()), overlap

    balance = {}
    for s, ids in sets.items():
        balance[s] = {
            "n": len(ids),
            "pii_type": dict(sorted(collections.Counter(inj[i]["pii_type"] for i in ids).items())),
            "placement": dict(sorted(collections.Counter(inj[i]["template_id"] for i in ids).items())),
            "min_per_intent": min(collections.Counter(records[i]["category"] for i in ids).values()),
            "max_per_intent": max(collections.Counter(records[i]["category"] for i in ids).values()),
        }
    out = {
        "seed": SEED,
        "method": "sklearn train_test_split(stratify=category, test_size=0.2, random_state=553) on eligible train rows; test = official test",
        "counts": {s: len(v) for s, v in sets.items()},
        "excluded": excluded,
        "normalized_text_overlap": overlap,
        "balance": balance,
        "assignment": dict(sorted(assignment.items())),
    }
    write_json(DATA_M3 / "splits.json", out)
    write_json(DATA_M3 / "label_map.json", {lab: k for k, lab in enumerate(labels)})
    return out


if __name__ == "__main__":
    o = build()
    print(o["counts"], "excluded:", len(o["excluded"]))
    for s, b in o["balance"].items():
        print(s, b)
