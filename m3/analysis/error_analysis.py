"""Data-driven error analysis of the frozen classifier and the selected detector.

Usage: python -m m3.analysis.error_analysis   (after m3.analysis.evaluate)
Writes results/m3/analysis/error_patterns.csv, error_patterns.md, method_induced.csv, detector_errors.csv.

Each candidate pattern is a rule over saved predictions and condition texts; it is kept only when it
matches records. Counts are reported against the relevant error population.
"""
import collections
import csv
import re

import numpy as np

from m3.analysis.evaluate import OUT, load, protocols, write_csv
from m3.common import DATA_M3, RESULTS, load_injections, read_json, read_jsonl

CONTACT_INTENTS = {"edit_personal_details", "verify_my_identity", "why_verify_identity", "change_pin",
                   "passcode_forgotten", "unable_to_verify_identity", "lost_or_stolen_phone"}


def texts_by(cond):
    return {r["record_id"]: r["text"] for r in read_jsonl(DATA_M3 / "conditions.jsonl.gz")
            if r["condition"] == cond and r["split"] == "test"}


def method_induced(P, labels):
    """Records the C1 model gets right on raw text but wrong after a method (and the reverse)."""
    inj = load_injections()
    rows, summary = [], []
    for proto in ("frozen", "retrained"):
        if proto not in P:
            continue
        ids, gold, p1, _ = load(*P[proto]["conds"]["C1"], labels)
        for c in ("C0", "C2", "C3", "C4", "D2", "D3", "D4"):
            if c not in P[proto]["conds"]:
                continue
            _, _, pk, _ = load(*P[proto]["conds"][c], labels)
            hurt = (p1 == gold) & (pk != gold)
            helped = (p1 != gold) & (pk == gold)
            for place in ("prefix", "suffix"):
                m = np.array([inj[i]["template_id"] == place for i in ids])
                summary.append({"protocol": proto, "condition": c, "placement": place,
                                "hurt": int((hurt & m).sum()), "helped": int((helped & m).sum()), "n": int(m.sum())})
            for i in np.where(hurt)[0]:
                rows.append({"protocol": proto, "condition": c, "record_id": ids[i], "direction": "hurt"})
    return rows, summary


def patterns(P, labels, best):
    id2 = {v: k for k, v in labels.items()}
    inj = load_injections()
    ids, gold, p1, conf1 = load(*P["frozen"]["conds"]["C1"], labels)
    found = []

    def add(name, mask, population, pop_name, cond_text, cause, mitigation):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        ex = [i for i in idx[:2]]
        found.append({"pattern": name, "count": int(len(idx)), "population": int(population.sum()),
                      "population_name": pop_name, "share": round(len(idx) / max(1, population.sum()), 4),
                      "example_ids": " ".join(ids[i] for i in ex),
                      "example": " || ".join(f"{cond_text[ids[i]]} -> {id2[gold[i]]} predicted as {id2[pred_for[ids[i]]]}"
                                             for i in ex),
                      "likely_cause": cause, "mitigation": mitigation})

    # 1. errors that are present on raw text already: inherently confusable intent pairs
    err1 = p1 != gold
    pair = collections.Counter((gold[i], p1[i]) for i in np.where(err1)[0])
    sym = collections.Counter()
    for (a, b), n in pair.items():
        sym[tuple(sorted((a, b)))] += n
    top_pairs = {k for k, _ in sym.most_common(10)}
    pred_for = dict(zip(ids, p1))
    raw_text = texts_by("C1")
    m_pairs = np.array([err1[i] and tuple(sorted((gold[i], p1[i]))) in top_pairs for i in range(len(ids))])
    add("Confusable intent pairs (10 most frequent pairs, errors on raw text)", m_pairs, err1, "errors on C1 raw",
        raw_text, "intents with near-identical wording (e.g. pending vs failed, card arrival vs delivery estimate)",
        "now: route low-confidence predictions to review; later: merge or relabel pairs with the business owner")

    # 2. small training intents
    from m3.analysis.evaluate import train_counts
    tc = train_counts()
    small = set(sorted(tc, key=lambda k: (tc[k], k))[:10])
    m_small = np.array([err1[i] and id2[gold[i]] in small for i in range(len(ids))])
    share_small_test = np.mean([id2[g] in small for g in gold])
    add(f"Errors on the 10 smallest training intents (they are {share_small_test:.1%} of test records)", m_small, err1,
        "errors on C1 raw", raw_text,
        f"fewer training examples ({min(tc[k] for k in small)}-{max(tc[k] for k in small)} in the M3 train split)",
        "now: report per-intent recall with Wilson CIs; later: targeted examples or class weighting")

    # 3. method-induced errors for the best method (frozen model)
    for c in (best, "C2"):
        if c not in P["frozen"]["conds"]:
            continue
        _, _, pk, _ = load(*P["frozen"]["conds"][c], labels)
        pred_for = dict(zip(ids, pk))
        tk = texts_by(c)
        hurt = (p1 == gold) & (pk != gold)
        contact = np.array([hurt[i] and id2[pk[i]] in CONTACT_INTENTS for i in range(len(ids))])
        name = {"C2": "delete", "C3": "generic", "C4": "typed"}[c]
        add(f"New errors after {name} masking that predict a contact/identity intent", contact, hurt,
            f"records right on raw, wrong after {name} (frozen model)", tk,
            "the leftover contact sentence ('My phone is [PHONE].' / 'My email is .') reads like a request about personal details",
            "now: retrain on sanitized text (matched protocol); later: drop the dangling contact sentence before classification")
        other = np.array([hurt[i] and id2[pk[i]] not in CONTACT_INTENTS for i in range(len(ids))])
        tau = read_json(OUT / "risk_coverage.json")["tau"]
        low = float((conf1[other] < tau).mean()) if other.any() else 0.0
        add(f"Other new errors after {name} masking", other, hurt,
            f"records right on raw, wrong after {name} (frozen model)", tk,
            f"the masked token shifts a mean-pooled sentence embedding; {low:.0%} of these records were already "
            f"below the review threshold (confidence < {tau:.2f}) on raw text",
            "now: retrain on sanitized text and use the confidence gate; later: more varied contact templates")

    # 4. detector over-redaction on clean queries
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    eng = {"V1": "lg", "V0": "lg", "V2": "trf"}.get(sel, "lg")
    from m3.detector import metrics as M
    clean = {r["id"]: r for r in read_jsonl(DATA_M3 / f"detector_spans_{eng}_clean.jsonl.gz")}
    dk = "D" + best[1]
    if dk in P["frozen"]["conds"]:
        _, _, pd, _ = load(*P["frozen"]["conds"][dk], labels)
        _, _, pc, _ = load(*P["frozen"]["conds"][best], labels)
        pred_for = dict(zip(ids, pd))
        over = np.array([M.over_redacted(clean[i]["spans"]) for i in ids])
        diff = (pc == gold) & (pd != gold)
        add("Detector masks words in the query itself (place names, brands) and the prediction changes",
            diff & over, over, "test records whose clean query has a detection", texts_by(dk),
            "country/city names or capitalised words detected as LOCATION/PERSON inside the banking question",
            "now: allow-list of currency/country terms for the intent model's input; later: entity-aware masking of query text")
    return found


def detector_errors():
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    fin = read_json(RESULTS / "detector" / f"final_{sel}.json")
    rows = []
    for key, v in fin["challenge_heldout"]["by_format"].items():
        t, tag = key.split("/")
        rows.append({"variant": sel, "pii_type": t, "format_tag": tag, "n": v["n"], "found": v["found"],
                     "detection_recall": v["detection_recall"], "residual_leak": v["residual_leak"]})
    v1 = read_json(RESULTS / "detector" / "final_V1.json")["challenge_heldout"]["by_format"]
    for r in rows:
        r["V1_detection_recall"] = v1[f"{r['pii_type']}/{r['format_tag']}"]["detection_recall"]
    return rows


def main():
    labels = read_json(DATA_M3 / "label_map.json")
    P = protocols()
    best = read_json(OUT / "analysis.json")["best_method_retrained"]
    rows, summary = method_induced(P, labels)
    write_csv(OUT / "method_induced.csv", summary)
    pats = patterns(P, labels, best)
    write_csv(OUT / "error_patterns.csv", pats)
    det = detector_errors()
    write_csv(OUT / "detector_errors.csv", det)
    lines = ["# Error patterns (test set; frozen classifier unless noted)", ""]
    for p in pats:
        lines += [f"## {p['pattern']}", f"- Count: {p['count']} of {p['population']} {p['population_name']} ({p['share']:.1%})",
                  f"- Examples: {p['example']}", f"- Likely cause: {p['likely_cause']}", f"- Mitigation: {p['mitigation']}", ""]
    weak = sorted(det, key=lambda r: r["detection_recall"])[:6]
    lines += ["## Detector: weakest challenge_heldout format groups", ""]
    lines += [f"- {r['pii_type']}/{r['format_tag']}: {r['found']}/{r['n']} found (V1: {r['V1_detection_recall']:.0%})" for r in weak]
    (OUT / "error_patterns.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(len(pats), "patterns;", [(p["pattern"][:40], p["count"]) for p in pats])


if __name__ == "__main__":
    main()
