"""Run Presidio over the PII ledger texts, the clean original queries and the challenge set.

Usage (detector environment):
  python -m m3.detector.run_detector --all                 # spaCy en_core_web_lg engine (V0 and V1)
  python -m m3.detector.run_detector --all --engine trf    # spaCy en_core_web_trf engine (V2)

All returned spans are stored with their recognizer, so V0 (no address rule) and V1 (with the
custom address rule from evidence/address_recognizer.py) are both derived from one pass.
Output: data/m3/detector_spans_<engine>_<corpus>.jsonl.gz
"""
import argparse
import sys
import time

from m3.common import DATA, DATA_M3, ROOT, norm, read_jsonl, write_jsonl

sys.path.insert(0, str(ROOT / "evidence"))
from address_recognizer import add_to as add_address_rule  # noqa: E402

MODELS = {"lg": "en_core_web_lg", "trf": "en_core_web_trf"}


def build_engine(engine):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    if engine == "lg":
        # Presidio's default configuration (en_core_web_lg), exactly as in the M2 pilot
        return add_address_rule(AnalyzerEngine())
    conf ={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": MODELS[engine]}]}
    if engine == "trf":
        # the trf pipeline uses OntoNotes-style labels; map the ones Presidio needs
        conf["ner_model_configuration"] = {
            "model_to_presidio_entity_mapping": {
                "PERSON": "PERSON", "PER": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION",
                "FAC": "LOCATION", "NORP": "NRP", "ORG": "ORGANIZATION", "DATE": "DATE_TIME", "TIME": "DATE_TIME"},
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ORG", "ORGANIZATION"],
            "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "LANGUAGE",
                                 "LAW", "EVENT", "PRODUCT", "WORK_OF_ART"],
        }
    nlp = NlpEngineProvider(nlp_configuration=conf).create_engine()
    eng = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    add_address_rule(eng)
    return eng


def corpus_texts(name):
    if name == "ledger":
        return [(r["record_id"], r["raw_text"]) for r in read_jsonl(DATA / "pii_injections.jsonl")]
    if name == "clean":
        return [(r["record_id"], norm(r["text"])) for r in read_jsonl(DATA / "banking_records.jsonl")]
    if name == "challenge":
        return [(r["id"], r["text"]) for r in read_jsonl(DATA_M3 / "challenge_set.jsonl")]
    raise ValueError(name)


def run(engine, corpus, limit=None):
    eng = build_engine(engine)
    items = corpus_texts(corpus)[:limit]
    eng.analyze(text="My name is Warm Up and my email is warm.up@example.com.", language="en")
    rows = []
    t_all = time.perf_counter()
    for rid, text in items:
        t0 = time.perf_counter()
        res = eng.analyze(text=text, language="en")
        ms = (time.perf_counter() - t0) * 1000
        spans = sorted(
            [[r.start, r.end, r.entity_type, round(r.score, 4),
              (r.recognition_metadata or {}).get("recognizer_name", "")] for r in res])
        rows.append({"id": rid, "spans": spans, "ms": round(ms, 3)})
    wall = time.perf_counter() - t_all
    out = DATA_M3 / f"detector_spans_{engine}_{corpus}.jsonl.gz"
    if limit is None:
        write_jsonl(out, rows)
    print(f"{engine}/{corpus}: {len(rows)} texts in {wall:.1f}s -> {out.name if limit is None else '(not saved)'}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=sorted(MODELS), default="lg")
    ap.add_argument("--corpus", choices=["ledger", "clean", "challenge"], action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    corpora = ["challenge", "ledger", "clean"] if a.all else (a.corpus or ["ledger"])
    for c in corpora:
        if c == "challenge" and not (DATA_M3 / "challenge_set.jsonl").exists():
            print("challenge_set.jsonl missing; skipped")
            continue
        run(a.engine, c, a.limit)


if __name__ == "__main__":
    main()
