"""Tests for splits, challenge set, conditions and detector metric functions."""
import collections
import re

import pytest

from m3.common import DATA_M3, RESULTS, load_injections, read_json, read_jsonl
from m3.detector import metrics as M
from m3.detector.challenge_set import ALLOWED_EMAIL, PHONE_OK, build, validate


# ---------- metric functions on hand-made cases ----------
TEXT = "My name is Ann Lee. Card lost"   # value "Ann Lee" at 11..18


def test_found_and_full_span():
    assert M.is_found([(11, 14, "PERSON")], "PERSON", 11, 18)
    assert not M.is_full_span([(11, 14, "PERSON")], "PERSON", 11, 18)
    assert M.is_full_span([(11, 14, "PERSON"), (14, 18, "PERSON")], "PERSON", 11, 18)
    assert not M.is_found([(11, 18, "LOCATION")], "PERSON", 11, 18)   # wrong type does not count
    assert M.is_found([(11, 18, "LOCATION")], "ADDRESS", 11, 18)


def test_mask_and_leaks():
    full = M.mask(TEXT, [(11, 18, "PERSON")], "typed")
    assert full == "My name is [PERSON]. Card lost"
    assert not M.is_residual_leak(full, "Ann Lee")
    part = M.mask(TEXT, [(11, 14, "PERSON")], "generic")
    assert part == "My name is [REDACTED] Lee. Card lost"
    assert not M.is_residual_leak(part, "Ann Lee")
    assert M.is_partial_leak(part, "Ann Lee")
    assert M.is_residual_leak(M.mask(TEXT, [], "delete"), "ann lee")
    # a surviving token that is also in the original query is not a partial leak
    assert not M.is_partial_leak("My name is [REDACTED] Card. Card lost", "Ann Card", "Card lost")


def test_merge_and_delete_normalizes():
    spans = [(11, 14, "PERSON", 0.5), (11, 18, "LOCATION", 0.9)]
    assert M.merge_spans(spans) == [(11, 18, "LOCATION")]
    assert M.mask(TEXT, spans, "delete") == "My name is . Card lost"


def test_over_redaction_uses_target_types_only():
    assert not M.over_redacted([(0, 5, "DATE_TIME", 0.8)])
    assert M.over_redacted([(0, 5, "LOCATION", 0.8)])


# ---------- splits ----------
def test_splits_disjoint_and_stratified():
    s = read_json(DATA_M3 / "splits.json")
    assert s["counts"] == {"train": 7993, "val": 1999, "test": 3080}
    assert len(s["excluded"]) == 11
    assert all(v == 0 for v in s["normalized_text_overlap"].values())
    a = s["assignment"]
    assert len(a) == 13072 and not set(a) & set(s["excluded"])
    labels = read_json(DATA_M3 / "label_map.json")
    assert sorted(labels.values()) == list(range(77))


# ---------- challenge set ----------
def test_challenge_set_integrity():
    items = read_jsonl(DATA_M3 / "challenge_set.jsonl")
    assert items == build()               # file matches the generator
    cats, tags = validate(items)
    assert sum(cats.values()) == len(items) >= 300
    splits = read_json(DATA_M3 / "splits.json")["assignment"]
    assert all(splits[it["carrier_record_id"]] == "train" for it in items)
    for it in items:
        for sp in it["spans"]:
            if sp["type"] == "EMAIL":
                assert ALLOWED_EMAIL.search(sp["value"])
            if sp["type"] == "PHONE":
                assert any(p.search(sp["value"]) for p in PHONE_OK)
    sub = collections.Counter(it["subset"] for it in items)
    assert abs(sub["challenge_dev"] - sub["challenge_heldout"]) <= 12


# ---------- conditions ----------
@pytest.fixture(scope="module")
def conditions():
    path = DATA_M3 / "conditions.jsonl.gz"
    if not path.exists():
        pytest.skip("conditions not built yet")
    return read_jsonl(path)


def test_conditions_counts_and_integrity(conditions):
    inj = load_injections()
    by_c = collections.Counter(r["condition"] for r in conditions)
    assert set(by_c.values()) == {13083} and len(by_c) == 8
    for r in conditions:
        if r["condition"] in ("C2", "C3", "C4"):
            assert inj[r["record_id"]]["pii_value"].casefold() not in r["text"].casefold()
        if r["condition"] == "C3":
            assert "[REDACTED]" in r["text"]
    # every record keeps one split across conditions
    sp = collections.defaultdict(set)
    for r in conditions:
        sp[r["record_id"]].add(r["split"])
    assert all(len(v) == 1 for v in sp.values())


def test_detector_leak_flags_match_metrics(conditions):
    sel = read_json(RESULTS / "detector" / "selection.json")
    final = RESULTS / "detector" / f"final_{sel['selected']}.json"
    if not final.exists():
        pytest.skip("final detector metrics not written yet")
    leaks = sum(r["residual_leak"] for r in conditions if r["condition"] == "D3")
    assert leaks == read_json(final)["ledger_all_13083"]["ALL"]["residual_leak"]
    assert re.fullmatch(r"V\d", sel["selected"])
