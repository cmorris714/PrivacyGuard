"""Tests for the statistics helpers and the deck numbers (re-derived from raw prediction files)."""
import csv
import random

import numpy as np
import pytest
from sklearn.metrics import f1_score

from m3.analysis import stats as S
from m3.common import DATA_M3, RESULTS, read_json
from m3.models.evaluation import read_predictions


def test_macro_f1_matches_sklearn():
    rng = np.random.RandomState(0)
    g, p = rng.randint(0, 77, 500), rng.randint(0, 77, 500)
    assert S.macro_f1(g, p) == pytest.approx(f1_score(g, p, labels=list(range(77)), average="macro", zero_division=0))


def test_mcnemar_and_holm():
    a = np.array([True] * 10 + [False] * 2)
    b = np.array([False] * 10 + [True] * 2)
    bb, cc, p = S.mcnemar_exact(a, b)
    assert (bb, cc) == (10, 2) and 0.03 < p < 0.04          # exact two-sided binomial, 2 of 12
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_wilson_interval():
    lo, hi = S.wilson(36, 40)
    assert 0.76 < lo < 0.77 and 0.96 < hi < 0.97


def test_bootstrap_identity_has_zero_delta():
    g = np.arange(77).repeat(4)
    out = S.paired_bootstrap(g, {"C1": g.copy(), "X": g.copy()}, "C1", n_boot=50)
    assert out["X"]["delta"] == 0 and out["X"]["delta_ci"] == [0.0, 0.0]


@pytest.fixture(scope="module")
def numbers():
    p = RESULTS / "deck_numbers.json"
    if not p.exists():
        pytest.skip("deck numbers not built yet")
    return read_json(p)


def test_five_random_deck_numbers_from_predictions(numbers):
    """Re-derive 5 random (protocol, condition) Macro-F1/accuracy values straight from the prediction files."""
    rows = list(csv.DictReader(open(RESULTS / "analysis" / "main_table.csv", encoding="utf-8")))
    an = read_json(RESULTS / "analysis" / "analysis.json")
    labels = read_json(DATA_M3 / "label_map.json")
    for r in random.Random(553).sample(rows, 5):
        run, cond = an["protocols"][r["protocol"]]["conds"][r["condition"]]
        pr = read_predictions(RESULTS / "runs" / run / f"predictions_test_{cond}.csv.gz")
        g = np.array([labels[x["gold"]] for x in pr])
        p = np.array([labels[x["pred"]] for x in pr])
        key = f"{r['protocol']}_{r['condition']}"
        assert numbers[key + "_f1"]["display"] == f"{S.macro_f1(g, p):.3f}"
        assert numbers[key + "_acc"]["display"] == f"{(g == p).mean() * 100:.1f}%"


def test_thresholds_present(numbers):
    assert all(numbers[f"thr_T{i}"]["display"] in ("PASS", "FAIL") for i in range(1, 8))


def test_test_log_only_frozen(numbers):
    log = list(csv.DictReader(open(RESULTS / "test_eval_log.csv", encoding="utf-8")))
    assert log[0]["run_id"] == "FROZEN_CONFIG_REGISTERED"
    reg = log[0]["frozen_config_sha256"]
    for r in log[1:]:
        assert r["frozen_config_sha256"] == reg and r["timestamp_utc"] >= log[0]["timestamp_utc"]
