"""Tests for metrics, prediction files, the TF-IDF model and the experiment registry."""
import numpy as np
import pytest
import yaml

from m3.common import ROOT
from m3.models.evaluation import classification_metrics, read_predictions, softmax, write_predictions


def test_metrics_perfect_and_macro():
    g = np.array([0, 0, 1, 1])
    assert classification_metrics(g, g, n_labels=2)["macro_f1"] == 1.0
    m = classification_metrics(g, np.array([0, 0, 0, 1]), n_labels=2)
    assert m["accuracy"] == 0.75 and 0.7 < m["macro_f1"] < 0.75


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(p.sum(1), 1) and p[0].argmax() == 2


def test_prediction_file_roundtrip(tmp_path):
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    write_predictions(tmp_path / "p.csv.gz", ["a", "b"], [0, 1], probs, {0: "x", 1: "y", 2: "z"})
    rows = read_predictions(tmp_path / "p.csv.gz")
    assert [r["pred"] for r in rows] == ["x", "z"] and rows[1]["gold"] == "y" and rows[0]["top2"] == "y"


def test_tfidf_svm_fixed_params():
    from m3.models.classifiers import TfidfSvm
    X = ["card lost", "lost my card", "refund missing", "where is my refund"] * 3
    y = np.array([0, 0, 1, 1] * 3)
    m = TfidfSvm({"seed": 553, "params": {"analyzer": "word", "ngram": [1, 1], "C": 1, "min_df": 1,
                                          "class_weight": None}})
    m.fit(X, y, X, y)
    assert (m.predict_proba(["refund please"]).argmax(1) == 1).all()


def test_majority_floor_predicts_most_frequent():
    from m3.models.classifiers import N_LABELS, Majority
    m = Majority({"seed": 553})
    m.fit(["a", "b", "c"], np.array([5, 2, 5]), [], np.array([]))
    p = m.predict_proba(["x", "y"])
    assert p.shape == (2, N_LABELS) and (p.argmax(1) == 5).all() and np.allclose(p.sum(1), 1)


def test_registry_ids_unique_and_test_only_for_frozen():
    spec = yaml.safe_load(open(ROOT / "m3" / "experiments.yaml", encoding="utf-8"))
    ids = [r["id"] for r in spec["runs"]]
    assert len(ids) == len(set(ids))
    for r in spec["runs"]:
        if r.get("test"):
            assert r["model"] in ("frozen", "b1_frozen"), r["id"]
        if r["stage"] == "e1":
            assert not r.get("test")


def test_config_hash_is_stable():
    from m3.common import config_hash
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


@pytest.mark.parametrize("cond", ["C_best", "D_best"])
def test_symbolic_conditions_need_2b(cond, monkeypatch, tmp_path):
    import m3.run_experiments as R
    monkeypatch.setattr(R, "RUNS", tmp_path)
    with pytest.raises(RuntimeError):
        R.best_sanitized()
