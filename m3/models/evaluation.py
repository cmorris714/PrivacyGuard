"""Classification metrics and prediction files shared by every model."""
import csv
import gzip
import io

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def softmax(scores):
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classification_metrics(gold, pred, n_labels=77):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=list(range(n_labels)), average="macro",
                                                 zero_division=0)
    return {"n": int(len(gold)), "accuracy": round(float(accuracy_score(gold, pred)), 6),
            "macro_precision": round(float(p), 6), "macro_recall": round(float(r), 6), "macro_f1": round(float(f), 6)}


def write_predictions(path, record_ids, gold, probs, id2label):
    """CSV gzip: record_id, gold, pred, prob_max, top3 labels and probabilities."""
    top = np.argsort(-probs, axis=1)[:, :3]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["record_id", "gold", "pred", "prob_max", "top1", "top2", "top3", "p1", "p2", "p3"])
    for i, rid in enumerate(record_ids):
        t = top[i]
        w.writerow([rid, id2label[gold[i]], id2label[t[0]], f"{probs[i, t[0]]:.5f}",
                    id2label[t[0]], id2label[t[1]], id2label[t[2]],
                    f"{probs[i, t[0]]:.5f}", f"{probs[i, t[1]]:.5f}", f"{probs[i, t[2]]:.5f}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def read_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))
