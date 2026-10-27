"""The three Exp 1 classifier families behind one interface.

  tfidf_svm   B1  TF-IDF + LinearSVC (grid search on the validation split via PredefinedSplit)
  embed_clf   B2  frozen sentence-transformers/all-MiniLM-L6-v2 embeddings + LogisticRegression or MLP
  transformer T1  fine-tuned distilbert-base-uncased (AdamW, linear warmup/decay, early stopping)

Each class: fit(train_texts, train_y, val_texts, val_y) -> info dict; predict_proba(texts) -> (n, 77) array;
save(dir) / load(dir). Grid searches only ever look at the validation split, never at test.
"""
import itertools
import json
import os
import pickle
import time

import numpy as np

from m3.models.evaluation import classification_metrics, softmax

N_LABELS = 77


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- B1
class TfidfSvm:
    GRID = {"analyzer": [("word", (1, 1)), ("word", (1, 2)), ("char_wb", (2, 5))],
            "C": [0.1, 0.3, 1, 3, 10], "min_df": [1, 2], "class_weight": [None, "balanced"]}

    def __init__(self, cfg):
        self.cfg = cfg
        self.pipe = None

    def _pipe(self, analyzer, ngram, C, min_df, class_weight):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC
        return make_pipeline(
            TfidfVectorizer(analyzer=analyzer, ngram_range=tuple(ngram), sublinear_tf=True, min_df=min_df,
                            lowercase=True),
            LinearSVC(C=C, class_weight=class_weight, random_state=self.cfg["seed"], max_iter=5000))

    def fit(self, Xtr, ytr, Xva, yva):
        params = self.cfg.get("params")
        grid_log = []
        if params is None:  # grid search: fit on train, score on validation
            best = None
            for (an, ng), C, md, cw in itertools.product(*self.GRID.values()):
                p = self._pipe(an, ng, C, md, cw).fit(Xtr, ytr)
                f1 = classification_metrics(yva, p.predict(Xva))["macro_f1"]
                rec = {"analyzer": an, "ngram": list(ng), "C": C, "min_df": md, "class_weight": cw, "val_macro_f1": f1}
                grid_log.append(rec)
                if best is None or f1 > best["val_macro_f1"]:
                    best = rec
            params = {k: best[k] for k in ("analyzer", "ngram", "C", "min_df", "class_weight")}
        self.params = params
        self.pipe = self._pipe(params["analyzer"], params["ngram"], params["C"], params["min_df"],
                               params["class_weight"]).fit(Xtr, ytr)
        return {"selected_params": params, "grid": grid_log}

    def predict_proba(self, X):
        # LinearSVC has no probabilities; softmax of the decision scores is used for ranking/confidence
        s = self.pipe.decision_function(X)
        if s.ndim == 1:  # binary case: one score for the positive class
            s = np.stack([-s, s], axis=1)
        return softmax(s)

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        pickle.dump(self.pipe, open(os.path.join(d, "model.pkl"), "wb"))

    def load(self, d):
        self.pipe = pickle.load(open(os.path.join(d, "model.pkl"), "rb"))
        return self


# ---------------------------------------------------------------- B2
class EmbedClf:
    ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    GRID = [("logreg", {"C": c}) for c in (1, 10, 100)] + [("mlp", {"alpha": a}) for a in (1e-4, 1e-3)]

    def __init__(self, cfg):
        self.cfg = cfg
        self._enc = None

    def encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(self.cfg.get("encoder", self.ENCODER), device=device())
        return self._enc

    def embed(self, X):
        return self.encoder().encode(list(X), batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def _clf(self, kind, hp):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        if kind == "logreg":
            return LogisticRegression(C=hp["C"], max_iter=3000)
        return MLPClassifier(hidden_layer_sizes=(512,), activation="relu", alpha=hp["alpha"], early_stopping=True,
                             random_state=self.cfg["seed"], max_iter=300)

    def fit(self, Xtr, ytr, Xva, yva):
        Etr, Eva = self.embed(Xtr), self.embed(Xva)
        params = self.cfg.get("params")
        grid_log = []
        if params is None:
            best = None
            for kind, hp in self.GRID:
                c = self._clf(kind, hp).fit(Etr, ytr)
                f1 = classification_metrics(yva, c.predict(Eva))["macro_f1"]
                grid_log.append({"head": kind, **hp, "val_macro_f1": f1})
                if best is None or f1 > best[2]:
                    best = (kind, hp, f1)
            params = {"head": best[0], **best[1]}
        self.params = params
        hp = {k: v for k, v in params.items() if k != "head"}
        self.clf = self._clf(params["head"], hp).fit(Etr, ytr)
        return {"selected_params": params, "grid": grid_log}

    def predict_proba(self, X):
        return self.clf.predict_proba(self.embed(X))

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        pickle.dump(self.clf, open(os.path.join(d, "head.pkl"), "wb"))

    def load(self, d):
        self.clf = pickle.load(open(os.path.join(d, "head.pkl"), "rb"))
        return self


# ---------------------------------------------------------------- T1
class Transformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    def _tok(self):
        from transformers import AutoTokenizer
        if not hasattr(self, "tok"):
            self.tok = AutoTokenizer.from_pretrained(self.cfg["model_id"])
        return self.tok

    def _batches(self, X, y=None, shuffle=False, rng=None, bs=None):
        import torch
        bs = bs or self.cfg["batch_size"]
        idx = np.arange(len(X))
        if shuffle:
            rng.shuffle(idx)
        tok = self._tok()
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            enc = tok([X[k] for k in j], padding=True, truncation=True, max_length=self.cfg["max_len"],
                      return_tensors="pt")  # dynamic padding per batch
            if y is not None:
                enc["labels"] = torch.tensor([y[k] for k in j])
            yield enc

    def fit(self, Xtr, ytr, Xva, yva):
        import torch
        from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
        c = self.cfg
        torch.manual_seed(c["seed"])
        dev = device()
        self.model = AutoModelForSequenceClassification.from_pretrained(c["model_id"], num_labels=N_LABELS).to(dev)
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [{"params": [p for n, p in self.model.named_parameters() if not any(k in n for k in no_decay)],
                   "weight_decay": c["weight_decay"]},
                  {"params": [p for n, p in self.model.named_parameters() if any(k in n for k in no_decay)],
                   "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=c["lr"])
        steps_per_epoch = (len(Xtr) + c["batch_size"] - 1) // c["batch_size"]
        total = steps_per_epoch * c["max_epochs"]
        sched = get_linear_schedule_with_warmup(opt, int(c["warmup_ratio"] * total), total)
        rng = np.random.RandomState(c["seed"])
        curve, best_f1, best_state, bad = [], -1.0, None, 0
        for epoch in range(1, c["max_epochs"] + 1):
            self.model.train()
            t0, losses = time.perf_counter(), []
            for b in self._batches(Xtr, ytr, shuffle=True, rng=rng):
                out = self.model(**{k: v.to(dev) for k, v in b.items()})
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(out.loss.item())
            probs = self.predict_proba(Xva)
            m = classification_metrics(yva, probs.argmax(1))
            val_loss = float(-np.mean(np.log(probs[np.arange(len(yva)), yva] + 1e-12)))
            curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "val_loss": round(val_loss, 5),
                          "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                          "epoch_seconds": round(time.perf_counter() - t0, 1)})
            if m["macro_f1"] > best_f1:
                best_f1, bad = m["macro_f1"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad > c["patience"]:
                    break
        self.model.load_state_dict(best_state)
        best_epoch = max(curve, key=lambda r: r["val_macro_f1"])["epoch"]
        return {"selected_params": {"lr": c["lr"], "best_epoch": best_epoch}, "train_curve": curve}

    def predict_proba(self, X):
        import torch
        dev = next(self.model.parameters()).device
        self.model.eval()
        out = []
        with torch.no_grad():
            for b in self._batches(list(X), bs=128):
                out.append(torch.softmax(self.model(**{k: v.to(dev) for k, v in b.items()}).logits, -1).cpu().numpy())
        return np.concatenate(out)

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        self.model.save_pretrained(d)
        self._tok().save_pretrained(d)

    def load(self, d):
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(d).to(device())
        return self


FAMILIES = {"tfidf_svm": TfidfSvm, "embed_clf": EmbedClf, "transformer": Transformer}


def build(cfg):
    return FAMILIES[cfg["family"]](cfg)


def describe(cfg):
    return json.dumps(cfg, sort_keys=True)
