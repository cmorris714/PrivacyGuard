"""Figures for the M3 deck (matplotlib, 200 dpi, sized to their slide placement so point sizes are true).

Usage: python -m m3.analysis.figures
Writes results/m3/figures/*.png and figures_manifest.json (file, size in inches, caption).
Identity is never carried by colour alone: series are also hatched, marker-shaped or directly labelled.
"""
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from m3.common import RESULTS, ROOT, read_json, write_json  # noqa: E402

FIG = RESULTS / "figures"
AN = RESULTS / "analysis"
CRIMSON, GRAY, ORANGE, DARK, LIGHT = "#990000", "#5A5A5A", "#B35900", "#7A0000", "#F0DCDA"
TEXT = "#262626"
plt.rcParams.update({"font.family": ["Calibri", "Carlito", "Arial"], "font.size": 16, "axes.titlesize": 18,
                     "axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "axes.edgecolor": "#8C8C8C", "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
NAMES = {"tfidf_svm": "TF-IDF + SVM", "embed_clf": "MiniLM + classifier", "transformer": "DistilBERT (fine-tuned)"}
COND = {"C0": "Original", "C1": "Raw", "C2": "Delete", "C3": "Generic", "C4": "Typed"}
manifest = {}


def save(fig, name, w, h, caption):
    fig.savefig(FIG / name, dpi=200, facecolor="white")
    plt.close(fig)
    manifest[name] = {"width_in": w, "height_in": h, "caption": caption}


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E3E3E3", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_model_tradeoff():
    lat = read_json(RESULTS / "latency.json")["classifiers"]
    pts = []
    for d in sorted((RESULTS / "runs").glob("e1_*")):
        m = read_json(d / "metrics.json")
        fam = read_json(d / "manifest.json")["config"]["model_cfg"]["family"]
        pts.append((NAMES[fam], lat[d.name]["batch1"]["median_ms"], m["val"]["C1"]["macro_f1"], fam))
    sel = yaml.safe_load(open(ROOT / "configs" / "frozen_model.yaml", encoding="utf-8"))["model"]["family"]
    w, h = 6.0, 3.9
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    for name, ms, f1, fam in pts:
        ax.scatter(ms, f1, s=110, color=CRIMSON if fam == sel else GRAY, marker="o" if fam == sel else "s",
                   zorder=3, edgecolor="white", linewidth=2)
        ax.annotate(f"{name}\n{f1:.3f}", (ms, f1), textcoords="offset points", xytext=(0, 12), ha="center",
                    va="bottom", fontsize=16, color=TEXT)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])  # plain labels; log-format exponents render below 16 pt
    ax.minorticks_off()
    ax.set_xlabel("Median ms per query (CPU, batch 1, log scale)")
    ax.set_ylabel("Validation Macro-F1")
    lo = min(p[2] for p in pts)
    ax.set_ylim(lo - 0.02, 1.0)
    ax.set_xlim(min(p[1] for p in pts) / 3, max(p[1] for p in pts) * 3)
    grid(ax, "both")
    save(fig, "fig_model_tradeoff.png", w, h,
         "Exp 1: validation Macro-F1 vs CPU latency (C1 raw)")


def main_rows():
    return list(csv.DictReader(open(AN / "main_table.csv", encoding="utf-8")))


def fig_main_results():
    rows = {(r["protocol"], r["condition"]): r for r in main_rows()}
    conds = ["C1", "C0", "C2", "C3", "C4"]
    w, h = 7.5, 4.2
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    x = np.arange(len(conds))
    bw = 0.36
    for k, (proto, col, hatch, lab) in enumerate((("frozen", GRAY, "", "Frozen (trained on raw)"),
                                                  ("retrained", CRIMSON, "//", "Retrained on condition"))):
        vals = [float(rows[(proto, c)]["macro_f1"]) if (proto, c) in rows else np.nan for c in conds]
        lo = [float(rows[(proto, c)]["macro_f1_ci_low"]) if (proto, c) in rows else np.nan for c in conds]
        hi = [float(rows[(proto, c)]["macro_f1_ci_high"]) if (proto, c) in rows else np.nan for c in conds]
        err = np.array([np.subtract(vals, lo), np.subtract(hi, vals)])
        ax.bar(x + (k - 0.5) * bw, vals, bw - 0.03, color=col, hatch=hatch, edgecolor="white", linewidth=0,
               label=lab, yerr=err, capsize=4, error_kw={"ecolor": TEXT, "elinewidth": 1.4})
    raw = float(rows[("frozen", "C1")]["macro_f1"])
    ax.axhline(raw, color=TEXT, linestyle="--", linewidth=1.4, label=f"Raw (C1) level, {raw:.3f}")
    allv = [float(r["macro_f1_ci_low"]) for r in rows.values() if r["protocol"] in ("frozen", "retrained")
            and r["condition"] in conds]
    ax.set_ylim(max(0, min(allv) - 0.03), min(1.0, max(float(r["macro_f1_ci_high"]) for r in rows.values()) + 0.008))
    ax.set_xticks(x, ["Raw\nC1", "No contact\nC0", "Delete\nC2", "Generic\nC3", "Typed\nC4"])
    ax.set_ylabel("Test Macro-F1 (axis does not start at 0)")
    ax.legend(loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.4,
              columnspacing=1.0, fontsize=16)
    grid(ax)
    save(fig, "fig_main_results.png", w, h,
         "Exp 2: test Macro-F1, frozen vs retrained, 95% CIs")


def fig_ablation():
    rows = {(r["protocol"], r["condition"]): r for r in main_rows()}
    sl = list(csv.DictReader(open(AN / "slices.csv", encoding="utf-8")))
    items = []
    for proto, lab in (("frozen", "Frozen"), ("retrained", "Retrained")):
        r = rows.get((proto, "C0"))
        if r:
            items.append((f"No contact sentence vs raw\n({lab})", float(r["delta_f1_vs_c1"]),
                          float(r["delta_f1_ci_low"]), float(r["delta_f1_ci_high"])))
    w, h = 6.0, 3.4
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    y = np.arange(len(items))
    for i, (lab, d, lo, hi) in enumerate(items):
        ax.errorbar(d * 100, i, xerr=[[(d - lo) * 100], [(hi - d) * 100]], fmt="o", color=CRIMSON, ecolor=TEXT,
                    capsize=5, markersize=9, elinewidth=1.4)
        ax.annotate(f"{d * 100:+.1f} pp", (d * 100, i), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=16, color=TEXT)
    ax.axvline(0, color=GRAY, linewidth=1.2)
    ax.set_yticks(y, [i[0] for i in items])
    ax.set_xlabel("Difference vs raw, pp (95% CI)")
    ax.set_ylim(-0.6, len(items) - 0.4)
    grid(ax, "x")
    save(fig, "fig_ablation.png", w, h, "Exp 4a: removing the contact sentence (C0 vs C1), test set")
    return sl


def fig_detector():
    sel = read_json(RESULTS / "detector" / "selection.json")["selected"]
    fs, f1 = read_json(RESULTS / "detector" / f"final_{sel}.json"), read_json(RESULTS / "detector" / "final_V1.json")
    types = ["PERSON", "EMAIL", "PHONE", "ADDRESS"]
    w, h = 6.0, 3.6
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    x = np.arange(len(types))
    bw = 0.36
    for k, (d, col, hatch, lab) in enumerate(((f1, GRAY, "", "V1 (M2 detector)"), (fs, CRIMSON, "//", f"{sel} (selected)"))):
        v = [d["challenge_heldout"]["by_type"][t]["detection_recall"] * 100 for t in types]
        bars = ax.bar(x + (k - 0.5) * bw, v, bw - 0.03, color=col, hatch=hatch, edgecolor="white", label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}", ha="center", va="bottom", fontsize=16)
    ax.set_xticks(x, ["Person", "Email", "Phone", "Address"])
    ax.set_ylabel("Detection recall (%)")
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    grid(ax)
    save(fig, "fig_detector.png", w, h, "Exp 3: held-out non-Faker recall, V1 vs selected detector")


def fig_small_intents():
    rows = list(csv.DictReader(open(AN / "small_intents.csv", encoding="utf-8")))
    best = read_json(AN / "analysis.json")["best_method_retrained"]
    intents = sorted({r["intent"] for r in rows}, key=lambda k: int(next(r["train_count"] for r in rows if r["intent"] == k)))
    w, h = 12.3, 4.4
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    y = np.arange(len(intents))
    for k, (proto, c, col, mk, lab) in enumerate((("frozen", "C1", GRAY, "s", "Raw (C1)"),
                                                  ("retrained", best, CRIMSON, "o", f"{COND[best]} (retrained)"))):
        sub = {r["intent"]: r for r in rows if r["protocol"] == proto and r["condition"] == c}
        if not sub:
            continue
        rec = np.array([float(sub[i]["recall"]) for i in intents]) * 100
        lo = np.array([float(sub[i]["wilson_low"]) for i in intents]) * 100
        hi = np.array([float(sub[i]["wilson_high"]) for i in intents]) * 100
        ax.errorbar(rec, y + (k - 0.5) * 0.3, xerr=[rec - lo, hi - rec], fmt=mk, color=col, ecolor=col,
                    capsize=3, markersize=8, label=lab, elinewidth=1.3)
    tc = {r["intent"]: r["train_count"] for r in rows}
    ax.set_yticks(y, [f"{i.replace('_', ' ')} (n={tc[i]})" for i in intents])
    ax.set_xlabel("Test recall per intent (%), 40 test queries each, Wilson 95% CI")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(loc="lower left", frameon=False, ncol=2)
    grid(ax, "x")
    save(fig, "fig_small_intents.png", w, h, "Per-intent test recall for the 10 smallest training intents (n = training examples)")


def fig_risk_coverage():
    rc = read_json(AN / "risk_coverage.json")
    rows = list(csv.DictReader(open(AN / "risk_curve.csv", encoding="utf-8")))
    w, h = 5.6, 3.4
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    for c, col, mk in (("C1", GRAY, "s"), (read_json(AN / "analysis.json")["best_method_retrained"], CRIMSON, "o")):
        sub = [r for r in rows if r["condition"] == c]
        ax.plot([float(r["coverage"]) * 100 for r in sub], [float(r["precision"]) * 100 for r in sub], color=col,
                marker=mk, markersize=5, linewidth=2, label=f"{COND[c]} (frozen model)")
    t = rc["test"]["C1"]
    ax.scatter([t["coverage"] * 100], [t["auto_precision"] * 100], s=120, color=DARK, zorder=4, marker="D",
               label=f"Confidence threshold = {rc['tau']:.2f} (raw)")
    ax.set_xlabel("Auto-routed share (%)")
    ax.set_ylabel("Precision (%)")
    ax.legend(loc="lower left", frameon=False)
    grid(ax, "both")
    save(fig, "fig_risk_coverage.png", w, h, "Risk-coverage on test: precision vs auto-routed share")


def fig_train_curve():
    src = next((d.name for d in sorted((RESULTS / "runs").glob("e1_*")) if (d / "train_curve.csv").exists()), None)
    if src is None:
        return
    p = RESULTS / "runs" / src / "train_curve.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    w, h = 6.0, 3.4
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    ax.plot(ep, [float(r["train_loss"]) for r in rows], color=GRAY, marker="s", linewidth=2, label="Train loss")
    ax.plot(ep, [float(r["val_loss"]) for r in rows], color=ORANGE, marker="^", linewidth=2, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xticks(ep)
    ax.set_xlim(0.6, len(ep) + 0.4)
    ax.set_ylim(min(float(r["val_loss"]) for r in rows) - 0.6, max(float(r["train_loss"]) for r in rows) + 0.3)
    for e, r in zip(ep, rows):  # validation F1 as labels, not a second y-scale
        ax.annotate(f"F1 {float(r['val_macro_f1']):.3f}", (e, float(r["val_loss"])), textcoords="offset points",
                    xytext=(0, -24), ha="center", fontsize=16, color=CRIMSON)
    ax.legend(loc="upper right", frameon=False)
    grid(ax)
    save(fig, "fig_train_curve.png", w, h, "Exp 1 DistilBERT (lr 2e-5): loss per epoch with validation Macro-F1 labelled; still improving at the 4-epoch cap")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig_model_tradeoff, fig_main_results, fig_ablation, fig_detector, fig_small_intents,
              fig_risk_coverage, fig_train_curve):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")
    write_json(FIG / "figures_manifest.json", manifest)
    print(json.dumps(list(manifest)))


if __name__ == "__main__":
    main()
