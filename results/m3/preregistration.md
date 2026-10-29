# M3 pre-registration

Fixed on 2026-10-27, before any test-set evaluation. The official BANKING77 test split (3,080 queries,
40 per intent) is used only by frozen configurations; every test evaluation is logged in
`results/m3/test_eval_log.csv` with its configuration hash.

## Research questions
- **RQ1** Which replacement (delete, generic `[REDACTED]`, typed `[EMAIL]`) keeps intent classification
  closest to the raw text, (a) for a classifier trained on raw text and kept frozen, and (b) when the
  classifier is retrained on sanitized text?
- **RQ2** What do real detector errors (Presidio spans instead of the recorded spans) cost in leaks and
  accuracy, and does the detector work on PII formats that Faker did not generate?
- **RQ3** Does the added contact sentence (and its position before or after the query) change classifier quality?

## Text conditions
C0 original query; C1 raw (contact sentence with the synthetic value); C2/C3/C4 delete/generic/typed using
the recorded spans; D2/D3/D4 the same three methods using the selected detector's spans.

## Models (Exp 1, trained on C1 train, compared on C1 validation)
B1 TF-IDF + LinearSVC (grid on validation), B2 frozen all-MiniLM-L6-v2 embeddings + logistic regression or
MLP (grid on validation), T1 fine-tuned distilbert-base-uncased (lr 2e-5, batch 32, up to 4 epochs, early
stopping on validation Macro-F1). Selection: highest validation Macro-F1; a faster model within 0.5
percentage points wins.

## Metrics
Primary: test Macro-F1. Secondary: accuracy, macro precision and recall, per-intent recall, detector leak
metrics (see `results/m3/detector/DEFINITIONS.md`), latency.

## Acceptance thresholds
| ID | Criterion | Threshold |
|---|---|---|
| T1 | Selected model, validation Macro-F1 on C1 | >= 0.90 |
| T2 | Matched retraining: best method's test Macro-F1 vs C1 | within 1.0 pp |
| T3 | Frozen model: drop of the best method vs C1 | <= 2.0 pp |
| T4 | Selected detector on test ledger records | residual leak rate <= 0.25% overall, 0 for EMAIL and PHONE; detection recall >= 99.5% |
| T5 | Detector on challenge_heldout | detection recall >= 90% |
| T6 | Over-redaction on clean test queries | <= 5% |
| T7 | End-to-end latency (detect + mask + classify), batch 1, this PC | median <= 100 ms per record |

## Statistics
95% confidence intervals from a paired bootstrap over test records (2,000 resamples, seed 553), for each
metric and for each difference against C1. Exact McNemar test for paired accuracy differences against C1,
with Holm correction across the three methods. Per-intent recall with Wilson 95% intervals (n = 40 per intent).
Seed spread (553, 554, 555) for the selected model on C1 and on the best sanitized condition.
