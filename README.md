# PrivacyGuard

PrivacyGuard is the S553 Milestone 2 project for a controlled study of how PII sanitization affects downstream intent classification.

## Code

- `prepare_data.py` — downloads and verifies BANKING77, profiles the data, applies duplicate/overlap controls, generates synthetic PII with Faker 37.12.0 using seed 553, and creates the raw/delete/generic/typed variants and evidence files.
- `verify_evidence.py` — verifies an existing generated evidence package, including hashes, row counts, joins, removal checks, eligibility logic, sample values, and human-review coverage.
- `requirements.txt` — pinned Python dependencies.

## Reproduce and verify the evidence

The repository includes the frozen data/evidence snapshot used by the M2 presentation.

```bash
python -m pip install -r requirements.txt
python prepare_data.py --download
python verify_evidence.py
```

`prepare_data.py` rebuilds the machine-generated evidence from the verified BANKING77 source files using Faker 37.12.0 and seed 553. It writes the deterministic pre-review sheet to `data/human_review_24.pre_review.csv` and deliberately does **not** overwrite the completed human-reviewed artifact `data/human_review_24.csv`.

`verify_evidence.py` checks the frozen source hashes, joins, counts, sanitization outputs, eligibility logic, the completed 24-record human review, and hashes recorded in `data/source_manifest.json`.

## Project documentation

Supporting audit/crosswalk material is in `docs/`.

## Data sources

- BANKING77 (PolyAI), CC BY 4.0: public banking intent-classification dataset.
- Faker, MIT license: used locally to generate synthetic PERSON, EMAIL, PHONE, and ADDRESS values.

No real customer or employer PII is used by the project.
## M3: model selection and experiments

Milestone 3 adds a detector evaluation, a non-Faker challenge set, three intent classifiers, frozen and
retrained comparisons of the sanitization methods, and the statistics behind the M3 deck. The M2 files
above are unchanged and `verify_evidence.py` still passes.

What was added:

- `m3/data/` — stratified train/validation split (seed 553), the eight text conditions per record
  (original, raw, three recorded-span masks, three detector-span masks), token-length check.
- `m3/detector/` — Presidio runs (spaCy `en_core_web_lg` and `en_core_web_trf`, plus the M2 address rule),
  a Hugging Face NER add-on, detector metrics, and the hand-written challenge set generator.
- `m3/models/`, `m3/run_experiments.py`, `m3/experiments.yaml` — TF-IDF + LinearSVC, frozen MiniLM
  embeddings + logistic regression, fine-tuned DistilBERT; a resumable runner with run manifests.
- `m3/analysis/` — bootstrap confidence intervals, McNemar tests, error patterns, figures and
  `results/m3/deck_numbers.json` (every number in the deck with its source file).
- `m3/tests/` — pytest checks for splits, conditions, the challenge set, metrics and statistics.

Environment: Python 3.12 for modeling (`requirements-m3.txt`; PyTorch CPU build for latency, ROCm build on
an AMD Radeon RX 7900 XTX for training) and the detector environments in `requirements-presidio-m3.txt`.
Device choice and timings are in `results/m3/env/`.

Reproduce (from the repository root, in the matching environment):

```bash
python -m m3.data.splits
python -m m3.detector.run_detector --all          # add --engine trf for the selected detector
python -m m3.data.build_conditions
python -m m3.run_experiments --all
python -m m3.analysis.numbers                     # after evaluate, risk, error_analysis, business, figures
python -m pytest m3/tests -q
```

Where results live:

- `results/m3/detector/` — detector variants, selection rule and decision, metric definitions.
- `results/m3/runs/<run_id>/` — manifest, metrics and predictions for each of the 18 runs.
- `results/m3/selection.md`, `configs/frozen_model.yaml`, `results/m3/test_eval_log.csv` — model choice,
  frozen configuration and every test-set evaluation.
- `results/m3/analysis/`, `results/m3/figures/`, `results/m3/latency.json` — tables, findings, figures, timings.
- `results/m3/preregistration.md` — research questions and thresholds fixed before testing.

`docs/M3_EVIDENCE_CROSSWALK.md` maps each slide of the M3 deck to the files behind its numbers, and
`docs/M3_REFERENCES_VERIFIED.md` records how each cited source was checked.
