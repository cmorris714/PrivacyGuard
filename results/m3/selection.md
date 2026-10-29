# Model selection (Exp 1)

All models trained on the C1 raw training split (7,993 queries) and compared on the C1 validation split (1,999). Latency: CPU, batch 1, 500 validation records (`results/m3/latency.json`).

| Model | Parameters | Val accuracy | Val macro-P | Val macro-R | Val Macro-F1 | p95 ms/query | Training s |
|---|---|---|---|---|---|---|---|
| T1 DistilBERT (fine-tuned) | 67.0M | 0.8019 | 0.7853 | 0.7609 | 0.7495 | 9.04 | 88 |
| B2 MiniLM embeddings + classifier | 22.7M | 0.9020 | 0.9118 | 0.9007 | 0.9036 | 5.72 | 131 |
| B1 TF-IDF + LinearSVC | 4.1M | 0.8794 | 0.8838 | 0.8773 | 0.8785 | 0.49 | 83 |

Highest validation Macro-F1: B2 MiniLM embeddings + classifier (0.9036).
Models within 0.5 pp: B2 MiniLM embeddings + classifier.
**Selected: B2 MiniLM embeddings + classifier** (`e1_minilm_clf_c1_s553`), frozen to `configs/frozen_model.yaml` (SHA-256 `f8da789dac479ca3dc9fa067c63f30bc695add30facf567ee014b152f4652ec8`).

Grid choices: T1 DistilBERT (fine-tuned): {'lr': 2e-05, 'best_epoch': 4}; B2 MiniLM embeddings + classifier: {'head': 'logreg', 'C': 10}; B1 TF-IDF + LinearSVC: {'analyzer': 'word', 'ngram': [1, 2], 'C': 10, 'min_df': 1, 'class_weight': 'balanced'}

Majority-class floor (B0, `b0_majority_c1_s553`, added 2026-10-28 as a reference, not a selection candidate): val accuracy 0.0190, Macro-F1 0.0005.

Considered but not run (scope decision): TextCNN, BiLSTM, BERT-base, RoBERTa-base, ModernBERT-base, zero-shot Qwen2.5 instruction model, a second DistilBERT learning rate (5e-5), all-mpnet-base-v2.
