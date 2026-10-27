# Compute budget

Measured 2026-10-27 on the project PC (`machine.json`).

| Item | Measurement |
|---|---|
| DistilBERT fine-tuning, GPU (ROCm), batch 16, max_len 96 | 0.118 s per step (50-step smoke test) |
| DistilBERT fine-tuning, CPU, same settings | 2.724 s per step (CPU shared with another job) |
| Presidio + en_core_web_lg (V1), CPU | 5.9 ms median per record |
| Presidio + en_core_web_trf (V2), CPU | 17.6 ms median per record |

Projection for one DistilBERT run (7,993 training rows, batch 32, at most 4 epochs = 1,000 steps):
about 2 x 0.118 s x 1,000 = 4 minutes on the GPU with validation passes, against more than an hour on the CPU.
The planned runs (11 DistilBERT trainings plus inference) therefore fit in about 1 hour of GPU time,
so no tight-budget cuts were needed. Model choice was narrowed for scope reasons, not compute:
see the run plan in `m3/experiments.yaml`.
