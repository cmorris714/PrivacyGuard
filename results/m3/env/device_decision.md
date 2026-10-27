# Device decision

Date: 2026-10-27. Machine: see `machine.json` (AMD Radeon RX 7900 XTX, 24 GB).

GPU stack: PyTorch 2.9.1+rocm7.2.1 for Windows (Python 3.12) with the ROCm 7.2.1 SDK wheels, installed
from AMD's wheel repository (https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/) into `.venv-m3-rocm`.
`torch.cuda.is_available()` returned True and the device name was "AMD Radeon RX 7900 XTX".

Smoke test (`distilbert-base-uncased`, 50 AdamW steps, batch 16, max_len 96, 512 C1-raw training rows, seed 553, fp32):

| Device | First loss | Mean loss, last 5 steps | Seconds per step |
|---|---|---|---|
| GPU (ROCm) | 4.4854 | 3.0189 | 0.118 |
| CPU (PyTorch 2.14.1+cpu) | 4.4851 | 3.0243 | 2.724 |

Losses agree within 0.2% (criterion: finite and within 10%). The CPU timing ran while another
CPU job was active, so the speed-up is approximate.

Decision: fine-tune transformer models and compute sentence embeddings on the GPU (fp32, no mixed
precision). Presidio detection, TF-IDF + LinearSVC and all latency measurements for deployment use the CPU.
