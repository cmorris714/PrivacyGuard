# Detector selection

Rule: `selection_rule.md` (written before scoring). Data: train+validation ledger records (n = 9992), challenge_dev, clean train+validation queries.

| Variant | Residual leak rate | Leaks | Detection recall | Over-redaction | challenge_dev recall | Median ms | Eligible |
|---|---|---|---|---|---|---|---|
| V0 | 9.57% | 956 | 82.87% | 3.23% | 87.18% | 5.89 | no |
| V1 | 0.24% | 24 | 99.65% | 3.23% | 89.10% | 5.89 | yes |
| V2 | 0.00% | 0 | 100.00% | 3.08% | 99.36% | 17.59 | yes |
| V3 | 0.00% | 0 | 100.00% | 3.32% | 92.95% | 20.62 | yes |

Over-redaction ceiling: V1 rate + 1.0 pp = 4.23%. Latency ceiling: 150 ms median. challenge_dev floor: V1's 89.10%.
Lowest leak rate among eligible variants: V2, V3 (tie). Tie broken by median latency.

**Selected: V2** (Presidio with spaCy `en_core_web_trf` as the NLP engine plus the custom address recognizer).

The address recognizer was built against Faker address shapes in M2; its results on Faker ledger records are therefore optimistic, and the challenge set measures formats it was not built for.
