# Detector selection rule

Written 2026-10-27, before any variant other than the M2 configuration was scored.

Candidates: V0 (Presidio, spaCy `en_core_web_lg`), V1 (V0 plus the custom address recognizer,
the final M2 configuration), V2 (V1 with spaCy `en_core_web_trf` as the NLP engine) and V3 (V1 plus
PERSON spans from the Hugging Face NER model `dslim/bert-base-NER`). Only the M3 target entity types
are masked (PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, ADDRESS).

Choice data: ledger records in the M3 train and validation splits, the `challenge_dev` half of the
challenge set, and the clean original train and validation queries. Ledger test records and
`challenge_heldout` are scored only after the choice, for the chosen variant and for V1.

Rule:
1. Primary criterion: lowest residual leak rate on train+validation ledger records.
2. Constraints (a variant that fails any of them is not eligible):
   - over-redaction rate on clean train+validation queries at most V1's rate + 1.0 percentage point;
   - median detector latency at most 150 ms per record on the project PC;
   - `challenge_dev` detection recall at least V1's.
3. Ties on the primary criterion go to the faster variant.
4. If no eligible variant has a lower leak rate than V1, V1 stays (reported as a null result).
