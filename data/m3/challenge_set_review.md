# Challenge set: manual review

Reviewer: Claude Code. Date: 2026-10-27. Method: 40 items drawn at random (seed 553) from the 376 items
and read one by one against `challenge_set_RUBRIC.md`.

Checked for every item: the span text equals the labeled value; the label follows the rubric (type, extent,
title excluded); the value is fictional and uses only reserved phone ranges and example.* domains; the
sentence reads naturally; negatives contain no PII.

| Result | Items |
|---|---|
| Correct, no change needed | 40 of 40 |
| Labels changed | 0 |

Items reviewed: CH_0005, CH_0013, CH_0014, CH_0023, CH_0025, CH_0037, CH_0069, CH_0089, CH_0091,
CH_0093, CH_0095, CH_0106, CH_0116, CH_0140, CH_0175, CH_0196, CH_0197, CH_0209, CH_0212, CH_0218,
CH_0230, CH_0231, CH_0232, CH_0233, CH_0241, CH_0267, CH_0276, CH_0277, CH_0290, CH_0292, CH_0295,
CH_0304, CH_0317, CH_0342, CH_0351, CH_0352, CH_0368, CH_0373, CH_0374, CH_0375.

Observations: carrier queries are real BANKING77 training queries, so some items read as two unrelated
requests (for example a negative sentence placed after a query about a different topic); this does not
affect the detector test. Lowercase names and given-name-only items are deliberately harder than the
ledger's "My name is First Last." pattern.

Pending: confirmation by Carson Morris during the final review.
