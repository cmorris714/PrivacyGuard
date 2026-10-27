# Detector metric definitions

Each ledger record holds one injected synthetic value with known character offsets. A detector
span counts toward a value only when its entity type maps to the value's PII type
(PERSON -> PERSON, EMAIL_ADDRESS -> EMAIL, PHONE_NUMBER -> PHONE, LOCATION or ADDRESS -> ADDRESS).

| Metric | Definition |
|---|---|
| Detection recall | share of injected values that at least one matching-type span overlaps |
| Full-span recall | share of injected values whose every character lies inside matching-type spans |
| Residual leak rate | share of injected values whose complete text is still present (case-insensitive) after all target-type spans are merged and masked |
| Partial leak count | values that are not residual leaks but where a word of 3+ characters from the value survives masking (words that also occur in the original query are not counted) |
| Over-redaction rate | share of clean original queries (no injected value) with at least one target-type span |
| Latency | median and 95th-percentile milliseconds per record for `AnalyzerEngine.analyze()` on one record at a time with a warm model; masking adds well under 1 ms |

Why the M2 deck showed both 44 and 32 PERSON misses: 44 PERSON values had no overlapping
PERSON span (detection misses). 32 is the number whose full text was still present after masking
with every span the M2 pilot returned: for the other 12, a span of a different type (for example
LOCATION) covered the name, so it was masked even though it was not detected as a PERSON. A third
number, the 90 found-but-partly-covered PERSON values (found minus full-span), is where a partial
leak can happen. `person_44_breakdown.csv` lists every one of the 44 values and the spans that covered it.
