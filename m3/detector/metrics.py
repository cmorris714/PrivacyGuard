"""Detector evaluation metrics (standard library only).

A span is (start, end, entity_type[, score]); ends are exclusive.
Presidio entity types map to project PII types through ENTITY_TO_TYPE.
"""
import re

from m3.common import norm, replacement

ENTITY_TO_TYPE = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "LOCATION": "ADDRESS",
    "ADDRESS": "ADDRESS",
}
TARGET_ENTITIES = tuple(ENTITY_TO_TYPE)


def merge_spans(spans):
    """Merge overlapping spans; the merged span takes the type of its highest-scoring member."""
    out = []
    for sp in sorted(spans, key=lambda x: (x[0], -x[1])):
        s, e, t = sp[0], sp[1], sp[2]
        sc = sp[3] if len(sp) > 3 else 1.0
        if out and s < out[-1][1]:
            ps, pe, pt, psc = out[-1]
            out[-1] = (ps, max(pe, e), t if sc > psc else pt, max(sc, psc))
        else:
            out.append((s, e, t, sc))
    return [(s, e, t) for s, e, t, _ in out]


def mask(text, spans, method):
    """Replace merged spans with the method's token, then normalize whitespace."""
    for s, e, t in sorted(merge_spans(spans), reverse=True):
        text = text[:s] + replacement(method, ENTITY_TO_TYPE.get(t, t)) + text[e:]
    return norm(text)


def filter_target(spans, entities=TARGET_ENTITIES):
    return [sp for sp in spans if sp[2] in entities]


def matching(spans, pii_type, start, end):
    return [sp for sp in spans if ENTITY_TO_TYPE.get(sp[2]) == pii_type and sp[0] < end and sp[1] > start]


def is_found(spans, pii_type, start, end):
    """Detection: at least one span of a matching type overlaps the injected value."""
    return bool(matching(spans, pii_type, start, end))


def is_full_span(spans, pii_type, start, end):
    """Full span: every character of the value is covered by matching-type spans."""
    covered = set()
    for s, e, *_ in matching(spans, pii_type, start, end):
        covered.update(range(max(s, start), min(e, end)))
    return len(covered) == end - start


def is_residual_leak(masked_text, value):
    """Residual leak: the complete value still appears (case-insensitive) after masking."""
    return value.casefold() in masked_text.casefold()


def value_tokens(value):
    return [t for t in re.findall(r"\w+", value) if len(t) >= 3]


def is_partial_leak(masked_text, value, context_text=""):
    """Partial leak: not a residual leak, but a token (>= 3 chars) of the value survives masking.

    Tokens that also occur in context_text (the original query without PII) are ignored,
    because their presence does not come from the injected value.
    """
    if is_residual_leak(masked_text, value):
        return False
    low, ctx = masked_text.casefold(), context_text.casefold()
    for tok in value_tokens(value):
        pat = r"(?<!\w)" + re.escape(tok.casefold()) + r"(?!\w)"
        if re.search(pat, low) and not re.search(pat, ctx):
            return True
    return False


def evaluate_record(text, spans, pii_type, start, end, value, context_text="", method="generic"):
    """All per-record metrics for one injected value. spans should already be filtered as intended."""
    masked = mask(text, spans, method)
    return {
        "found": is_found(spans, pii_type, start, end),
        "full_span": is_full_span(spans, pii_type, start, end),
        "residual_leak": is_residual_leak(masked, value),
        "partial_leak": is_partial_leak(masked, value, context_text),
        "masked": masked,
    }


def over_redacted(spans):
    """Clean text (no injected PII) with at least one detection of a target type."""
    return bool(filter_target(spans))
