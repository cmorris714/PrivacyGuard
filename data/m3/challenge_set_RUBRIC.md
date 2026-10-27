# Challenge set: labeling rubric

The challenge set (`challenge_set.jsonl`, built by `m3/detector/challenge_set.py`) tests the detector on
PII written in shapes that the Faker ledger does not contain. Every value is fictional and hand-written.

## What counts as PII
| Type | Labeled | Not labeled |
|---|---|---|
| PERSON | a person's name: full name, given name on its own, or a surname used with a title (only the surname is in the span; the title is left out) | companies, brands and places, even when they share a word with a name (Chase bank, Wells Fargo, Mercedes-Benz, the city of Sydney) |
| EMAIL | the whole address, including `+tags`, dots, digits and subdomains | the word "email" |
| PHONE | the whole number, including country code, brackets, separators and any extension | other numbers (amounts, card prefixes, transaction ids) |
| ADDRESS | a street-level or PO box / rural route address, with unit, city, region and postcode when present (including a connecting "in"/"at" between street and city) | a city, region or country on its own |

## Values used
- Phone numbers: reserved fictional ranges only (North America 555-0100 to 555-0199, UK 020 7946 0xxx,
  Australia 02 5550 xxxx).
- E-mail domains: example.com, example.org, example.net and their subdomains.
- Street names: invented; postcodes are well-formed but the addresses do not exist.

## Structure
- 312 positive items in 39 format groups (8 items each, e.g. `intl_phone`, `po_box`, `plus_email`,
  `name_is_word`), 64 negative items in 8 groups of name-like or PII-like words.
- Each value sits in a new template sentence (not the four M2 templates) placed before or after a
  BANKING77 query from the M3 train split.
- Split 50/50 by category and format group into `challenge_dev` (used to pick the detector) and
  `challenge_heldout` (scored once, after the choice).
