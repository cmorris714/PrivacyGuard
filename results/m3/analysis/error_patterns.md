# Error patterns (test set; frozen classifier unless noted)

## Confusable intent pairs (10 most frequent pairs, errors on raw text)
- Count: 69 of 326 errors on C1 raw (21.2%)
- Examples: Is there a way to know when my card will arrive? My phone is 202-555-0112. -> card_arrival predicted as card_delivery_estimate || My name is Kiara Solomon. When will I get my card? -> card_arrival predicted as card_delivery_estimate
- Likely cause: intents with near-identical wording (e.g. pending vs failed, card arrival vs delivery estimate)
- Mitigation: now: route low-confidence predictions to review; later: merge or relabel pairs with the business owner

## Errors on the 10 smallest training intents (they are 13.0% of test records)
- Count: 53 of 326 errors on C1 raw (16.3%)
- Examples: My email is jesse46@example.com. My entire gym bag, including my wallet, was stolen out of my locker today. Everything in my wallet is gone - how do I block the card to make it can't be used? -> lost_or_stolen_card predicted as cash_withdrawal_not_recognised || Is a copy of the police report necessary for completing the report process? My phone is 202-555-0175. -> lost_or_stolen_card predicted as verify_my_identity
- Likely cause: fewer training examples (28-78 in the M3 train split)
- Mitigation: now: report per-intent recall with Wilson CIs; later: targeted examples or class weighting

## New errors after generic masking that predict a contact/identity intent
- Count: 12 of 214 records right on raw, wrong after generic (frozen model) (5.6%)
- Examples: Can I make transfers before identity verification? My name is [REDACTED]. -> why_verify_identity predicted as verify_my_identity || My email is [REDACTED]. Do I have to do the identity check? -> why_verify_identity predicted as verify_my_identity
- Likely cause: the leftover contact sentence ('My phone is [PHONE].' / 'My email is .') reads like a request about personal details
- Mitigation: now: retrain on sanitized text (matched protocol); later: drop the dangling contact sentence before classification

## Other new errors after generic masking
- Count: 202 of 214 records right on raw, wrong after generic (frozen model) (94.4%)
- Examples: My email is [REDACTED]. How did you come up with your exchange rates? -> exchange_rate predicted as card_payment_wrong_exchange_rate || Please advise what is the exchange rate My email is [REDACTED]. -> exchange_rate predicted as card_payment_wrong_exchange_rate
- Likely cause: the masked token shifts a mean-pooled sentence embedding; 64% of these records were already below the review threshold (confidence < 0.64) on raw text
- Mitigation: now: retrain on sanitized text and use the confidence gate; later: more varied contact templates

## New errors after delete masking that predict a contact/identity intent
- Count: 6 of 65 records right on raw, wrong after delete (frozen model) (9.2%)
- Examples: My phone is . What should I do if my card is missing? -> lost_or_stolen_card predicted as lost_or_stolen_phone || My email is . It doesn't let me verify my identity. -> unable_to_verify_identity predicted as why_verify_identity
- Likely cause: the leftover contact sentence ('My phone is [PHONE].' / 'My email is .') reads like a request about personal details
- Mitigation: now: retrain on sanitized text (matched protocol); later: drop the dangling contact sentence before classification

## Other new errors after delete masking
- Count: 59 of 65 records right on raw, wrong after delete (frozen model) (90.8%)
- Examples: My email is . Why won't my card show up on the app? -> card_linking predicted as card_not_working || My email is . The app doesn't show the card I received. -> card_linking predicted as card_arrival
- Likely cause: the masked token shifts a mean-pooled sentence embedding; 81% of these records were already below the review threshold (confidence < 0.64) on raw text
- Mitigation: now: retrain on sanitized text and use the confidence gate; later: more varied contact templates

## Detector masks words in the query itself (place names, brands) and the prediction changes
- Count: 41 of 91 test records whose clean query has a detection (45.1%)
- Examples: how long does it take to post atm with [REDACTED] My phone is [REDACTED]. -> pending_cash_withdrawal predicted as pending_transfer || My email is [REDACTED]. Does delivery to the [REDACTED] take long? -> card_delivery_estimate predicted as transfer_not_received_by_recipient
- Likely cause: country/city names or capitalised words detected as LOCATION/PERSON inside the banking question
- Mitigation: now: allow-list of currency/country terms for the intent model's input; later: entity-aware masking of query text

## Detector: weakest challenge_heldout format groups

- ADDRESS/uk_address: 3/4 found (V1: 75%)
- PERSON/first_name_only: 3/4 found (V1: 100%)
- PERSON/south_asian: 3/4 found (V1: 75%)
- ADDRESS/ca_address: 4/4 found (V1: 75%)
- ADDRESS/no_commas: 4/4 found (V1: 100%)
- ADDRESS/no_zip: 4/4 found (V1: 50%)
