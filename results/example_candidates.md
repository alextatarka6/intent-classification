# Example candidates: bert-base-uncased, k = 10, seed 4

Source: `results/predictions_k10_seed4.csv` (test_all, n = 3080). Rerun accuracy 0.7399, macro-F1 0.7312 (original sweep: 0.7312 / 0.7213).

Examples are sampled with RNG seed 0, one per true intent, from messages of at most 15 words. "test row" is the 0-based row in test_all.csv.

## a. Correct predictions

Filter: correct, and confidence >= 0.363 (the 75th percentile of confidence over the 2279 correct predictions).

- `I have a payment I didn't do` (test row 1094)  
  true: **card_payment_not_recognised**, confidence: 0.385
- `How do I reset a forgotten passcode, please?` (test row 1528)  
  true: **passcode_forgotten**, confidence: 0.441
- `I lost my phone and need help securing it.` (test row 1651)  
  true: **lost_or_stolen_phone**, confidence: 0.477
- `What do I need to do to activate my card?` (test row 2879)  
  true: **activate_my_card**, confidence: 0.526
- `Are there limits to top-ups?` (test row 744)  
  true: **top_up_limits**, confidence: 0.453
- `It wants me to verify the top up. Why?` (test row 2361)  
  true: **verify_top_up**, confidence: 0.533
- `Is it possible to have both a Visa and a Mastercard from you?` (test row 1309)  
  true: **visa_or_mastercard**, confidence: 0.591
- `What do I do if my card is expiring soon?` (test row 2928)  
  true: **card_about_to_expire**, confidence: 0.493

## b. Incorrect predictions

"Confusable" means the true and predicted intent names share their first word (e.g. card_*, transfer_*). 4 sampled from confusable errors, 4 from the remaining errors.

Confusable:

- `What does a pending transaction mean?` (test row 1636)  
  true: **pending_card_payment**, predicted: **pending_transfer**, confidence: 0.155
- `how long do money transfers take?` (test row 849)  
  true: **transfer_not_received_by_recipient**, predicted: **transfer_timing**, confidence: 0.286
- `I need a new card, my current card is nearly expired.` (test row 2935)  
  true: **card_about_to_expire**, predicted: **card_arrival**, confidence: 0.123
- `What methods can I use to top up my account?` (test row 2466)  
  true: **top_up_by_cash_or_cheque**, predicted: **top_up_limits**, confidence: 0.154

Other:

- `What's the deal with no cryptocurrency on your app?` (test row 2172)  
  true: **beneficiary_not_allowed**, predicted: **verify_top_up**, confidence: 0.104
- `I think my identity has been stolen, can you check on unauthorized charges?` (test row 1413)  
  true: **compromised_card**, predicted: **unable_to_verify_identity**, confidence: 0.221
- `why isnt top up working` (test row 2678)  
  true: **top_up_failed**, predicted: **pending_top_up**, confidence: 0.106
- `Something is wrong with my payment account since it has been declined` (test row 1818)  
  true: **declined_card_payment**, predicted: **reverted_card_payment?**, confidence: 0.103

## c. Top 10 confusion pairs (of 801 errors)

| # | true intent | predicted intent | count |
|---|---|---|---|
| 1 | why_verify_identity | verify_my_identity | 24 |
| 2 | top_up_reverted | pending_top_up | 19 |
| 3 | exchange_via_app | fiat_currency_support | 18 |
| 4 | top_up_failed | pending_top_up | 17 |
| 5 | supported_cards_and_currencies | fiat_currency_support | 16 |
| 6 | request_refund | Refund_not_showing_up | 15 |
| 7 | top_up_reverted | top_up_failed | 14 |
| 8 | disposable_card_limits | get_disposable_virtual_card | 13 |
| 9 | change_pin | get_physical_card | 11 |
| 10 | unable_to_verify_identity | why_verify_identity | 11 |

## d. 5 intents with the lowest recall

| intent | recall | correct / support |
|---|---|---|
| top_up_reverted | 0.050 | 2 / 40 |
| supported_cards_and_currencies | 0.175 | 7 / 40 |
| cash_withdrawal_not_recognised | 0.225 | 9 / 40 |
| beneficiary_not_allowed | 0.350 | 14 / 40 |
| why_verify_identity | 0.350 | 14 / 40 |
