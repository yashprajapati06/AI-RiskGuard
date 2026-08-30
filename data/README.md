# IBM TabFormer-derived synthetic transaction data

`transactions.csv` is derived from the fully synthetic credit-card transaction
data published by the [IBM/TabFormer repository](https://github.com/IBM/TabFormer)
and the [Credit Card Transactions Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions).
It contains simulated activity—not live records and not anonymized real cardholder
transactions. IBM and the source authors do not sponsor or endorse AI RiskGuard.

## Processing and provenance

The source archive contains 24,386,900 transaction rows. The adapter:

1. removes 1,264,896 refunds, credits, and zero-value events where `Amount <= 0`;
2. derives causal history from the remaining 23,122,004 positive payments;
3. takes a deterministic, uniform, label-independent bottom-k sample with seed 42;
4. writes 500,000 chronologically ordered rows containing 635 fraud labels; and
5. records its policy and output SHA-256 in `dataset_metadata.json`.

The positive-only filter and sampler never inspect or rebalance `Is Fraud?`, so
the final 0.127% fraud rate is the sample's natural imbalance. Historical features
are calculated before sampling so retained rows can use eligible earlier events.

Source USD amounts are multiplied by the fixed educational normalization factor
`83.0` for compatibility with the INR interface. This is not a live exchange
rate or financial conversion service.

## Feature derivation

- `previous_failed_txns` counts earlier same-user rows with nonblank `Errors?`
  during the previous 24 hours, excludes the current row, and is capped at 12.
  It is a processing-error proxy, not confirmed failed authentication.
- `txn_count_10min` counts the current and earlier same-user payments in the
  trailing 10-minute window and is capped at 20.
- `avg_user_transaction_amount` is the expanding mean of strictly earlier
  positive payments, with the current amount used only for the first-row fallback.
- `location_change` compares current and immediately previous known merchant
  locations; it is not a cardholder GPS or device-location check.
- `account_age_days` uses the matched synthetic card's account-open date.
- `international_transaction` is a merchant-state heuristic. Blank/unknown state
  maps to 0; it is not a definitive cross-border indicator.
- `event_timestamp` is used for ordering and the chronological split only.
- `transaction_id`, `user_id`, and `merchant_id` are synthetic identifiers used
  for grouping/storage only. `fraud` is the target only. None enters the model.

The source does not provide trustworthy equivalents for device type, new-device
status, the UI's UPI/Card/Wallet/NetBanking categories, or an independent merchant
risk score. These fields are not fabricated for training. `is_new_device` and
`merchant_risk_score` remain manual rule-only inputs; `payment_method` and
`device_type` are manual context/storage fields and do not affect the v2 model or
rules. Merchant risk is never derived from fraud labels.

## Privacy minimization

The adapter does not copy full card numbers, CVV, PIN, expiry, person names, email
addresses, postal addresses, or raw merchant locations into `transactions.csv`.
It reads only the synthetic card index and account-open date needed for the
account-age join. Keep the approximately 276 MB source ZIP and approximately
2.35 GB extracted transaction CSV outside version control; repository ignore
rules cover common raw locations.

## Reproduce

Download the Kaggle ZIP containing both the transactions and cards CSVs, then run
from the project root:

```powershell
python -m src.tabformer_adapter "C:\path\to\credit-card-transactions.zip" --target-rows 500000 --chunksize 200000 --seed 42
python -m src.train_model
```

The adapter defaults to `data/transactions.csv`. The model then uses the earliest
400,000 rows for training and the latest 100,000 rows for locked testing. It tunes
on 60,000 training-only rows and refits each candidate on all 400,000 training
rows. The repository `.tgz`/Git LFS pointer is not accepted by the ZIP adapter.

`sample_transactions.json` contains safe manual demonstration inputs.
`src.data_generator` remains a small offline fallback for testing or recovery,
but artifacts trained from it must not be described as TabFormer-derived.

The upstream source supplies an Apache License 2.0 notice. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) before redistributing source
or derived material.
