# Synthetic transaction data

`transactions.csv` is generated locally by `python -m src.data_generator` or
automatically during model training. It contains only synthetic anonymous IDs
and behavioral risk features. It never contains card numbers, CVV, OTP, PIN,
bank passwords, UPI PIN, or other authentication secrets.

The random seed is fixed at 42. Fraud labels are sampled probabilistically from
multiple noisy risk factors, so they are imbalanced without being a simple copy
of the prototype rule engine.

`sample_transactions.json` contains three safe demonstration inputs.
