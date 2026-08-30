# Third-party notices

AI RiskGuard includes a processed sample and model artifacts derived from the
fully synthetic credit-card transaction data associated with IBM's TabFormer
research project. No TabFormer transformer model or source code is incorporated
into AI RiskGuard.

## IBM TabFormer synthetic credit-card transaction data

- Project and data source: [IBM/TabFormer](https://github.com/IBM/TabFormer)
- Downloadable dataset mirror: [Credit Card Transactions on Kaggle](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)
- Data-generation paper: Erik R. Altman, [“Synthesizing Credit Card Transactions”](https://arxiv.org/abs/1910.03033), 2019
- TabFormer paper: Inkit Padhi et al., [“Tabular Transformers for Modeling Multivariate Time Series”](https://arxiv.org/abs/2011.01843), ICASSP 2021
- Upstream license: [Apache License 2.0](https://github.com/IBM/TabFormer/blob/main/LICENSE)
- Redistributed license text: [local Apache-2.0 copy](LICENSES/IBM-TabFormer-Apache-2.0.txt)

The upstream repository and dataset publish an Apache License 2.0 notice. Users
who download or redistribute upstream or derived material are responsible for
retaining applicable copyright, license, attribution, and modification notices.
The Apache License text and its warranty/liability terms govern the upstream work;
nothing in this notice changes those terms.

## Modifications made for AI RiskGuard

AI RiskGuard does not redistribute the approximately 276 MB compressed archive
or its approximately 2.35 GB extracted transaction CSV. Its adapter makes the
following documented transformations:

- excludes source rows with non-positive amounts;
- converts positive source USD amounts using a fixed educational factor of 83.0
  for INR-interface compatibility;
- calculates causal history features using only current or earlier positive
  payments for the same synthetic user;
- selects a deterministic, label-independent 500,000-row sample;
- removes raw card/profile and merchant-location fields not required by the model;
  and
- renames and validates fields for AI RiskGuard's Logistic Regression and Random
  Forest training pipeline.

The source records are fully synthetic. They are not live financial records and
not anonymized records of real people. IBM, the TabFormer authors, Kaggle, and
Razorpay do not sponsor, endorse, or maintain AI RiskGuard. Their names are used
only for source attribution and use-case description. No third-party logo is used.
