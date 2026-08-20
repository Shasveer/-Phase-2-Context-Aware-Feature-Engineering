# Context-Aware Feature Engineering

Milestone 2 transforms the cleaned customer data from Milestone 1 into business-
intelligent predictors for SME financial behaviour in South Africa. The pipeline
keeps source columns intact, adds documented features, and applies safeguards
against division errors, target leakage, and multicollinearity.

## Quick Start

The repository expects the cleaned input at:

```text
data/processed/cleaned_customers.csv
```

Run the complete pipeline from the repository root:

```bash
python feature_engineer.py
```

The script writes:

```text
data/processed/engineered_features.csv
```

The only runtime dependencies are Python 3.10 or newer, `pandas`, and `numpy`.
Install them with:

```bash
python -m pip install pandas numpy
```

## Input Data

The feature engineer is tolerant of common Milestone 1 column names. The
recommended schema includes:

| Column | Purpose |
| --- | --- |
| `customer_id` | Stable customer identifier, preserved unchanged. |
| `region` | South African region used for contextual benchmarks. `province` and `location` are also accepted. |
| `debt` | Customer debt. `monthly_debt` and `total_debt` are also accepted. |
| `income` | Customer income. `monthly_income` and `annual_income` are also accepted. |
| `load_shedding_hours` | Load-shedding exposure in hours on a weekly cycle. |
| `support_tickets` | Count of customer support interactions, including structural zeros. |
| `township_flag` | Existing infrastructure-context signal, preserved but not used to create a punitive proxy. |
| `ethical_metadata` | Cleaning and fairness notes from Milestone 1, preserved in the output. |

Additional source columns are retained automatically. Missing optional columns
cause only their dependent feature family to be skipped; missing required input
artifacts cause the script to fail clearly.

## Generated Features

### Financial strain

`financial_strain_ratio` is calculated as:

```text
debt / income
```

Non-numeric values and missing values are treated as zero. When income is zero
or negative, the ratio is set to zero rather than producing infinity or a
misleading undefined value. Its metadata explains that it is a debt-
overcommitment indicator, not a verdict about a customer's worthiness.

### Load-shedding impact

`load_shedding_sin` and `load_shedding_cos` encode weekly timing using:

```text
sin(2 * pi * hours / 168)
cos(2 * pi * hours / 168)
```

The 168-hour period captures a weekly cycle and avoids treating the start and
end of the cycle as artificially distant. Together, the features represent
infrastructure-related transaction risk without assuming that risk increases
linearly with the hour value.

### Regional benchmarks

`regional_income_benchmark` is a region-level mean calculated from training
rows only. `regional_sample_count` records the amount of training evidence
behind each benchmark. Regions with fewer than 50 training observations are
recorded in `feature_metadata` as a warning so downstream users can apply
appropriate caution.

For inference, pass precomputed training statistics rather than recalculating
them on inference data:

```python
from feature_engineer import FeatureEngineer

inference_engineer = FeatureEngineer(inference_df)
inference_engineer.regional_benchmarks(train_stats)
```

`train_stats` may be a dataframe containing `region`,
`regional_income_benchmark`, and optionally `regional_sample_count`, or a
mapping from region to those statistic names. This separation prevents
validation and production observations from influencing the benchmark.

### Zero-inflated support demand

`support_tickets_zero_flag` identifies whether any support was required.
`support_tickets_log1p` represents the positive ticket volume while compressing
large values. This two-part representation preserves the difference between
structural zero demand and increasing service intensity.

## Metadata and Safeguards

Every generated feature has a business rationale in the
`FeatureEngineer.feature_metadata` dictionary. After `run_full_engineering()`,
the same dictionary is also available at `result.attrs["feature_metadata"]`.
Existing dataframe attributes are retained, and all source columns, including
ethical cleaning metadata, remain in the engineered dataframe.

The pipeline applies two multicollinearity checks:

1. Engineered numeric features with absolute pairwise correlation greater than
	`0.85` are rejected when they are ancillary duplicates.
2. For sufficiently large datasets, features are removed iteratively until the
	reported maximum VIF is at most `5.0`.

The core semantic outputs are retained so that the financial, infrastructure,
regional, and zero-inflated business concepts remain inspectable. Very small
fixtures do not produce a meaningful VIF estimate; these are reported as
`1.0` rather than triggering unstable removals.

## Python API

```python
import pandas as pd
from feature_engineer import FeatureEngineer

cleaned = pd.read_csv("data/processed/cleaned_customers.csv")
engineer = FeatureEngineer(cleaned)
engineered = engineer.run_full_engineering()

print(engineer)
print(engineer.feature_metadata)
```

The summary has the form:

```text
Created 6 new features | Max VIF: 1.0
```

Individual stages can also be called when a workflow needs to inject training
statistics or inspect intermediate results:

```python
engineer.create_financial_strain_ratio()
engineer.encode_load_shedding_impact()
engineer.regional_benchmarks(train_stats=train_stats)
engineer.encode_support_tickets()
engineer.validate_multicollinearity()
```

## Ethical Design Notes

Infrastructure context is included to explain operational conditions that can
affect transaction success, not to penalise a township or region. Regional
benchmarks are descriptive context and are deliberately isolated to training
data. Small-region warnings make weak evidence visible instead of presenting a
small sample as a universal economic truth. Downstream model owners should
still test subgroup performance, calibration, and disparate impact before using
these features for decisions about access to financial services.

## Repository Layout

```text
.
├── feature_engineer.py
├── README.md
└── data/
	 └── processed/
		  ├── cleaned_customers.csv
		  └── engineered_features.csv
```