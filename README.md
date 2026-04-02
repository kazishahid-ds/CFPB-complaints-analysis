# CFPB Complaints Analysis

This repository contains the CFPB workflow currently documented for notebooks 01-03:

- data loading, preprocessing, and feature engineering
- exploratory data analysis and profiling
- baseline and tree-based time-series forecasting

Notebook 04 workflows exist in the repository but are intentionally excluded from this README for now.

## Key Data Files

- `data/complaints.csv`
	- raw CFPB complaint data source
- `data/processed/credit_reporting_monthly_data.csv`
	- monthly national credit-reporting aggregates (count, total, share)
- `data/processed/credit_reporting_monthly_south_data.csv`
	- monthly South-region credit-reporting aggregates
- `data/processed/credit_reporting_monthly_non_south_data.csv`
	- monthly Non-South-region credit-reporting aggregates
- `data/complaints_cleaned.csv`
	- cleaned dataset exported by notebook 01 for downstream EDA and diagnostics
- `data/processed/credit_card_dataset.parquet`
	- trimmed credit-card subset exported by notebook 01 for downstream analysis

## Notebook Scope (01-03)

- `notebooks/01_cfpb_data_overview_preprocessing_eda.ipynb`
	- loads full CFPB data, engineers time/geography features, and performs product-level diagnostics
	- creates monthly credit-reporting aggregates (counts + share) and exports cleaned artifacts
	- exports trimmed credit-card subset for downstream notebooks
- `notebooks/02_credit_reporting_eda.ipynb`
	- runs EDA/profiling on the cleaned dataset and then on the Credit Reporting subset (`df_cr`)
	- includes South vs Others segment diagnostics for Credit Reporting complaints
- `notebooks/03a_cfpb_credit_reporting_tsf_monthly_baseline.ipynb`
	- classical monthly share forecasting baselines (naive family, ETS, ARIMA/SARIMA, Theta)
	- Section 12 regional comparison is Theta-only (South vs Non-South)
- `notebooks/03b_cfpb_creditreporting_monthly_tree.ipynb`
	- tree-based monthly forecasting workflow and comparisons against baseline family

## Reports

- `reports/`
	- rendered HTML/PDF outputs by run date
- `reports/2026-03-31/`
	- latest exported HTML/PDF reports for notebooks 01, 02, and 03a

## Environment Setup

Option 1: install from pip requirements

```bash
pip install -r requirements.txt
```

Option 2: use the conda environment file

```bash
conda env create -f environments/env_ts-forecast.yml
conda activate tmu_capstone
```

## Recommended Run Order (01-03)

1. `notebooks/01_cfpb_data_overview_preprocessing_eda.ipynb`
2. `notebooks/02_credit_reporting_eda.ipynb`
3. `notebooks/03a_cfpb_credit_reporting_tsf_monthly_baseline.ipynb`
4. `notebooks/03b_cfpb_creditreporting_monthly_tree.ipynb`

## Current Status

- 01-03 workflow documentation is current in this README
- notebook 03a is aligned to credit-reporting share targets with Theta as the retained regional baseline in Section 12
- notebook reports for 01, 02, and 03a are available under `reports/2026-03-31/`
- notebook 04 content is deferred from README coverage until explicitly requested
