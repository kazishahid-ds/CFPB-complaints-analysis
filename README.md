# CFPB Complaints Analysis

This repository contains the CFPB complaint-analysis workflow currently maintained for the **01a-02b** notebook sequence:

- data loading, preprocessing, and feature engineering
- exploratory data analysis and profiling
- baseline and tree-based time-series forecasting for credit-reporting share

Supplemental notebooks (`02c`, `04`) are present but not the primary submission path.

## Key Data Files

- `data/complaints.csv`
  - raw CFPB complaint data source
- `data/complaints_cleaned.parquet`
  - cleaned complaint dataset exported for downstream EDA/modeling
- `data/processed/credit_reporting_timeseries_all.parquet`
  - monthly aggregate target series and covariates used in `02a` and `02b`
- `data/processed/credit_reporting_timeseries_south.parquet`
  - South segment monthly aggregate series
- `data/processed/credit_reporting_timeseries_non_south.parquet`
  - Non-South segment monthly aggregate series
- `data/processed/credit_reporting_sample_500k.parquet`
  - stratified sample used for exploratory diagnostics/profiling

## Notebook Scope (Primary)

- `notebooks/01a_cfpb_data_overview_preprocessing_eda.ipynb`
  - loads CFPB data, engineers temporal/geographic fields, and creates processed artifacts
  - builds monthly credit-reporting aggregate series for forecasting notebooks
- `notebooks/01b_credit_reporting_eda.ipynb`
  - credit-reporting focused EDA and stratified-sample diagnostics
- `notebooks/02a_credit reporting_monthly_tsf_baseline models.ipynb`
  - classical monthly baseline forecasting (Naive/ETS/ARIMA/SARIMA/Theta/FFT)
  - exports baseline results artifact for downstream comparison
- `notebooks/02b_credit reporting_monthly_tsf_tree based models.ipynb`
  - leakage-safe tree-based forecasting workflow (RF/XGB) with direct comparison to `02a` baselines

## Supplemental Notebooks (Optional)

- `notebooks/02c_credit reporting_monthly_tsf_deep learning models.ipynb`
- `notebooks/04_cfpb_time_series_south_vs_others.ipynb`

## Reports and Artifacts

- `reports/cfpb_credit_reporting_ydata_profile.html`
  - profile report for exploratory diagnostics
- `reports/artifacts/02a_creditreporting_monthly_baseline_results.csv`
  - canonical exported baseline result table from `02a`
- `reports/2026-03-26/tables/`
  - tree-vs-baseline comparison tables and diagnostic exports (including `02b_creditreporting_monthly_*`)
- `reports/final/`
  - generated final report deliverables (`.docx`)

## Environment Setup

Option 1: install from pip requirements

```bash
pip install -r requirements.txt
```

Option 2: use conda environment file

```bash
conda env create -f environments/env_ts-forecast.yml
conda activate tmu_capstone
```

## Quick Start (01a -> 02b)

Run the primary notebook workflow in order.

1. Launch Jupyter from the project root:

```bash
jupyter lab
```

2. Open and run the notebooks in this sequence:

- `notebooks/01a_cfpb_data_overview_preprocessing_eda.ipynb`
- `notebooks/01b_credit_reporting_eda.ipynb`
- `notebooks/02a_credit reporting_monthly_tsf_baseline models.ipynb`
- `notebooks/02b_credit reporting_monthly_tsf_tree based models.ipynb`

3. Confirm expected outputs after execution:

- baseline artifact: `reports/artifacts/02a_creditreporting_monthly_baseline_results.csv`
- tree comparison tables: `reports/2026-03-26/tables/02b_creditreporting_monthly_*.csv`
- final report docs (if generated): `reports/final/*.docx`

## Recommended Run Order (Primary Path)

1. `notebooks/01a_cfpb_data_overview_preprocessing_eda.ipynb`
2. `notebooks/01b_credit_reporting_eda.ipynb`
3. `notebooks/02a_credit reporting_monthly_tsf_baseline models.ipynb`
4. `notebooks/02b_credit reporting_monthly_tsf_tree based models.ipynb`

## Current Status (2026-04-15)

- primary workflow docs are aligned to 01a-02b naming and paths
- `02a` and `02b` section labels were standardized (numbered subsection style)
- baseline artifact export from `02a` and tree comparison tables from `02b` are available
- final report Word outputs are available under `reports/final/`
