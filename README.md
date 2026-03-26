# CFPB Complaints Analysis

This repository contains the end-to-end workflow used for CFPB complaints analysis:

- stratified sampling and data preparation
- exploratory data analysis and profiling
- quarterly time-series forecasting baselines
- tree-based and neural forecasting experiments
- regional-segment forecasting

## Key Data Files

- `data/processed/full_dataset.parquet`
	- primary dataset used by 03a/03b/03c/04 forecasting workflows
- `data/processed/cfpb_sample_300k.parquet`
	- stratified sample used by EDA workflow (02)

## Project Structure


- `notebooks/01_cfpb_stratified_sampling_fixed.ipynb`
	- builds the sample pipeline and time features
	- exports `data/processed/cfpb_sample_300k.parquet`
- `notebooks/01_cfpb_stratified_sampling.ipynb`
	- earlier/original variant of the 01 workflow
- `notebooks/02_cfpb_eda_analysis.ipynb`
	- EDA, descriptive visuals, and profiling reports
- `notebooks/03a_cfpb_time_series_baseline.ipynb`
	- classical baseline forecasting workflow (ETS/ARIMA/SARIMA/Theta/naive family)
- `notebooks/03b_cfpb_time_series_tree.ipynb`
	- leakage-safe RF + native XGBoost multi-horizon workflow
	- includes tuning, baseline comparison, and exports
- `notebooks/03c_cfpb_time_series_deep_learning.ipynb`
	- NeuralProphet-based forecasting experiments
- `notebooks/04_cfpb_time_series_south_vs_others.ipynb`
	- segmented ETS comparison and forward forecast (South vs Others)

Output locations:

- `reports/`
	- rendered HTML reports and figures
- `reports/2026-03-25/tables/`
	- exported modeling tables from 03b/04
- `models/`
	- serialized model artifacts from 03c and 04

## Environment Setup

Option 1: install from pip requirements

```bash
pip install -r requirements.txt
```

Option 2: use conda environment file

```bash
conda env create -f environments/env_ts-forecast.yml
conda activate ts-forecast
```

## Running the Workflows

Recommended notebook order:

1. `01_cfpb_stratified_sampling_fixed.ipynb`
2. `02_cfpb_eda_analysis.ipynb`
3. `03a_cfpb_time_series_baseline.ipynb`
4. `03b_cfpb_time_series_tree.ipynb`
5. `03c_cfpb_time_series_deep_learning.ipynb`
6. `04_cfpb_time_series_south_vs_others.ipynb`

## Current Status

- Baseline, tree, and NeuralProphet workflows are implemented
- Segmented South vs Others ETS workflow is implemented in 04
- 03b and 04 export comparable model tables for reporting
- EDA and profile reports are available under `reports/`
