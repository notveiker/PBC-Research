# Maple Factor and Valuation Analysis

This repository contains the core notebook and scripts for the Maple analysis workflow.

## Included

- `factor_analysis_MAPLE.ipynb` (main notebook)
- `maple_factor_analysis_charts.py` (factor chart generator)
- `maple_kalman_valuation.py` (supporting valuation script)
- `charts/data/` (local input CSV files used by notebook/scripts)

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API key (if running API-backed cells/scripts):

```bash
export ARTEMIS_API_KEY="your_key_here"
```

4. Open and run `factor_analysis_MAPLE.ipynb`.

## Notes

- Generated figures are written to `charts/figures/`.
- Generated tables/summaries are written to `outputs/tables/` and `outputs/summaries/`.
- This repo intentionally excludes pre-generated outputs so it stays lightweight and reproducible.
