# Exoplanet Candidate Validation

## Overview
This project builds a small, reproducible pipeline to classify exoplanet *candidates* as likely confirmed or unlikely based on orbital and stellar parameters. It combines transparent modeling with validation and interpretability, producing a simple Deepnote app and a short literature summary.

## Functionality
- Fetch and version NASA Exoplanet Archive tables and optional Gaia cross-matches.
- Clean and join datasets, engineer features, and persist schemas.
- Train baseline (logistic regression) and tree-based models.
- Evaluate with ROC-AUC, precision, recall, F1, and calibration.
- Explain results with permutation importance, SHAP, and partial dependence.
- Export a lightweight PDF or Markdown report.
- Run an app notebook to score new candidate files and view interpretability panels.

## Key Insights
- The model uses physical and observational features to separate likely confirmations from low-likelihood candidates.
- Emphasis is on rigor and reproducibility. All inputs are hashed, and environments are pinned.
- Interpretability is central. The repo favors readable models and calibrated probabilities.

## Results
- Placeholder: Add metrics after first training run.
- Example targets: reliable probability calibration, ROC-AUC above a naive baseline, and clear top features.

## Tech Stack
Python, pandas, numpy, scikit-learn, XGBoost, SHAP, matplotlib, plotly, astroquery, Deepnote

## Installation
```bash
git clone <your-repo-url>.git
cd exoplanet-validation
make setup
```

## How to run
```bash
# 1) Fetch data from NASA Exoplanet Archive and write hashes
make fetch

# 2) Train models and save artifacts
make train

# 3) Evaluate and generate figures
make evaluate

# 4) Launch app notebook locally or in Deepnote
make app

# 5) Build a short report for sharing
make report
```

## Data
- Primary: NASA Exoplanet Archive tables such as `pscomppars` and mission catalogs.
- Optional: Gaia DR3 cross-matches for stellar properties.
- Data are stored in `data/raw` and `data/processed`. Hashes are tracked in `artifacts/data_hash.json`.

## Methods
- Feature engineering: orbital period, estimated transit depth proxies, equilibrium temperature category, host star temperature and radius, system multiplicity, discovery method indicators.
- Models: logistic regression baseline and a tree-based model (Random Forest or XGBoost).
- Validation: stratified CV, holdout set, calibration curve, permutation importance, partial dependence, SHAP review.

## Reproducibility
- `requirements.txt` pins key versions.
- `Makefile` standardizes all commands.
- `artifacts/data_hash.json` records input hashes.
- `src/` provides modular, testable components.
- Random seeds are fixed for repeatability.

## Deepnote
Use `notebooks/app.ipynb` as the main app. Expose inputs for file upload, run scoring, show top features and calibration, and allow report export.

## Future Improvements
- Add TESS and Kepler candidate tables for robustness tests.
- Add Bayesian calibration for probability tuning.
- Expand interpretability to include counterfactual explanations.

## Author
**Rhiannon Fillingham**
