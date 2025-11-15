# Exoplanet Candidate Validation Findings

## Summary
This report summarizes current model results and feature insights. Replace placeholders once you run training on real data.

## Metrics
```json
{
  "logistic": {
    "cv": {
      "mean": {
        "roc_auc": 0.8908329692455708,
        "precision": 1.0,
        "recall": 0.74105721411487,
        "f1": 0.8512615687610119
      },
      "std": {
        "roc_auc": 0.0070328795840469485,
        "precision": 0.0,
        "recall": 0.006048253840589901,
        "f1": 0.003995542048076473
      },
      "folds": 5
    },
    "holdout": {
      "roc_auc": 0.8844045934587955,
      "precision": 1.0,
      "recall": 0.7336268574573472,
      "f1": 0.8463492063492063
    }
  },
  "xgboost": {
    "cv": {
      "mean": {
        "roc_auc": 0.9898039930668571,
        "precision": 0.9648777520031947,
        "recall": 0.9375370682649496,
        "f1": 0.9509852339666315
      },
      "std": {
        "roc_auc": 0.0016407592794180352,
        "precision": 0.003415633577107062,
        "recall": 0.010234001789465583,
        "f1": 0.005490298027028028
      },
      "folds": 5
    },
    "holdout": {
      "roc_auc": 0.9889887585346314,
      "precision": 0.9656338028169014,
      "recall": 0.9433131535498074,
      "f1": 0.9543429844097996
    }
  }
}
```

## Notes
- Calibration and interpretability panels to be added.
- Replace sample data with NASA Exoplanet Archive extracts.
