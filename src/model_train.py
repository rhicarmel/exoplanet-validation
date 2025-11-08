import argparse, json, os, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier
import joblib


# -----------------------------
# Utility Functions
# -----------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_data(raw_dir, target_column):
    df = pd.read_csv(os.path.join(raw_dir, "pscomppars_sample.csv"))
    if target_column not in df.columns:
        df[target_column] = 1
    return df


def build_pipeline(model_name, numeric_features, categorical_features):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    if model_name == "logistic":
        model = LogisticRegression(max_iter=200, random_state=42)
    else:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    return Pipeline(steps=[("pre", pre), ("model", model)])


# -----------------------------
# Evaluation Functions
# -----------------------------
def eval_holdout(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    pred = (proba > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    return {
        "roc_auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def eval_cv(pipe, X, y, folds=5, seed=42):
    max_folds = int(min(folds, y.value_counts().min()))
    if max_folds < 2:
        raise ValueError("Cross-validation requires at least 2 samples in the minority class.")

    skf = StratifiedKFold(n_splits=max_folds, shuffle=True, random_state=seed)
    scores = []

    for tr, te in skf.split(X, y):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y.iloc[te], proba)
        pred = (proba > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y.iloc[te], pred, average="binary", zero_division=0
        )
        scores.append({"roc_auc": auc, "precision": prec, "recall": rec, "f1": f1})

    mean = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    std = {k: float(np.std([s[k] for s in scores], ddof=1)) for k in scores[0]}
    return {"mean": mean, "std": std, "folds": max_folds}


# -----------------------------
# Main Routine
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.get("seed", 42))

    df = load_data(cfg["data"]["raw_dir"], cfg["data"]["target_column"])
    target = cfg["data"]["target_column"]
    features = cfg["features"]["include"]

    X = df[features]
    y = df[target]

    numeric_features = [c for c in features if df[c].dtype != "object"]
    categorical_features = [c for c in features if df[c].dtype == "object"]

    baseline = build_pipeline("logistic", numeric_features, categorical_features)
    tree = build_pipeline("xgboost", numeric_features, categorical_features)

    # Split for holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["model"]["test_size"],
        stratify=y,
        random_state=cfg["seed"]
    )

    if np.unique(y_train).size < 2:
        raise ValueError("Training data has one class after split. Increase data or change test_size.")

    # Read CV folds from config
    cv_folds = int(cfg["model"].get("cv_folds", 5))
    seed = cfg.get("seed", 42)

    # Evaluate logistic
    log_cv = eval_cv(baseline, X, y, folds=cv_folds, seed=seed)
    log_hold = eval_holdout(baseline, X_train, X_test, y_train, y_test)

    # Evaluate XGBoost
    xgb_cv = eval_cv(tree, X, y, folds=cv_folds, seed=seed)
    xgb_hold = eval_holdout(tree, X_train, X_test, y_train, y_test)

    metrics = {
        "logistic": {"cv": log_cv, "holdout": log_hold},
        "xgboost": {"cv": xgb_cv, "holdout": xgb_hold},
    }

    # Save results
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, model_dir / "baseline_logistic.joblib")
    joblib.dump(tree, model_dir / "model_xgboost.joblib")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
