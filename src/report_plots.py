# src/report_plots.py
import argparse, json, os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict) -> pd.DataFrame:
    raw_dir = cfg["data"]["raw_dir"]
    raw_file = cfg["data"].get("raw_file")
    csv_path = raw_file if raw_file else os.path.join(raw_dir, "pscomppars_sample.csv")
    df = pd.read_csv(csv_path)
    return df


def train_test(cfg: dict, df: pd.DataFrame):
    target = cfg["data"]["target_column"]
    features = cfg["features"]["include"]
    X = df[features].copy()
    y = df[target].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["model"]["test_size"],
        stratify=y,
        random_state=cfg["seed"],
    )
    return X_train, X_test, y_train, y_test


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def plot_calibration(models: dict, X_test, y_test, out_png: Path):
    plt.figure(figsize=(6, 5), dpi=140)
    for name, pipe in models.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    # perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction positive")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_pr_curves(models: dict, X_test, y_test, out_png: Path):
    plt.figure(figsize=(6, 5), dpi=140)
    for name, pipe in models.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_permutation_importance(models: dict, X_test, y_test, feature_names, out_dir: Path, topn: int = 12, seed: int = 42):
    for name, pipe in models.items():
        # permutation_importance works on the pipeline directly; it permutes raw columns
        r = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=10, random_state=seed, scoring="average_precision"
        )
        # Map importances back to raw features
        importances = pd.Series(r.importances_mean, index=feature_names)
        top = importances.sort_values(ascending=False).head(topn)
        plt.figure(figsize=(7, max(3, 0.35 * len(top))), dpi=140)
        top.sort_values().plot(kind="barh")
        plt.title(f"Permutation importance (top {len(top)}) — {name}")
        plt.xlabel("Mean importance (Δ AP)")
        plt.tight_layout()
        out_png = out_dir / f"perm_importance_{name}.png"
        plt.savefig(out_png)
        plt.close()


def write_metrics_table(metrics_json: Path, out_md: Path):
    if not metrics_json.exists():
        return
    m = json.loads(metrics_json.read_text())
    rows = []
    for model_name, blocks in m.items():
        cv = blocks.get("cv", {}).get("mean", {})
        hold = blocks.get("holdout", {})
        rows.append({
            "Model": model_name,
            "CV ROC AUC": f"{cv.get('roc_auc', float('nan')):.4f}" if "roc_auc" in cv else "n/a",
            "CV F1": f"{cv.get('f1', float('nan')):.4f}" if "f1" in cv else "n/a",
            "Holdout ROC AUC": f"{hold.get('roc_auc', float('nan')):.4f}" if "roc_auc" in hold else "n/a",
            "Holdout F1": f"{hold.get('f1', float('nan')):.4f}" if "f1" in hold else "n/a",
        })
    df = pd.DataFrame(rows, columns=["Model", "CV ROC AUC", "CV F1", "Holdout ROC AUC", "Holdout F1"])
    md = ["| " + " | ".join(df.columns) + " |",
          "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(x) for x in r.values) + " |")
    out_md.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg.get("seed", 42))

    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    plots_dir = artifacts_dir / "plots"
    ensure_dirs(artifacts_dir, model_dir, plots_dir)

    # Data and split
    df = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test(cfg, df)

    # Load trained pipelines
    log_path = model_dir / "baseline_logistic.joblib"
    xgb_path = model_dir / "model_xgboost.joblib"
    models = {}
    if log_path.exists():
        models["logistic"] = joblib.load(log_path)
    if xgb_path.exists():
        models["xgboost"] = joblib.load(xgb_path)

    if not models:
        raise FileNotFoundError("No trained model artifacts found in artifacts/models/")

    # Plots
    plot_calibration(models, X_test, y_test, plots_dir / "calibration.png")
    plot_pr_curves(models, X_test, y_test, plots_dir / "pr_curves.png")
    feature_names = cfg["features"]["include"]
    plot_permutation_importance(models, X_test, y_test, feature_names, plots_dir, topn=args.topn, seed=cfg.get("seed", 42))

    # Metrics table
    write_metrics_table(artifacts_dir / "metrics.json", plots_dir / "metrics_table.md")

    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
