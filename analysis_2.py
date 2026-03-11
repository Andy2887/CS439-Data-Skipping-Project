"""
analysis_2.py — Gradient Boosted Regression & SHAP Analysis (Phase 2)

Same pipeline as analysis_1.py but restricted to three features:
clustering_ratio, row_group_size, and selectivity.

Dependence plots show each feature colored by each of the other two features
(6 plots total).

Usage:
    python analysis_2.py s3://cs439-project-bucket/result/20260225_154602/results.csv
"""

from __future__ import annotations

import argparse
import os
import io
import sys
import warnings
from urllib.parse import urlparse

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# S3 Helpers
# ---------------------------------------------------------------------------

def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse an S3 URL into (bucket, key)."""
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URL, got: {url}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def fetch_csv_from_s3(s3_url: str) -> pd.DataFrame:
    """Download a CSV file from S3 and return it as a DataFrame."""
    bucket, key = parse_s3_url(s3_url)
    print(f"Fetching s3://{bucket}/{key} ...")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    df = pd.read_csv(io.BytesIO(body))
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "clustering_ratio",
    "row_group_size",
    "selectivity",
]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features and target for modeling.

    Target: R = T_skip / T_scan (performance ratio; R > 1 means skipping is slower)
    Features: clustering_ratio, row_group_size, selectivity
    """
    # Drop rows with errors
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()

    # Compute auxiliary columns
    df["total_groups"] = df["skip_groups_read"] + df["skip_groups_skipped"]
    df["skip_rate"] = df["skip_groups_skipped"] / df["total_groups"]
    df["skip_rate"] = df["skip_rate"].fillna(0.0)

    # Compute performance ratio R = T_skip / T_scan
    df["R"] = df["skip_avg_execution_time"] / df["noskip_avg_execution_time"]

    # Build feature matrix
    X = df[FEATURE_COLS].copy()
    y = df["R"]

    print(f"\n--- Target: R = T_skip / T_scan ---")
    print(f"  Mean:   {y.mean():.4f}")
    print(f"  Median: {y.median():.4f}")
    print(f"  Min:    {y.min():.4f}")
    print(f"  Max:    {y.max():.4f}")
    print(f"  Std:    {y.std():.4f}")
    print(f"  (R > 1.0 means skipping is slower than scanning)")

    return X, y, df


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def train_model(
    X: pd.DataFrame, y: pd.Series
) -> GradientBoostingRegressor:
    """Train a GradientBoostingRegressor and report performance."""
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X, y)

    # In-sample metrics
    y_pred = model.predict(X)
    print(f"\n--- Model Performance (in-sample) ---")
    print(f"  R²:   {r2_score(y, y_pred):.4f}")
    print(f"  MAE:  {mean_absolute_error(y, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"\n--- 5-Fold Cross-Validation R² ---")
    print(f"  Mean:  {cv_scores.mean():.4f}")
    print(f"  Std:   {cv_scores.std():.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in cv_scores]}")

    # Built-in feature importances
    print(f"\n--- GBR Feature Importances ---")
    importances = model.feature_importances_
    for name, imp in sorted(
        zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name:25s}: {imp:.4f}")

    return model


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
    output_prefix: str = "shap",
) -> None:
    """Compute SHAP values and generate plots."""
    out_dir = "shap/shap_phase_2"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- Computing SHAP Values ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values (global importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    print(f"\n--- Mean |SHAP| Values (Global Feature Importance) ---")
    for name, val in sorted(
        zip(FEATURE_COLS, mean_abs_shap), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name:25s}: {val:.4f}")

    # ---- Plot 1: SHAP Summary (Beeswarm) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.title("SHAP Summary Plot — Impact on R (T_skip / T_scan)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{output_prefix}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir}/{output_prefix}_summary.png")

    # ---- Plot 2: SHAP Bar Plot (Mean |SHAP|) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        shap_values, X, feature_names=FEATURE_COLS, plot_type="bar", show=False
    )
    plt.title("Mean |SHAP| — Feature Importance for R (T_skip / T_scan)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{output_prefix}_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir}/{output_prefix}_bar.png")

    # ---- Plot 3: Individual SHAP Dependence Plots ----
    # For each feature, create 2 plots colored by each of the other 2 features
    # (3 features x 2 colorings = 6 plots total)
    for i, feat in enumerate(FEATURE_COLS):
        other_features = [f for f in FEATURE_COLS if f != feat]
        for color_feat in other_features:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                feat,
                shap_values,
                X,
                feature_names=FEATURE_COLS,
                interaction_index=color_feat,
                show=False,
                ax=ax,
            )
            ax.set_title(
                f"SHAP Dependence — {feat} (colored by {color_feat})", fontsize=13
            )
            plt.tight_layout()
            fname = f"{out_dir}/{output_prefix}_dep_{feat}_by_{color_feat}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def print_summary(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame) -> None:
    """Print a high-level summary of the experiment results."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total experiment runs: {len(df)}")
    print(f"  Runs where skipping helped (R < 1.0): {(df['R'] < 1.0).sum()}")
    print(f"  Runs where skipping hurt  (R > 1.0): {(df['R'] > 1.0).sum()}")
    print(f"  Runs with any groups skipped: {(df['skip_rate'] > 0).sum()}")
    print(f"  Runs with zero groups skipped: {(df['skip_rate'] == 0).sum()}")
    print(f"\n  Performance ratio R = T_skip / T_scan")
    print(f"    Mean R:   {df['R'].mean():.4f}")
    print(f"    Median R: {df['R'].median():.4f}")
    print(f"    Best R:   {df['R'].min():.4f}")
    print(f"    Worst R:  {df['R'].max():.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GBR + SHAP analysis (phase 2) on data-skipping benchmark results.",
    )
    parser.add_argument(
        "s3_url",
        type=str,
        help="S3 URL to the results CSV (e.g., s3://bucket/path/results.csv)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="shap",
        help="Prefix for output plot filenames (default: shap)",
    )
    args = parser.parse_args()

    # Suppress noisy warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Fetch data
    df = fetch_csv_from_s3(args.s3_url)

    # Prepare features & target
    X, y, df = prepare_data(df)

    # Print summary
    print_summary(X, y, df)

    # Train model
    model = train_model(X, y)

    # SHAP analysis
    run_shap_analysis(model, X, output_prefix=args.output_prefix)

    print(f"\nDone. All plots saved in shap/shap_phase_2 with prefix '{args.output_prefix}_'.")


if __name__ == "__main__":
    main()
