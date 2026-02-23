"""
run_experiments.py — Batch orchestrator for Parquet data-skipping benchmarks.

Reads a TSV plan file, runs writer + reader for each row, and collects
results into a CSV file (wide format: one row per experiment).

Usage:
    python run_experiments.py \
        --plan plan.tsv \
        --s3-prefix-base experiments \
        --output results.csv \
        [--s3-bucket BUCKET] [--total-rows N] [--num-files N] \
        [--table-width N] [--seed N]
"""

from __future__ import annotations

import argparse
import csv
import math
import traceback
from datetime import datetime
from typing import Any, Dict

from writer import GeneratorConfig, generate_parquet_files
from reader import run_query

# ---------------------------------------------------------------------------
# Type coercion map for GeneratorConfig fields
# ---------------------------------------------------------------------------

FIELD_TYPES: Dict[str, type] = {
    "total_rows": int,
    "num_files": int,
    "table_width": int,
    "clustering_ratio": float,
    "cardinality": int,
    "selectivity": float,
    "row_group_size": int,
    "predicate_type": str,
    "predicate_value_1": int,
    "predicate_value_2": float,  # float so "nan" parses via float()
    "seed": int,
}

# Writer uses "equality"; reader uses "equal". Map writer → reader terms.
PREDICATE_TYPE_MAP = {
    "equality": "equal",
    "greater_than": "greater_than",
    "less_than": "less_than",
    "range": "range",
}

# CSV columns for writer params (order matters for output)
WRITER_PARAM_COLS = [
    "total_rows",
    "num_files",
    "table_width",
    "clustering_ratio",
    "cardinality",
    "selectivity",
    "row_group_size",
    "predicate_type",
    "predicate_value_1",
    "predicate_value_2",
    "seed",
    "s3_bucket",
    "s3_prefix",
]

METRIC_KEYS = ["total_matches", "groups_read", "groups_skipped", "execution_time"]

CSV_COLUMNS = (
    ["run_index"]
    + WRITER_PARAM_COLS
    + [f"skip_{k}" for k in METRIC_KEYS]
    + [f"noskip_{k}" for k in METRIC_KEYS]
    + ["error"]
)


# ---------------------------------------------------------------------------
# Plan file parsing
# ---------------------------------------------------------------------------

def parse_plan(plan_path: str) -> list[dict[str, Any]]:
    """Read a CSV plan file and return a list of per-row override dicts.

    Plan columns map to GeneratorConfig fields:
      predicate_value_1 → predicate_value
      predicate_value_2 → predicate_upper  (nan means unused)
    """
    with open(plan_path, newline="") as f:
        reader = csv.DictReader(f)

        # Validate column names
        for col in reader.fieldnames or []:
            if col not in FIELD_TYPES:
                print(f"Warning: unknown plan column '{col}' (will be ignored)")

        rows = []
        for raw_row in reader:
            coerced: Dict[str, Any] = {}
            for col, val in raw_row.items():
                if col not in FIELD_TYPES:
                    continue
                val = val.strip() if val else ""
                if val == "" or val.lower() == "nan":
                    continue  # use default
                coerced[col] = FIELD_TYPES[col](val)
            rows.append(coerced)

    print(f"Loaded {len(rows)} experiment(s) from {plan_path}")
    return rows


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

# Plan column name → GeneratorConfig field name
PLAN_TO_CONFIG = {
    "predicate_value_1": "predicate_value",
    "predicate_value_2": "predicate_upper",
}


def build_config(
    plan_overrides: Dict[str, Any],
    cli_defaults: Dict[str, Any],
    s3_bucket: str,
    s3_prefix: str,
) -> GeneratorConfig:
    """Build a GeneratorConfig with priority: plan row > CLI arg > dataclass default."""
    kwargs: Dict[str, Any] = {}

    for field in FIELD_TYPES:
        config_key = PLAN_TO_CONFIG.get(field, field)
        if field in plan_overrides:
            kwargs[config_key] = plan_overrides[field]
        elif field in cli_defaults:
            kwargs[config_key] = cli_defaults[field]
        # else: let GeneratorConfig use its own default

    # predicate_upper must be int or None for GeneratorConfig
    if "predicate_upper" in kwargs:
        v = kwargs["predicate_upper"]
        if isinstance(v, float) and math.isnan(v):
            del kwargs["predicate_upper"]
        else:
            kwargs["predicate_upper"] = int(v)

    kwargs["s3_bucket"] = s3_bucket
    kwargs["s3_prefix"] = s3_prefix

    return GeneratorConfig(**kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch experiment runner for Parquet data-skipping benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--plan", required=True, help="Path to TSV plan file")
    p.add_argument("--s3-bucket", default="cs439-project-bucket",
                   help="S3 bucket for all experiments")
    p.add_argument("--s3-prefix-base", default="experiments",
                   help="Base S3 prefix; each run gets {base}/{timestamp}/run_{NNN}")
    p.add_argument("--output", default="results.csv", help="Output CSV path")

    # Batch-level defaults (override GeneratorConfig defaults, overridden by plan)
    p.add_argument("--total-rows", type=int, default=None)
    p.add_argument("--table-width", type=int, default=None)
    p.add_argument("--num-files", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def cli_defaults_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract non-None CLI defaults into a dict keyed by GeneratorConfig field names."""
    mapping = {
        "total_rows": args.total_rows,
        "num_files": args.num_files,
        "table_width": args.table_width,
        "seed": args.seed,
    }
    return {k: v for k, v in mapping.items() if v is not None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    plan_rows = parse_plan(args.plan)
    cli_defs = cli_defaults_from_args(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Open CSV for incremental writing
    csv_file = open(args.output, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    csv_file.flush()

    try:
        for idx, plan_row in enumerate(plan_rows):
            s3_prefix = f"{args.s3_prefix_base}/{timestamp}/run_{idx:03d}"

            print(f"\n{'='*60}")
            print(f"  Experiment {idx} / {len(plan_rows) - 1}")
            print(f"  S3 prefix: s3://{args.s3_bucket}/{s3_prefix}")
            print(f"  Plan overrides: {plan_row}")
            print(f"{'='*60}")

            csv_row: Dict[str, Any] = {"run_index": idx, "error": ""}

            try:
                config = build_config(plan_row, cli_defs, args.s3_bucket, s3_prefix)

                # Record writer params
                for col in WRITER_PARAM_COLS:
                    config_key = PLAN_TO_CONFIG.get(col, col)
                    val = getattr(config, config_key, "")
                    # Show nan for unused predicate_value_2
                    if col == "predicate_value_2" and val is None:
                        val = "nan"
                    csv_row[col] = val

                # --- Write ---
                print("\n>>> Running writer …")
                generate_parquet_files(config)

                # --- Reader query args ---
                s3_data_path = f"{args.s3_bucket}/{config.s3_prefix}"
                reader_query_type = PREDICATE_TYPE_MAP.get(
                    config.predicate_type, config.predicate_type
                )
                value1 = config.predicate_value
                value2 = config.predicate_upper  # None for non-range

                # --- Read with data skipping ---
                print("\n>>> Running reader (data_skipping=True) …")
                skip_metrics = run_query(
                    s3_data_path, reader_query_type, value1, value2, data_skipping=True
                )
                for k in METRIC_KEYS:
                    csv_row[f"skip_{k}"] = skip_metrics[k]

                # --- Read without data skipping ---
                print("\n>>> Running reader (data_skipping=False) …")
                noskip_metrics = run_query(
                    s3_data_path, reader_query_type, value1, value2, data_skipping=False
                )
                for k in METRIC_KEYS:
                    csv_row[f"noskip_{k}"] = noskip_metrics[k]

            except Exception:
                tb = traceback.format_exc()
                print(f"\n!!! Experiment {idx} failed:\n{tb}")
                csv_row["error"] = tb.splitlines()[-1]

            writer.writerow(csv_row)
            csv_file.flush()

    finally:
        csv_file.close()

    print(f"\nAll experiments complete. Results written to {args.output}")


if __name__ == "__main__":
    main()
