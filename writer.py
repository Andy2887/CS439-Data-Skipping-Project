"""
writer.py — Synthetic Parquet Data Generator for Data Skipping Benchmarks

Generates Parquet files with configurable clustering ratio, cardinality,
selectivity, row group size, and table width. Designed to benchmark the
break-even point between metadata-based data skipping and sequential scanning
on AWS S3.

Usage (CLI):
    python writer.py \
        --total-rows 1000000 \
        --num-files 4 \
        --clustering-ratio 0.3 \
        --cardinality 10000 \
        --selectivity 0.05 \
        --row-group-size 100000 \
        --table-width 20 \
        --predicate-type equality \
        --predicate-value 42 \
        --s3-bucket my-bucket \
        --s3-prefix experiments/run1

Usage (Python):
    from writer import generate_parquet_files, GeneratorConfig

    config = GeneratorConfig(
        total_rows=1_000_000,
        num_files=4,
        clustering_ratio=0.3,
        cardinality=10_000,
        selectivity=0.05,
        row_group_size=100_000,
        table_width=20,
        predicate_type="equality",
        predicate_value=42,
        s3_bucket="my-bucket",
        s3_prefix="experiments/run1",
    )
    uris = generate_parquet_files(config)
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
import random
import string
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREDICATE_TYPES = ("equality", "greater_than", "less_than", "range")


@dataclass
class GeneratorConfig:
    """All parameters that control synthetic Parquet generation."""

    # Core table dimensions
    total_rows: int = 1_000_000
    num_files: int = 1
    table_width: int = 20  # total columns including the target column

    # Data-skipping-relevant knobs
    clustering_ratio: float = 0.0  # σ: 0.0 = perfectly sorted, 1.0 = fully random
    cardinality: int = 10_000  # unique values in target column
    selectivity: float = 0.05  # fraction of rows matching the predicate
    row_group_size: int = 100_000

    # Predicate specification
    predicate_type: str = "equality"  # equality | greater_than | less_than | range
    predicate_value: int = 42  # single value for equality/greater_than/less_than
    predicate_upper: Optional[int] = None  # upper bound for range predicates

    # Output (S3 only)
    s3_bucket: str = ""
    s3_prefix: Optional[str] = None

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        assert 0.0 <= self.clustering_ratio <= 1.0, "clustering_ratio must be in [0, 1]"
        assert 0.0 < self.selectivity <= 1.0, "selectivity must be in (0, 1]"
        assert self.cardinality > 0, "cardinality must be > 0"
        assert self.row_group_size > 0, "row_group_size must be > 0"
        assert self.table_width >= 1, "table_width must be >= 1 (need at least the target column)"
        assert self.predicate_type in PREDICATE_TYPES, (
            f"predicate_type must be one of {PREDICATE_TYPES}"
        )
        if self.predicate_type == "range" and self.predicate_upper is None:
            raise ValueError("predicate_upper is required for range predicates")
        if not self.s3_bucket:
            raise ValueError("s3_bucket is required")


# ---------------------------------------------------------------------------
# Target Column Generation
# ---------------------------------------------------------------------------

def _build_int_value_pool(
    cardinality: int,
    seed: int,
    ensure_values: Optional[List[int]] = None,
) -> np.ndarray:
    """Create a pool of `cardinality` unique integer values.

    If `ensure_values` is provided, those values are guaranteed to be in the
    pool (useful so the predicate value is always present).
    """
    rng = np.random.default_rng(seed)
    ensure_values = [int(v) for v in (ensure_values or [])]

    # Range must cover the ensured values
    upper = max(cardinality * 10, max(ensure_values, default=0) + cardinality)

    # Start with the guaranteed values
    pool_set = set(ensure_values)

    # Fill remaining slots with random unique values
    candidates = rng.choice(upper, size=min(upper, cardinality * 20), replace=False)
    for v in candidates:
        if len(pool_set) >= cardinality:
            break
        pool_set.add(int(v))

    pool = np.sort(np.array(list(pool_set), dtype=np.int64))
    return pool



def _matching_mask(
    values: np.ndarray,
    predicate_type: str,
    predicate_value: int,
    predicate_upper: Optional[int] = None,
) -> np.ndarray:
    """Return a boolean mask: True where `values` satisfy the predicate."""
    if predicate_type == "equality":
        return values == predicate_value
    elif predicate_type == "greater_than":
        return values > predicate_value
    elif predicate_type == "less_than":
        return values < predicate_value
    elif predicate_type == "range":
        return (values >= predicate_value) & (values <= predicate_upper)
    else:
        raise ValueError(f"Unknown predicate_type: {predicate_type}")


def _compute_ensured_values(
    predicate_type: str,
    predicate_value: int,
    predicate_upper: Optional[int],
    seed: int,
) -> List[int]:
    """Return up to 100 values guaranteed to satisfy the predicate."""
    if predicate_type == "equality":
        return [predicate_value]
    elif predicate_type == "greater_than":
        return [predicate_value + 1 + i for i in range(100)]
    elif predicate_type == "less_than":
        start = max(0, predicate_value - 100)
        return list(range(start, predicate_value))
    elif predicate_type == "range":
        lo, hi = predicate_value, predicate_upper
        if hi - lo + 1 <= 100:
            return list(range(lo, hi + 1))
        else:
            rng = np.random.default_rng(seed)
            vals = rng.choice(np.arange(lo, hi + 1), size=100, replace=False)
            return sorted(int(v) for v in vals)
    return []


def _generate_target_int_column(
    total_rows: int,
    value_pool: np.ndarray,
    selectivity: float,
    predicate_type: str,
    predicate_value: int,
    predicate_upper: Optional[int],
    clustering_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate integer target column with exact selectivity, then apply
    displacement-based shuffle controlled by clustering_ratio (σ).

    Steps:
      1. Determine which pool values match the predicate.
      2. Sample `n_match` rows from matching values and `total_rows - n_match`
         rows from non-matching values.
      3. Sort the combined column.
      4. Apply displacement shuffle with probability σ.
    """
    n_match = int(round(selectivity * total_rows))
    n_non_match = total_rows - n_match

    # Partition pool into matching / non-matching
    match_mask = _matching_mask(value_pool, predicate_type, predicate_value, predicate_upper)
    matching_vals = value_pool[match_mask]
    non_matching_vals = value_pool[~match_mask]

    if len(matching_vals) == 0:
        raise ValueError(
            "No values in the pool satisfy the predicate. "
            "Adjust predicate_value / cardinality."
        )
    if len(non_matching_vals) == 0 and n_non_match > 0:
        raise ValueError(
            "All pool values match the predicate — cannot achieve the "
            "requested selectivity. Adjust predicate or cardinality."
        )

    # Sample with replacement from each partition
    col_match = rng.choice(matching_vals, size=n_match, replace=True)
    if n_non_match > 0:
        col_non_match = rng.choice(non_matching_vals, size=n_non_match, replace=True)
        column = np.concatenate([col_match, col_non_match])
    else:
        column = col_match

    # Sort, then apply displacement shuffle
    column.sort()
    column = _displacement_shuffle(column, clustering_ratio, rng)
    return column



# ---------------------------------------------------------------------------
# Displacement-based Shuffle
# ---------------------------------------------------------------------------

def _displacement_shuffle(
    arr: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Displacement-based shuffle: for each element, with probability σ,
    relocate it to a uniformly random position.

    σ = 0.0  →  array stays perfectly sorted
    σ = 1.0  →  effectively a random permutation
    """
    if sigma == 0.0:
        return arr
    n = len(arr)
    # Choose which elements to displace
    mask = rng.random(n) < sigma
    displaced_indices = np.where(mask)[0]
    del mask
    # For each displaced element, pick a random target position and swap
    for idx in displaced_indices:
        target = rng.integers(0, n)
        arr[idx], arr[target] = arr[target], arr[idx]
    return arr


# ---------------------------------------------------------------------------
# Non-Target (Filler) Column Generation
# ---------------------------------------------------------------------------

def _generate_filler_columns_chunk(
    n_filler: int,
    chunk_rows: int,
    rng: np.random.Generator,
    py_rng: random.Random,
) -> List[Tuple[str, pa.Array]]:
    """
    Generate non-target filler columns for a single chunk of rows.
    Distributes across four types: int, string, timestamp, and float.
    """
    if n_filler <= 0:
        return []

    # Distribute columns across the four types as evenly as possible
    type_cycle = ["int", "string", "timestamp", "float"]
    assignments = [type_cycle[i % len(type_cycle)] for i in range(n_filler)]

    columns: List[Tuple[str, pa.Array]] = []

    for i, col_type in enumerate(assignments):
        col_name = f"filler_{col_type}_{i}"

        if col_type == "int":
            data = rng.integers(0, 1_000_000, size=chunk_rows)
            columns.append((col_name, pa.array(data, type=pa.int64())))
            del data

        elif col_type == "string":
            lengths = rng.integers(5, 50, size=chunk_rows)
            strs = [
                "".join(py_rng.choices(string.ascii_letters, k=int(l)))
                for l in lengths
            ]
            columns.append((col_name, pa.array(strs, type=pa.string())))
            del lengths, strs

        elif col_type == "timestamp":
            base = datetime.datetime(2020, 1, 1)
            offsets = rng.integers(0, 365 * 5 * 86400, size=chunk_rows)
            ts = [base + datetime.timedelta(seconds=int(o)) for o in offsets]
            columns.append((col_name, pa.array(ts, type=pa.timestamp("us"))))
            del offsets, ts

        elif col_type == "float":
            data = rng.uniform(-1e6, 1e6, size=chunk_rows)
            columns.append((col_name, pa.array(data, type=pa.float64())))
            del data

    return columns


# ---------------------------------------------------------------------------
# Core Generator
# ---------------------------------------------------------------------------

def _build_schema(config: GeneratorConfig) -> pa.Schema:
    """Build the Parquet schema from config (no data allocated)."""
    fields = [pa.field("target_int", pa.int64())]
    n_filler = config.table_width - 1
    type_cycle = ["int", "string", "timestamp", "float"]
    type_map = {
        "int": pa.int64(),
        "string": pa.string(),
        "timestamp": pa.timestamp("us"),
        "float": pa.float64(),
    }
    for i in range(n_filler):
        col_type = type_cycle[i % len(type_cycle)]
        fields.append(pa.field(f"filler_{col_type}_{i}", type_map[col_type]))
    return pa.schema(fields)


def _write_parquet_chunked(config: GeneratorConfig, file_idx: int, tmp_path: str) -> None:
    """Generate data chunk-by-chunk and write row groups to a temp file."""
    rng = np.random.default_rng(config.seed + file_idx)

    # Value pool (shared across files for consistency)
    ensure = _compute_ensured_values(
        config.predicate_type, config.predicate_value, config.predicate_upper, config.seed,
    )
    int_pool = _build_int_value_pool(config.cardinality, config.seed, ensure_values=ensure)

    # Generate the full target column (~8 bytes/row — must be global for sorting + shuffle)
    int_target = _generate_target_int_column(
        total_rows=config.total_rows,
        value_pool=int_pool,
        selectivity=config.selectivity,
        predicate_type=config.predicate_type,
        predicate_value=config.predicate_value,
        predicate_upper=config.predicate_upper,
        clustering_ratio=config.clustering_ratio,
        rng=rng,
    )
    del int_pool

    # Separate py_rng for filler string columns
    py_rng = random.Random(int(rng.integers(0, 2**31)))

    schema = _build_schema(config)
    n_filler = config.table_width - 1
    row_group_size = config.row_group_size
    total = config.total_rows

    writer = pq.ParquetWriter(tmp_path, schema, write_statistics=True, version="2.6")

    for start in range(0, total, row_group_size):
        end = min(start + row_group_size, total)
        chunk_rows = end - start

        # Slice target column for this chunk
        target_chunk = int_target[start:end]

        # Generate filler columns only for this chunk
        filler_cols = _generate_filler_columns_chunk(n_filler, chunk_rows, rng, py_rng)

        # Build chunk table
        columns: List[Tuple[str, pa.Array]] = [
            ("target_int", pa.array(target_chunk, type=pa.int64())),
        ]
        columns.extend(filler_cols)
        names = [c[0] for c in columns]
        arrays = [c[1] for c in columns]
        chunk_table = pa.table(dict(zip(names, arrays)))

        writer.write_table(chunk_table)

        # Free chunk memory
        del target_chunk, filler_cols, columns, names, arrays, chunk_table
        gc.collect()

    writer.close()
    del int_target
    gc.collect()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_parquet(
    file_path: str,
    s3_key: str,
    config: GeneratorConfig,
) -> None:
    """
    Read metadata of a written Parquet file and print validation info:
      - Number of row groups and their sizes
      - Min/max stats for target columns per row group
      - Actual vs requested selectivity
    """
    pf = pq.ParquetFile(file_path)
    metadata = pf.metadata
    n_groups = metadata.num_row_groups

    print(f"\n{'='*60}")
    print(f"  Validating: {s3_key}")
    print(f"{'='*60}")
    print(f"  Row groups : {n_groups}")
    print(f"  Total rows : {metadata.num_rows}")
    print(f"  Columns    : {metadata.num_columns}")

    for rg_idx in range(n_groups):
        rg = metadata.row_group(rg_idx)
        print(f"\n  Row Group {rg_idx}: {rg.num_rows} rows")

        # target_int is column 0
        col_meta = rg.column(0)
        if col_meta.statistics is not None and col_meta.statistics.has_min_max:
            stats = col_meta.statistics
            print(f"    target_int  min={stats.min}  max={stats.max}")
        else:
            print(f"    target_int  (no min/max stats)")

    # Verify actual selectivity
    # table = pq.read_table(pa.BufferReader(buf), columns=["target_int"])
    # int_col = table.column("target_int").to_numpy()
    # match_mask = _matching_mask(
    #     int_col, config.predicate_type, config.predicate_value, config.predicate_upper
    # )
    # actual_sel = match_mask.sum() / len(int_col)
    # print(f"\n  Selectivity (requested) : {config.selectivity:.4f}")
    # print(f"  Selectivity (actual)    : {actual_sel:.4f}")
    # diff = abs(actual_sel - config.selectivity)
    # status = "OK" if diff < 0.01 else "MISMATCH"
    # print(f"  Status                  : {status} (diff={diff:.6f})")


# ---------------------------------------------------------------------------
# S3 Upload
# ---------------------------------------------------------------------------

def upload_to_s3(file_path: str, filename: str, bucket: str, prefix: str) -> str:
    """Upload a Parquet file to S3 and return the S3 URI."""
    import boto3

    s3 = boto3.client("s3")
    key = f"{prefix.rstrip('/')}/{filename}" if prefix else filename
    s3.upload_file(file_path, bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"  Uploaded → {uri}")
    return uri


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_parquet_files(config: GeneratorConfig) -> List[str]:
    """
    Generate synthetic Parquet files according to `config`.

    Returns a list of S3 URIs where files were uploaded.
    """
    uris: List[str] = []

    for i in range(config.num_files):
        filename = f"data_{i:04d}.parquet"
        s3_key = f"{config.s3_prefix.rstrip('/')}/{filename}" if config.s3_prefix else filename

        print(f"\n[File {i+1}/{config.num_files}] Generating {config.total_rows} rows …")

        fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        try:
            print(f"  Writing to temp file (row_group_size={config.row_group_size}) …")
            _write_parquet_chunked(config, file_idx=i, tmp_path=tmp_path)

            # validate_parquet(tmp_path, s3_key, config)

            uri = upload_to_s3(tmp_path, filename, config.s3_bucket, config.s3_prefix or "")
            uris.append(uri)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            gc.collect()

    print(f"\nDone. Generated {len(uris)} file(s) → s3://{config.s3_bucket}/{config.s3_prefix or ''}")
    return uris


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> GeneratorConfig:
    """Parse command-line arguments into a GeneratorConfig."""
    p = argparse.ArgumentParser(
        description="Generate synthetic Parquet files for data-skipping benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--total-rows", type=int, default=1_000_000,
                    help="Total number of rows per file")
    p.add_argument("--num-files", type=int, default=1,
                    help="Number of Parquet files to generate")
    p.add_argument("--clustering-ratio", type=float, default=0.0,
                    help="Shuffle parameter σ ∈ [0,1]; 0=sorted, 1=random")
    p.add_argument("--cardinality", type=int, default=10_000,
                    help="Number of unique values in target column")
    p.add_argument("--selectivity", type=float, default=0.05,
                    help="Fraction of rows matching the predicate")
    p.add_argument("--row-group-size", type=int, default=100_000,
                    help="Maximum number of rows per row group")
    p.add_argument("--table-width", type=int, default=20,
                    help="Total number of columns (including the target column)")

    p.add_argument("--predicate-type", choices=PREDICATE_TYPES, default="equality",
                    help="Predicate type for selectivity calibration")
    p.add_argument("--predicate-value", type=int, default=42,
                    help="Predicate value (or lower bound for range)")
    p.add_argument("--predicate-upper", type=int, default=None,
                    help="Upper bound for range predicates")

    p.add_argument("--s3-bucket", type=str, required=True,
                    help="S3 bucket name")
    p.add_argument("--s3-prefix", type=str, default=None,
                    help="S3 key prefix (optional)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    args = p.parse_args()

    return GeneratorConfig(
        total_rows=args.total_rows,
        num_files=args.num_files,
        clustering_ratio=args.clustering_ratio,
        cardinality=args.cardinality,
        selectivity=args.selectivity,
        row_group_size=args.row_group_size,
        table_width=args.table_width,
        predicate_type=args.predicate_type,
        predicate_value=args.predicate_value,
        predicate_upper=args.predicate_upper,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    generate_parquet_files(cfg)