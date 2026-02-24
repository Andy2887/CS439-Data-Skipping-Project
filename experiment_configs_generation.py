import csv
import io
import sys

import boto3
import numpy as np
from scipy.stats.qmc import LatinHypercube

n_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 100

sampler = LatinHypercube(d=5, seed=42)
sample = sampler.random(n=n_configs)

# Map each dimension back to your discrete levels
clustering_levels = [0.0, 1.0]
selectivity_levels = [0.01, 0.05, 0.20, 0.50, 0.90]
cardinality_levels = [1_000, 10_000, 100_000, 1_000_000]
row_group_levels = [10_000, 50_000, 100_000, 500_000]
predicate_levels = ["equality", "greater_than", "less_than", "range"]

def map_to_level(unit_val, levels):
    idx = int(unit_val * len(levels))
    idx = min(idx, len(levels) - 1)
    return levels[idx]

rng = np.random.default_rng(42)
configs = []
for row in sample:
    pred_type = map_to_level(row[4], predicate_levels)
    v1 = int(rng.integers(0, 10_000_001))
    v2 = int(rng.integers(0, 10_000_001)) if pred_type == "range" else "nan"
    if v2 != "nan" and v2 < v1:
        v1, v2 = v2, v1
    configs.append({
        "clustering_ratio": map_to_level(row[0], clustering_levels),
        "selectivity": map_to_level(row[1], selectivity_levels),
        "cardinality": map_to_level(row[2], cardinality_levels),
        "row_group_size": map_to_level(row[3], row_group_levels),
        "predicate_type": pred_type,
        "predicate_value_1": v1,
        "predicate_value_2": v2,
    })

S3_BUCKET = "cs439-project-bucket"
S3_KEY = "plan/plan.csv"

fieldnames = ["clustering_ratio", "selectivity", "cardinality", "row_group_size",
              "predicate_type", "predicate_value_1", "predicate_value_2"]

csv_buf = io.StringIO()
writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(configs)

s3 = boto3.client("s3")
s3.upload_fileobj(
    io.BytesIO(csv_buf.getvalue().encode("utf-8")),
    S3_BUCKET,
    S3_KEY,
)

print(f"Wrote {len(configs)} configs to s3://{S3_BUCKET}/{S3_KEY}")

