import csv
import io
import itertools

import boto3
import numpy as np

CLUSTERING_RATIO = 0
selectivity_levels = [0.05, 0.20, 0.40, 0.50, 0.60, 0.80]
row_group_levels = [2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]

CARDINALITY = 10_000
PREDICATE_TYPE = "equality"

rng = np.random.default_rng(42)
configs = []
for selectivity, row_group in itertools.product(
    selectivity_levels, row_group_levels
):
    v1 = int(rng.integers(0, 10_000_001))
    configs.append({
        "clustering_ratio": CLUSTERING_RATIO,
        "selectivity": selectivity,
        "cardinality": CARDINALITY,
        "row_group_size": row_group,
        "predicate_type": PREDICATE_TYPE,
        "predicate_value_1": v1,
        "predicate_value_2": "nan",
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
