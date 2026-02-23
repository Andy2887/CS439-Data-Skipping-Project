import csv
import os
from scipy.stats.qmc import LatinHypercube
import numpy as np

sampler = LatinHypercube(d=5, seed=42)
sample = sampler.random(n=100)  # 100 points in [0,1]^5

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

os.makedirs("plan", exist_ok=True)
fieldnames = ["clustering_ratio", "selectivity", "cardinality", "row_group_size",
              "predicate_type", "predicate_value_1", "predicate_value_2"]
with open("plan/plan.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(configs)

print(f"Wrote {len(configs)} configs to plan/plan.csv")

