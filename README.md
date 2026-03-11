```bash
# activate environment
source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# run program
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
    --s3-bucket cs439-project-bucket \
    --s3-prefix experiments/run1

# Path to phase 1 results
s3://cs439-project-bucket/result/20260225_154602/results.csv
# Path to phase 2 results
s3://cs439-project-bucket/result/20260311_144509/results.csv
```

## File Size

A file containing 160,000,000 data (1000000 rows * 80 columns) is 1.1GB.

We will try to keep our per file data under 1GB.