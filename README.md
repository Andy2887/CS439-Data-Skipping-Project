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
```