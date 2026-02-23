import time
import pyarrow.parquet as pq
import pyarrow.fs as fs

s3 = fs.S3FileSystem(region="us-east-2")

pf = pq.ParquetFile(s3.open_input_file("cs439-project-bucket/experiments/run2/data_0000.parquet"))

TARGET = 42
metadata = pf.metadata
num_row_groups = metadata.num_row_groups
total_count = 0
groups_read = 0
groups_skipped = 0

start = time.time()

for i in range(num_row_groups):
    stats = metadata.row_group(i).column(0).statistics
    if stats.min <= TARGET <= stats.max:
        table = pf.read_row_group(i, columns=[metadata.row_group(i).column(0).path_in_schema])
        col = table.column(0)
        total_count += col.to_pylist().count(TARGET)
        groups_read += 1
    else:
        groups_skipped += 1

elapsed = time.time() - start

print(f"Total occurrences of {TARGET}: {total_count}")
print(f"Row groups read: {groups_read}")
print(f"Row groups skipped: {groups_skipped}")
print(f"Execution time: {elapsed:.4f} seconds")
