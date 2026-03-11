import sys
import pyarrow.parquet as pq
from pyarrow import fs

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <s3_url>")
        sys.exit(1)

    s3_url = sys.argv[1]

    # Parse s3://bucket/key
    path = s3_url.replace("s3://", "", 1)
    s3 = fs.S3FileSystem()

    parquet_file = pq.ParquetFile(s3.open_input_file(path))
    col_name = parquet_file.schema.names[0]
    col_index = 0

    num_row_groups = min(20, parquet_file.metadata.num_row_groups)
    for i in range(num_row_groups):
        row_group_metadata = parquet_file.metadata.row_group(i)
        col_metadata = row_group_metadata.column(col_index)
        print(f"--- Row Group {i} ---")
        print("Min: ", col_metadata.statistics.min)
        print("Max: ", col_metadata.statistics.max)
        print()

if __name__ == "__main__":
    main()
