import sys
import pyarrow.parquet as pq
from pyarrow import fs

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <s3_url> <num_rows>")
        sys.exit(1)

    s3_url = sys.argv[1]
    num_rows = int(sys.argv[2])

    # Parse s3://bucket/key
    path = s3_url.replace("s3://", "", 1)
    s3 = fs.S3FileSystem()

    parquet_file = pq.ParquetFile(s3.open_input_file(path))
    col_name = parquet_file.schema.names[0]

    rows_read = 0
    for batch in parquet_file.iter_batches(columns=[col_name]):
        col = batch.column(0)
        for i in range(len(col)):
            if rows_read >= num_rows:
                return
            print(col[i].as_py())
            rows_read += 1

if __name__ == "__main__":
    main()
