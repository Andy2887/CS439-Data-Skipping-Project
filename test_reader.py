import struct
import boto3
import pyarrow.parquet as pq
import io

s3 = boto3.client('s3')

BUCKET = 'cs439-project-bucket'
KEY = 'experiments/run1/data_0000.parquet'

# First, get the file size
response = s3.head_object(Bucket=BUCKET, Key=KEY)
file_size = response['ContentLength']
print(f"File size: {file_size} bytes")

# Step 1: Read the last 8 bytes to get the footer length + magic
tail_resp = s3.get_object(
    Bucket=BUCKET, Key=KEY,
    Range=f'bytes={file_size - 8}-{file_size - 1}'
)
tail = tail_resp['Body'].read()

magic = tail[4:8]
if magic != b'PAR1':
    raise ValueError(f"Not a valid Parquet file (magic={magic!r})")

footer_length = struct.unpack('<I', tail[0:4])[0]
print(f"Footer length: {footer_length} bytes")

# Step 2: Fetch the footer + 8-byte suffix
footer_start = file_size - footer_length - 8
footer_resp = s3.get_object(
    Bucket=BUCKET, Key=KEY,
    Range=f'bytes={footer_start}-{file_size - 1}'
)
footer_bytes = footer_resp['Body'].read()

# Step 3: Parse metadata using pyarrow by wrapping in a minimal in-memory file
# pyarrow can read metadata from a file-like object that contains the footer
# We build a buffer: [padding to align offsets] + footer + footer_length(4) + PAR1(4)
# But the simplest way is to use pq.read_metadata with a full parquet-like buffer

# Use pyarrow's built-in Thrift deserializer via ParquetFile
# We need to give it a seekable file that looks like a parquet file
# Pad the front so that the footer offset math works out
padding = b'\x00' * footer_start
parquet_buf = padding + footer_bytes
pf = pq.ParquetFile(io.BytesIO(parquet_buf))
metadata = pf.metadata

# Print file-level metadata
print(f"\n=== Parquet File Metadata ===")
print(f"Created by: {metadata.created_by}")
print(f"Format version: {metadata.format_version}")
print(f"Number of rows: {metadata.num_rows}")
print(f"Number of columns: {metadata.num_columns}")
print(f"Number of row groups: {metadata.num_row_groups}")
print(f"Serialized size: {metadata.serialized_size} bytes")

# Print schema
print(f"\n=== Schema ===")
schema = metadata.schema
for i in range(len(schema)):
    col = schema.column(i)
    print(f"  [{i}] {col.name}: {col.physical_type} (logical: {col.logical_type})")

# Print key-value metadata if present
if metadata.metadata:
    print(f"\n=== Key-Value Metadata ===")
    for key, value in metadata.metadata.items():
        val_str = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else str(value)
        # Truncate long values
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"  {key}: {val_str}")

# Print row group details
print(f"\n=== Row Groups ===")
for rg_idx in range(metadata.num_row_groups):
    rg = metadata.row_group(rg_idx)
    print(f"\n  Row Group {rg_idx}: {rg.num_rows} rows, {rg.total_byte_size} bytes")
    for col_idx in range(rg.num_columns):
        col = rg.column(col_idx)
        print(f"    Column {col_idx} ({col.path_in_schema}):")
        print(f"      Compression: {col.compression}")
        print(f"      Encodings: {col.encodings}")
        print(f"      Compressed size: {col.total_compressed_size} bytes")
        print(f"      Uncompressed size: {col.total_uncompressed_size} bytes")
        if col.is_stats_set:
            stats = col.statistics
            print(f"      Stats: min={stats.min}, max={stats.max}, "
                  f"null_count={stats.null_count}, num_values={stats.num_values}")