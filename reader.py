import sys
import time
import pyarrow.parquet as pq
import pyarrow.fs as fs


def should_skip(stats, query_type, value1, value2):
    col_min = stats.min
    col_max = stats.max

    if query_type == "equal":
        return value1 < col_min or value1 > col_max
    elif query_type == "range":
        return value2 < col_min or value1 > col_max
    elif query_type == "greater_than":
        return col_max <= value1
    elif query_type == "less_than":
        return col_min >= value1
    return False


def matches(value, query_type, value1, value2):
    if query_type == "equal":
        return value == value1
    elif query_type == "range":
        return value1 <= value <= value2
    elif query_type == "greater_than":
        return value > value1
    elif query_type == "less_than":
        return value < value1
    return False


def run_query(s3_path, query_type, value1, value2, data_skipping, num_runs=3):
    s3 = fs.S3FileSystem(region="us-east-2")
    file_infos = s3.get_file_info(fs.FileSelector(s3_path))
    parquet_files = sorted(
        info.path for info in file_infos if info.path.endswith(".parquet")
    )

    execution_times = []

    for run_idx in range(num_runs):
        total_count = 0
        groups_read = 0
        groups_skipped = 0

        start = time.time()

        for path in parquet_files:
            pf = pq.ParquetFile(s3.open_input_file(path))
            metadata = pf.metadata
            num_row_groups = metadata.num_row_groups
            col_name = metadata.row_group(0).column(0).path_in_schema

            if data_skipping:
                for i in range(num_row_groups):
                    stats = metadata.row_group(i).column(0).statistics
                    if should_skip(stats, query_type, value1, value2):
                        groups_skipped += 1
                        continue

                    table = pf.read_row_group(i, columns=[col_name])
                    col = table.column(0).to_pylist()
                    total_count += sum(1 for v in col if matches(v, query_type, value1, value2))
                    groups_read += 1
            else:
                table = pf.read(columns=[col_name])
                col = table.column(0).to_pylist()
                total_count += sum(1 for v in col if matches(v, query_type, value1, value2))
                groups_read += num_row_groups

        elapsed = time.time() - start
        execution_times.append(elapsed)

        print(f"  Run {run_idx + 1}/{num_runs}: {elapsed:.4f} seconds")

    avg_time = sum(execution_times) / len(execution_times)

    print(f"Query type: {query_type}")
    if query_type == "range":
        print(f"Query range: [{value1}, {value2}]")
    else:
        print(f"Query value: {value1}")
    print(f"Data skipping: {data_skipping}")
    print(f"Total matches: {total_count}")
    print(f"Row groups read: {groups_read}")
    print(f"Row groups skipped: {groups_skipped}")
    for i, t in enumerate(execution_times):
        print(f"Execution time {i + 1}: {t:.4f} seconds")
    print(f"Average execution time: {avg_time:.4f} seconds")

    return {
        "total_matches": total_count,
        "groups_read": groups_read,
        "groups_skipped": groups_skipped,
        "execution_times": execution_times,
        "avg_execution_time": avg_time,
    }


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python reader.py <s3_path> <data_skipping> <query_type> <value1> [value2]"
        )
        print("  data_skipping: true or false")
        print("  query_type: equal, range, greater_than, less_than")
        sys.exit(1)

    s3_path = sys.argv[1]
    data_skipping = sys.argv[2].lower() == "true"
    query_type = sys.argv[3]

    if query_type == "range":
        if len(sys.argv) < 6:
            print("Range query requires both value1 and value2.")
            sys.exit(1)
        value1 = int(sys.argv[4])
        value2 = int(sys.argv[5])
    else:
        value1 = int(sys.argv[4])
        value2 = None

    run_query(s3_path, query_type, value1, value2, data_skipping)


if __name__ == "__main__":
    main()
