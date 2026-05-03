import argparse
import pandas as pd
import sys

def append_csv_data(source_csv_path, target_csv_path="dataset.csv", max_rows=None):
    """
    Append data from a source CSV to the target CSV (dataset.csv).
    
    Args:
        source_csv_path (str): Path to the source CSV file to append
        target_csv_path (str): Path to the target CSV file (default: dataset.csv)
        max_rows (int | None): Maximum number of rows to append from source
    """
    try:
        # Read the source CSV
        source_data = pd.read_csv(source_csv_path)

        if max_rows is not None:
            source_data = source_data.head(max_rows)
        
        # Try to read existing target CSV
        try:
            target_data = pd.read_csv(target_csv_path)
            # Append source data to target data
            combined_data = pd.concat([target_data, source_data], ignore_index=True)
        except FileNotFoundError:
            # If target doesn't exist, use source data as is
            combined_data = source_data
        
        # Write combined data to target CSV
        combined_data.to_csv(target_csv_path, index=False)
        print(f"Successfully appended data from {source_csv_path} to {target_csv_path}")
        
    except FileNotFoundError:
        print(f"Error: Source CSV file '{source_csv_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error appending CSV data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Append CSV data to a target CSV."
    )

    parser.add_argument(
        "source_csv_path",
        type=str,
        help="Path to the source CSV file to append."
    )

    parser.add_argument(
        "target_csv_path",
        type=str,
        nargs="?",
        default="dataset.csv",
        help="Path to the target CSV file (default: dataset.csv)."
    )

    parser.add_argument(
        "--nlines",
        type=int,
        default=None,
        help="Maximum number of rows to append from source."
    )

    args = parser.parse_args()

    append_csv_data(
        args.source_csv_path,
        args.target_csv_path,
        args.nlines
    )
