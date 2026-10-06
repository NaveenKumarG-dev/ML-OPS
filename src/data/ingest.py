from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/bank-full.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw Bank Marketing dataset."""
    df = pd.read_csv(path, sep=";")

    return df


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())