from pathlib import Path

from src.data.ingest import load_data
from src.data.validate import validate_dataset


RAW_DATA_PATH = Path("data/raw/bank-full.csv")


def run_pipeline() -> None:
    print("Starting pipeline...")

    print("\n[1/2] Loading data...")
    df = load_data(RAW_DATA_PATH)

    print(f"Loaded {len(df)} rows.")

    print("\n[2/2] Validating data...")
    validate_dataset(df)

    print("Dataset validation passed.")

    print("\nPipeline completed successfully.")

    

if __name__ == "__main__":
    run_pipeline()