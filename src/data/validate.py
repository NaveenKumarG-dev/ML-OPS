from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]

BINARY_COLUMNS = [
    "default",
    "housing",
    "loan",
    "y",
    
]

NUMERIC_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset has the expected columns."""

    actual_columns = df.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Got: {actual_columns}"
        )


def validate_binary_columns(df: pd.DataFrame) -> None:
    """Validate values in binary columns."""

    for column in BINARY_COLUMNS:
        values = set(df[column].dropna().unique())

        if not values.issubset({"yes", "no"}):
            raise ValueError(
                f"Invalid values found in '{column}': {values}"
            )


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """Validate that expected numeric columns contain numeric data."""

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' is not numeric."
            )

def validate_missing_values(df: pd.DataFrame) -> None:
    """Validate that the dataset contains no missing values."""

    missing_values = df.isnull().sum()

    columns_with_missing = missing_values[missing_values > 0]

    if not columns_with_missing.empty:
        raise ValueError(
            f"Missing values found:\n{columns_with_missing}"
        )

def validate_dataset(df: pd.DataFrame) -> None:
    """Run all dataset validation checks."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    validate_columns(df)
    validate_binary_columns(df)
    validate_numeric_columns(df)
    validate_missing_values(df)

if __name__ == "__main__":
    data_path = Path("data/raw/bank-full.csv")

    df = pd.read_csv(data_path, sep=";")

    validate_dataset(df)

    print("Dataset validation passed.")