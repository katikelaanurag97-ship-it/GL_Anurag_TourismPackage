from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path("/content/tourism_project/data/tourism.csv")
ARTIFACT_DIR = Path("/content/tourism_project/artifacts")
REPORT_DIR = Path("/content/tourism_project/reports")
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names and remove duplicate records.
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove non-predictive index and identifier columns.
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Trim text values.
    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()

    # Correct known category inconsistencies.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace(
            {"Fe Male": "Female", "FeMale": "Female"}
        )

    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace(
            {"Unmarried": "Single"}
        )

    # Remove records that cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Remove completely empty and constant feature columns.
    empty_columns = [
        column for column in df.columns
        if column != TARGET and df[column].isna().all()
    ]
    constant_columns = [
        column for column in df.columns
        if column != TARGET and df[column].nunique(dropna=False) <= 1
    ]
    df = df.drop(
        columns=sorted(set(empty_columns + constant_columns)),
        errors="ignore"
    )

    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.20,
        random_state=42,
        stratify=cleaned_df[TARGET]
    )

    train_path = ARTIFACT_DIR / "train.csv"
    test_path = ARTIFACT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "raw_shape": list(raw_df.shape),
        "cleaned_shape": list(cleaned_df.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "removed_columns": [
            column for column in DROP_COLUMNS if column in raw_df.columns
        ],
        "duplicate_rows_removed": int(raw_df.duplicated().sum()),
        "target_distribution_full": {
            str(k): int(v)
            for k, v in cleaned_df[TARGET].value_counts().items()
        },
        "target_distribution_train": {
            str(k): int(v)
            for k, v in train_df[TARGET].value_counts().items()
        },
        "target_distribution_test": {
            str(k): int(v)
            for k, v in test_df[TARGET].value_counts().items()
        },
        "remaining_missing_values": {
            k: int(v)
            for k, v in cleaned_df.isna().sum().items() if v > 0
        }
    }

    summary_path = REPORT_DIR / "data_preparation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("=" * 65)
    print("DATA PREPARATION SUMMARY")
    print("=" * 65)
    print(f"Raw shape              : {raw_df.shape}")
    print(f"Cleaned shape          : {cleaned_df.shape}")
    print(f"Training shape         : {train_df.shape}")
    print(f"Testing shape          : {test_df.shape}")
    print(f"Removed columns        : {summary['removed_columns']}")
    print(f"Train file             : {train_path}")
    print(f"Test file              : {test_path}")
    print(f"Summary file           : {summary_path}")
    print("\\nTraining target distribution:")
    print(train_df[TARGET].value_counts(normalize=True).sort_index())
    print("\\nTesting target distribution:")
    print(test_df[TARGET].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
