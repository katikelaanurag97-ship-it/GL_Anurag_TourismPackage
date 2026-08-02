from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
TARGET = "ProdTaken"
DROP_COLUMNS = ["Unnamed: 0", "CustomerID"]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace(
            {"Fe Male": "Female", "FeMale": "Female"}
        )
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace(
            {"Unmarried": "Single"}
        )

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    removable = [
        column for column in df.columns
        if column != TARGET and (
            df[column].isna().all() or df[column].nunique(dropna=False) <= 1
        )
    ]
    return df.drop(columns=removable, errors="ignore")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)

    train_df, test_df = train_test_split(
        cleaned_df,
        test_size=0.20,
        random_state=42,
        stratify=cleaned_df[TARGET],
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
        "removed_identifier_columns": [
            c for c in DROP_COLUMNS if c in raw_df.columns
        ],
        "duplicate_rows_removed": int(raw_df.duplicated().sum()),
        "target_distribution_train": {
            str(k): int(v) for k, v in train_df[TARGET].value_counts().items()
        },
        "target_distribution_test": {
            str(k): int(v) for k, v in test_df[TARGET].value_counts().items()
        },
    }

    summary_path = REPORT_DIR / "data_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Data preparation completed")
    print(f"Raw shape     : {raw_df.shape}")
    print(f"Cleaned shape : {cleaned_df.shape}")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape : {test_df.shape}")
    print(f"Train artifact: {train_path}")
    print(f"Test artifact : {test_path}")


if __name__ == "__main__":
    main()
