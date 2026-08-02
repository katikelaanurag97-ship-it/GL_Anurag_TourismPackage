from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch",
]


def register_dataset() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Upload tourism.csv to data/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        raise ValueError(
            f"Dataset validation failed. Missing columns: {missing_columns}"
        )

    print("Dataset validation successful")
    print(f"Path              : {DATA_PATH}")
    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Duplicate rows    : {int(df.duplicated().sum())}")
    print(f"Unexpected columns: {unexpected_columns or 'None'}")
    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts(dropna=False).sort_index())
    print("\nColumns with missing values:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False) if missing.any() else "None")


if __name__ == "__main__":
    register_dataset()
