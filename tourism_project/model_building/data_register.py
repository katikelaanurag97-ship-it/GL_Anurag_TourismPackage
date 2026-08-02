
from pathlib import Path
import pandas as pd


DATA_PATH = Path("/content/tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch",
]


def register_dataset() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    unexpected_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if missing_columns:
        raise ValueError(
            f"Dataset validation failed. Missing columns: {missing_columns}"
        )

    print("Dataset validation successful")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Unexpected columns: {unexpected_columns or 'None'}")
    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts(dropna=False))
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))


if __name__ == "__main__":
    register_dataset()
