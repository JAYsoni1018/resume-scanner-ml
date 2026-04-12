
import os
import pandas as pd
from src.utils import setup_logging, get_project_root

logger = setup_logging()


def load_dataset(csv_path: str = None) -> pd.DataFrame:

    if csv_path is None:
        root = get_project_root()
        csv_path = os.path.join(root, "data", "processed_resume_dataset.csv")

    df = pd.read_csv(csv_path)
    logger.info(
        f"Loaded dataset: {len(df)} records, {df['category'].nunique()} categories")

    # Validate required columns
    required = {"category", "resume"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns: {required}. Found: {set(df.columns)}")

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["category", "resume"])
    df = df[df["resume"].str.strip().astype(bool)]
    logger.info(
        f"After cleaning: {len(df)} records (dropped {before - len(df)})")

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
