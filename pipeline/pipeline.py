"""
End-to-end ML pipeline: orchestrates training and inference steps.
Can be run directly to retrain the model from scratch.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocess import preprocess_dataframe
from src.evaluate import evaluate_model
from src.utils import setup_logging, get_project_root, load_config

logger = setup_logging()


def build_sklearn_pipeline(max_features: int = 10000) -> Pipeline:
    """
    Build a scikit-learn Pipeline combining TF-IDF and Logistic Regression.
    Note: label encoding is handled separately (Pipeline doesn't support it natively).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def run_pipeline(csv_path: str = None):
    """
    Full pipeline run:
      1. Load & preprocess data
      2. Split train/test
      3. Fit pipeline
      4. Evaluate
      5. Save model + label encoder
    """
    root = get_project_root()
    config = load_config(os.path.join(root, "config.yaml"))

    logger.info("Pipeline: Loading data...")
    df = load_dataset(csv_path)
    df = preprocess_dataframe(df)

    X = df["cleaned_resume"]
    le = LabelEncoder()
    y = le.fit_transform(df["Category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    logger.info("Pipeline: Training...")
    pipe = build_sklearn_pipeline(config["model"]["vectorizer_max_features"])
    pipe.fit(X_train, y_train)

    logger.info("Pipeline: Evaluating...")
    evaluate_model(pipe, X_test, y_test, le)

    # Save
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(pipe, os.path.join(models_dir, "model_v1.pkl"))
    joblib.dump(
        {"label_encoder": le},
        os.path.join(models_dir, "feature_artifacts.pkl"),
    )
    logger.info("Pipeline: Saved model_v1.pkl and feature_artifacts.pkl ✓")

    return pipe, le


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(csv)
