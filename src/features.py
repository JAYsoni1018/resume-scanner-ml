
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.utils import setup_logging, get_project_root, load_config

logger = setup_logging()
config = load_config(os.path.join(get_project_root(), "config.yaml"))


def build_vectorizer(max_features: int = None) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with good defaults for resume text."""
    if max_features is None:
        max_features = config["model"]["vectorizer_max_features"]

    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),        # unigrams + bigrams
        sublinear_tf=True,         # log normalization
        min_df=2,                  # ignore very rare terms
        max_df=0.95,               # ignore very common terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r'(?u)\b\w+\b',    # 🔥 keep C, R, SQL
        lowercase=True,
        dtype=np.float32                # 🔥 memory optimization
    )


def fit_transform_features(texts, labels):
    """
    Fit vectorizer and label encoder, then transform.

    Returns:
        X (sparse matrix), y (ndarray), vectorizer, label_encoder
    """
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    logger.info(f"Feature matrix: {X.shape}, Classes: {len(le.classes_)}")
    return X, y, vectorizer, le


# def transform_features(texts, vectorizer: TfidfVectorizer):
#     """Transform new texts using a fitted vectorizer."""
#     return vectorizer.transform(texts)


def save_artifacts(vectorizer, label_encoder, path: str = None):
    """Save vectorizer and label encoder as a tuple."""
    if path is None:
        path = os.path.join(get_project_root(), "models",
                            "feature_artifacts.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"vectorizer": vectorizer,
                "label_encoder": label_encoder}, path)
    logger.info(f"Feature artifacts saved to {path}")


# def load_artifacts(path: str = None) -> dict:
#     """Load vectorizer and label encoder."""
#     if path is None:
#         path = os.path.join(get_project_root(), "models", "feature_artifacts.pkl")
#     return joblib.load(path)
