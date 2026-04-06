

import os
import re
import logging
import joblib
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    """Configure logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("resume_classifier")


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove special chars, normalize whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # keep only alphanum
    text = re.sub(r"\s+", " ", text).strip()             # normalize spaces
    return text


def get_project_root() -> Path:
    """Return the absolute path to the project root."""
    return Path(__file__).resolve().parent.parent


# def save_artifacts(vectorizer, label_encoder, path: str = None):
#     """Save vectorizer and label encoder as a tuple."""
#     if path is None:
#         path = os.path.join(get_project_root(), "models",
#                             "feature_artifacts.pkl")
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     joblib.dump({"vectorizer": vectorizer,
#                 "label_encoder": label_encoder}, path)
#     setup_logging.logger.info(f"Feature artifacts saved to {path}")


# def load_artifacts(path: str = None) -> dict:
#     """Load vectorizer and label encoder."""
#     if path is None:
#         path = os.path.join(get_project_root(), "models",
#                             "feature_artifacts.pkl")
#     return joblib.load(path)


def load_artifacts(path=None):
    if path is None:
        path = os.path.join(get_project_root(), "models",
                            "feature_artifacts.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature artifacts not found at {path}.")
    return joblib.load(path)


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(get_project_root(), "models", "model_v1.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python generate_and_train.py"
        )
    return joblib.load(model_path)
