

from model_trainer import ModelTrainer
from src.utils import setup_logging, get_project_root, load_config
from src.evaluate import evaluate_model
from src.features import fit_transform_features, save_artifacts
from src.preprocess import preprocess_dataframe
from src.data_loader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import sys
import joblib

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = setup_logging()


def train(csv_path: str = None):
    """Full training pipeline: load → preprocess → features → train → evaluate → save."""
    root = get_project_root()
    config = load_config(os.path.join(root, "config.yaml"))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Step 1: Loading dataset...")
    df = load_dataset(csv_path)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    logger.info("Step 2: Preprocessing...")
    df = preprocess_dataframe(df)

    logger.info("Step 3: Train/Test Split...")
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        df["resume_cleaned"],
        df["category_cleaned"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["category_cleaned"]
    )

    logger.info(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

    logger.info("Step 4: Extracting features...")
    X_train, y_train, vectorizer, label_encoder = fit_transform_features(
        X_train_text, y_train_text
    )

    X_test = vectorizer.transform(X_test_text)
    y_test = label_encoder.transform(y_test_text)

    # # ── 3. Feature engineering ────────────────────────────────────────────────
    # logger.info("Step 3: Extracting features...")
    # X, y, vectorizer, label_encoder = fit_transform_features(
    #     df["resume_cleaned"], df["category_cleaned"]
    # )

    # # ── 4. Train/test split ───────────────────────────────────────────────────
    # test_size = config["model"]["test_size"]
    # random_state = config["model"]["random_state"]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=random_state, stratify=y
    # )
    # logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── 5. Train model ────────────────────────────────────────────────────────
    logger.info("Step 5: Training Logistic Regression...")
    # model = LogisticRegression(
    #     max_iter=1000,
    #     C=5.0,
    #     solver="lbfgs",
    #     random_state=random_state,
    # )
    # model.fit(X_train, y_train)
    model = ModelTrainer()

    best_model, best_score, report = model.initiate_model_trainer(
        X_train, y_train, X_test, y_test
    )

    # print("Best Model:", best_model)
    # print("Best Accuracy:", best_score)
    # print("All Models:", report)
    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    logger.info("Step 6: Evaluating...")
    evaluate_model(best_model, X_test, y_test, label_encoder)

    # ── 7. Save model & artifacts ─────────────────────────────────────────────
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model_v1.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved = {model_path}")

    save_artifacts(vectorizer, label_encoder,
                   os.path.join(model_dir, "feature_artifacts.pkl"))

    logger.info("Training complete and artifacts saved.")
    return best_model, vectorizer, label_encoder


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    train(csv)
