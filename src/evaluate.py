
from src.utils import setup_logging
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = setup_logging()


def evaluate_model(model, X_test, y_test, label_encoder) -> dict:
    """
    Evaluate a trained model on the test set.

    Returns:
        dict with accuracy, per-class metrics
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Test Accuracy: {acc * 100:.2f}%")

    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    # Pretty print summary
    print("\n" + "=" * 60)
    print(f"  ACCURACY:  {acc * 100:.2f}%")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
          target_names=label_encoder.classes_))
    print("=" * 60 + "\n")

    return {
        "accuracy": acc,
        "report": report,
    }
