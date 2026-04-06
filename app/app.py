
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from app.schema import allowed_file, MAX_FILE_SIZE_MB, PredictionResult, ErrorResponse
from src.predict import predict_role
from src.utils import setup_logging, get_project_root, load_config, load_artifacts, load_model
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


# ── Setup ──────────────────────────────────────────────────────────────────────
root = get_project_root()
config = load_config(os.path.join(root, "config.yaml"))
logger = setup_logging(
    log_file=os.path.join(root, "logs", "app.log"),
    level=config["logging"]["level"],
)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

# ── Load model once at startup ─────────────────────────────────────────────────
try:
    MODEL = load_model(os.path.join(root, "models", "model_v1.pkl"))
    ARTIFACTS = load_artifacts(os.path.join(
        root, "models", "feature_artifacts.pkl"))
    logger.info("Model and artifacts loaded successfully.")
except FileNotFoundError as e:
    logger.warning(f"Model not found: {e}. Run `python src/train.py` first.")
    MODEL = None
    ARTIFACTS = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL is not None,
        "version": config["app"]["version"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict job role from uploaded resume.

    Request: multipart/form-data with key 'file'
    Response: JSON with predicted_role, confidence, top_predictions
    """
    if MODEL is None or ARTIFACTS is None:
        err = ErrorResponse(
            status="error",
            message="Model not loaded. Run `python src/train.py` to train first.",
            code=503,
        )
        return jsonify(err.to_dict()), 503

    # Validate file presence
    if "file" not in request.files:
        return jsonify(ErrorResponse("error", "No file provided. Use key 'file'.", 400).to_dict()), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify(ErrorResponse("error", "Empty filename.", 400).to_dict()), 400

    if not allowed_file(file.filename):
        return jsonify(ErrorResponse(
            "error",
            "Unsupported file type. Upload a .pdf, .docx, or .txt file.",
            415,
        ).to_dict()), 415

    filename = secure_filename(file.filename)
    file_bytes = file.read()

    if len(file_bytes) == 0:
        return jsonify(ErrorResponse("error", "Uploaded file is empty.", 400).to_dict()), 400

    try:
        result = predict_role(
            file_bytes=file_bytes,
            filename=filename,
            model=MODEL,
            artifacts=ARTIFACTS,
            top_n=5,
        )
        response = PredictionResult(
            predicted_role=result["predicted_role"],
            confidence=result["confidence"],
            top_predictions=result["top_predictions"],
            filename=filename,
        )
        logger.info(
            f"Predicted '{result['predicted_role']}' ({result['confidence']}%) for {filename}")
        return jsonify(response.to_dict()), 200

    except ValueError as e:
        logger.warning(f"Prediction error for {filename}: {e}")
        return jsonify(ErrorResponse("error", str(e), 422).to_dict()), 422

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify(ErrorResponse("error", "Internal server error.", 500).to_dict()), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify(ErrorResponse(
        "error", f"File too large. Max size: {MAX_FILE_SIZE_MB}MB.", 413
    ).to_dict()), 413


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
