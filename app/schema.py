"""Request/response schemas and validation for the Flask API."""

from dataclasses import dataclass, asdict
from typing import List, Optional


ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE_MB = 10


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@dataclass
class PredictionResult:
    predicted_role: str
    confidence: float
    top_predictions: List[dict]
    filename: str
    status: str = "success"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorResponse:
    status: str
    message: str
    code: int = 400

    def to_dict(self) -> dict:
        return asdict(self)
