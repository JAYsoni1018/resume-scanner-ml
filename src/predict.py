
from src.preprocess import preprocess_text
from src.utils import setup_logging, get_project_root, load_artifacts, load_model
import numpy as np
import joblib
import re
import io
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = setup_logging()


def extract_text_from_pdf(file_bytes):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            return pdfminer_extract(io.BytesIO(file_bytes))
        except ImportError:
            raise RuntimeError("Install PyPDF2: pip install PyPDF2")


def extract_text_from_docx(file_bytes):
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return " ".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(file_bytes, filename):
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext == ".txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}. Use .pdf, .docx, or .txt")


# def _preprocess(text):

#     text = text.lower()

#     text = re.sub(r"http\S+|www\S+", " ", text)
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()


# def load_artifacts(path=None):
#     if path is None:
#         path = os.path.join(get_project_root(), "models", "feature_artifacts.pkl")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Feature artifacts not found at {path}.")
#     return joblib.load(path)


# def predict_role(file_bytes, filename, model=None, artifacts=None, top_n=5):
#     """
#     Predict job role from resume bytes.

#     Returns:
#         {"predicted_role": str, "confidence": float, "top_predictions": [...]}
#     """
#     if model is None:
#         model = load_model()
#     if artifacts is None:
#         artifacts = load_artifacts()

#     label_encoder = artifacts["label_encoder"]

#     raw_text = extract_text(file_bytes, filename)
#     if not raw_text.strip():
#         raise ValueError("Could not extract any text from the uploaded file.")

#     cleaned = preprocess_text(raw_text)

#     # model is a sklearn Pipeline (tfidf + clf)
#     proba = model.predict_proba([cleaned])[0]
#     top_indices = np.argsort(proba)[::-1][:top_n]

#     top_predictions = [
#         {
#             "role": label_encoder.classes_[i],
#             "confidence": round(float(proba[i]) * 100, 2),
#         }
#         for i in top_indices
#     ]

#     return {
#         "predicted_role": top_predictions[0]["role"],
#         "confidence": top_predictions[0]["confidence"],
#         "top_predictions": top_predictions,
#     }

def predict_role(file_bytes, filename, model=None, artifacts=None, top_n=5):

    if model is None:
        model = load_model()
    if artifacts is None:
        artifacts = load_artifacts()

    vectorizer = artifacts["vectorizer"]
    label_encoder = artifacts["label_encoder"]

    raw_text = extract_text(file_bytes, filename)

    if not raw_text.strip():
        raise ValueError("Could not extract any text from the uploaded file.")

    cleaned = preprocess_text(raw_text)

    X = vectorizer.transform([cleaned])

    proba = model.predict_proba(X)[0]

    top_indices = np.argsort(proba)[::-1][:top_n]

    top_predictions = [
        {
            "role": label_encoder.classes_[i],
            "confidence": round(float(proba[i]) * 100, 2),
        }
        for i in top_indices
    ]

    return {
        "predicted_role": top_predictions[0]["role"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }


# if __name__ == "__main__":
#     sample = b"Results-oriented HR Professional with a proven track record of aligning human resources strategies with organizational goals. Expert in streamlining recruitment processes, enhancing employee engagement, and ensuring rigorous compliance with labor laws. Adept at fostering a positive corporate culture through effective conflict resolution, performance management, and comprehensive training programs. Committed to leveraging data-driven insights to optimize workforce productivity and support long-term business growth."
#     result = predict_role(sample, "test.txt")
#     print(result)
