
import re
import string
import pandas as pd
from src.utils import setup_logging, clean_text

logger = setup_logging()

# Skills/stopwords to keep (domain-specific)
RESUME_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "same", "so", "than", "too", "very",
    "just", "can", "will", "don", "should", "now",
}


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single resume text.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove non-alphanumeric chars
        4. Remove stopwords
        5. Normalize whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = clean_text(text)

    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in RESUME_STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the Resume column of a DataFrame."""
    df = df.copy()
    logger.info("Preprocessing resume texts...")
    df["resume_cleaned"] = df["resume"].apply(preprocess_text)
    df["category_cleaned"] = df["category"].apply(preprocess_text)
    logger.info("Preprocessing complete.")
    return df


# if __name__ == "__main__":
#     sample = "I am a Python Developer with 3 years experience in Django, Flask and REST APIs."
#     print("Original:", sample)
#     print("Processed:", preprocess_text(sample))
