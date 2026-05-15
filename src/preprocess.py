import re
import string
from pathlib import Path

import pandas as pd

import nltk
from nltk.corpus import stopwords

# Download stopwords on first run
nltk.download("stopwords", quiet=True)
EN_STOPWORDS = set(stopwords.words("english"))

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "fake_or_real_news.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_fake_or_real_news.csv"


def clean_text(text: str) -> str:
    """Clean a single news article string."""
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3. Remove @mentions and #hashtags
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)

    # 4. Remove emojis and non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # 5. Remove numbers
    text = re.sub(r"\d+", " ", text)

    # 6. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 7. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 8. Remove stopwords
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in EN_STOPWORDS]

    return " ".join(tokens)


def main():
    # Load original dataset
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Map labels: FAKE -> 0, REAL -> 1  (clear for viva)
    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})

    # Keep only useful columns
    df = df[["title", "text", "label"]].dropna(subset=["text"])

    print("Total rows:", len(df))

    # Apply cleaning
    print("Cleaning text... please wait...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Show sample before/after
    print("\nSample BEFORE cleaning:\n", df["text"].iloc[0][:200], "...")
    print("\nSample AFTER cleaning:\n", df["clean_text"].iloc[0][:200], "...")

    # Save cleaned data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()