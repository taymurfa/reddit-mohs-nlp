"""
02_preprocess.py
================
Clean and tokenize the raw Reddit text collected in 01_collect.py.

What this script does, step by step:
  1. Reads data/raw/mohs_raw.csv (produced by 01_collect.py).
  2. Downloads required NLTK resources on first run (stopwords, wordnet).
  3. For each row, applies a cleaning pipeline to the `body` column:
       a. Lowercase everything
       b. Remove URLs (http/https links)
       c. Remove punctuation and digits
       d. Tokenize into individual words
       e. Remove English stopwords (NLTK's built-in list)
       f. Lemmatize each token (reduce to dictionary base form)
  4. Drops any row whose cleaned token list has fewer than 5 tokens
     (very short posts/comments add noise without meaningful signal).
  5. Saves the cleaned data to data/processed/mohs_clean.csv, keeping all
     original columns and adding a new `tokens` column that stores the
     cleaned token list as a space-separated string.

Run from the project root:
    python scripts/02_preprocess.py
"""

import os
import re
import ast
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH  = os.path.join("data", "raw",       "mohs_raw.csv")
OUTPUT_PATH = os.path.join("data", "processed", "mohs_clean.csv")

# Minimum number of tokens required to keep a row
MIN_TOKENS = 5

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── NLTK resource download ─────────────────────────────────────────────────────

def ensure_nltk_resources():
    """
    Download required NLTK corpora if they are not already on disk.
    These are small one-time downloads (~3 MB total).
    """
    resources = {
        "corpora/stopwords":         "stopwords",
        "corpora/wordnet":           "wordnet",
        "corpora/omw-1.4":           "omw-1.4",   # multilingual wordnet (needed by WordNetLemmatizer)
        "tokenizers/punkt":          "punkt",
        "tokenizers/punkt_tab":      "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            log.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


# ── Text cleaning pipeline ─────────────────────────────────────────────────────

# Compiled regex patterns (compiled once for speed across thousands of rows)
URL_PATTERN   = re.compile(r"https?://\S+|www\.\S+")
NONALPHA_PATTERN = re.compile(r"[^a-z\s]")   # keep only lowercase letters and spaces


def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> list[str]:
    """
    Apply the full cleaning pipeline to a single text string.

    Returns a list of clean tokens, or an empty list if the input is invalid.

    Pipeline:
      1. Validate input (skip NaN / non-string values)
      2. Lowercase
      3. Remove URLs
      4. Remove punctuation and digits (keep only letters and spaces)
      5. Tokenize by whitespace
      6. Remove stopwords
      7. Lemmatize
      8. Drop tokens shorter than 2 characters (single letters add no value)
    """
    # Step 1 — validate
    if not isinstance(text, str) or not text.strip():
        return []

    # Step 2 — lowercase
    text = text.lower()

    # Step 3 — remove URLs
    text = URL_PATTERN.sub(" ", text)

    # Step 4 — remove punctuation and digits
    text = NONALPHA_PATTERN.sub(" ", text)

    # Step 5 — tokenize (split on whitespace; fast and sufficient after cleaning)
    tokens = text.split()

    # Step 6 — remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Step 7 — lemmatize (default POS is noun; covers most medical vocabulary well)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # Step 8 — drop very short tokens
    tokens = [t for t in tokens if len(t) > 1]

    return tokens


# ── Main preprocessing logic ───────────────────────────────────────────────────

def main():
    # ── Step 1: Ensure NLTK data is available ─────────────────────────────────
    log.info("Checking NLTK resources ...")
    ensure_nltk_resources()

    stop_words  = set(stopwords.words("english"))
    lemmatizer  = WordNetLemmatizer()
    log.info(f"Stopwords loaded ({len(stop_words)} words). Lemmatizer ready.")

    # ── Step 2: Load raw data ─────────────────────────────────────────────────
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run 01_collect.py first to generate the raw data."
        )

    log.info(f"Reading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH, dtype=str)   # read as str to avoid type surprises
    log.info(f"Loaded {len(df):,} rows.")

    # ── Step 3: Clean each row's body text ───────────────────────────────────
    log.info("Cleaning text ...")
    tqdm.pandas(desc="Cleaning rows")
    df["tokens"] = df["body"].progress_apply(
        lambda text: clean_text(text, stop_words, lemmatizer)
    )

    # ── Step 4: Filter out rows with too few tokens ───────────────────────────
    before = len(df)
    df = df[df["tokens"].apply(len) >= MIN_TOKENS].copy()
    after = len(df)
    log.info(
        f"Filtered rows with < {MIN_TOKENS} tokens: "
        f"{before - after:,} removed, {after:,} kept "
        f"({100 * after / before:.1f}% retained)."
    )

    # ── Step 5: Serialize token lists as space-separated strings ──────────────
    # Storing as a plain string (rather than JSON) makes the CSV human-readable
    # and easy to reload with a simple .split() call in 03_model.py.
    df["tokens"] = df["tokens"].apply(lambda toks: " ".join(toks))

    # ── Step 6: Save to disk ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    log.info(f"Saved cleaned data to {OUTPUT_PATH}")

    # ── Step 7: Print a quick sample so the user can sanity-check ─────────────
    log.info("\nSample of cleaned tokens (first 5 rows):")
    for _, row in df.head(5).iterrows():
        preview = row["tokens"][:120]   # truncate long token strings for display
        log.info(f"  [{row['type']}] {preview} ...")


if __name__ == "__main__":
    main()
