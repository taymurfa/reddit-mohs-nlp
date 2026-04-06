"""
03_model.py
===========
Train LDA topic models on the preprocessed Reddit data and evaluate them.

What this script does, step by step:
  1. Reads data/processed/mohs_clean.csv (produced by 02_preprocess.py).
  2. Converts the space-separated `tokens` column into lists of words.
  3. Builds a Gensim Dictionary (vocabulary) and Bag-of-Words corpus.
  4. Trains LDA models for k = 5, 10, 15, and 20 topics (passes=10).
  5. Computes the c_v coherence score for each k and prints a summary table.
  6. Saves the best model (highest coherence score) to outputs/topics/.
  7. Prints the top 10 words for every topic in the best model.
  8. Generates a pyLDAvis interactive HTML visualization and saves it to
     outputs/figures/lda_vis.html.

Run from the project root:
    python scripts/03_model.py
"""

import os
import logging
import warnings

import pandas as pd
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from topic_graph import generate_topic_graph

# Suppress noisy deprecation warnings from pyLDAvis / gensim internals
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_PATH      = os.path.join("data", "processed", "mohs_clean.csv")
MODEL_DIR       = os.path.join("outputs", "topics")
VIS_OUTPUT_PATH = os.path.join("outputs", "figures", "lda_vis.html")

# Topic counts to evaluate
K_VALUES = [5, 10, 15, 20]

# LDA training parameters
LDA_PASSES       = 10    # number of passes through the corpus during training
LDA_ITERATIONS   = 50    # maximum iterations per pass
LDA_RANDOM_STATE = 42    # for reproducibility

# Dictionary filtering thresholds:
#   - no_below: ignore tokens that appear in fewer than N documents
#   - no_above: ignore tokens that appear in more than X% of documents
DICT_NO_BELOW = 5
DICT_NO_ABOVE = 0.5

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Quiet gensim's own verbose logging (it can be very chatty)
logging.getLogger("gensim").setLevel(logging.WARNING)


# ── Helper functions ───────────────────────────────────────────────────────────

def load_tokens(csv_path: str) -> list[list[str]]:
    """
    Load the cleaned CSV and return a list of token lists.

    Each element is one document (post or comment) represented as a list
    of individual word strings.
    """
    log.info(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, dtype=str)

    # The `tokens` column was saved as space-separated strings in 02_preprocess.py
    token_lists = df["tokens"].dropna().apply(str.split).tolist()
    log.info(f"Loaded {len(token_lists):,} documents.")
    return token_lists


def build_corpus(token_lists: list[list[str]]):
    """
    Build a Gensim Dictionary and BoW corpus from token lists.

    Returns:
        dictionary : gensim.corpora.Dictionary
            Maps each unique token to an integer ID.
        corpus : list of list of (int, int) tuples
            Bag-of-Words representation of every document.
    """
    log.info("Building Gensim dictionary ...")
    dictionary = corpora.Dictionary(token_lists)
    vocab_size_before = len(dictionary)

    # Filter out very rare and very common tokens — they hurt coherence scores
    dictionary.filter_extremes(no_below=DICT_NO_BELOW, no_above=DICT_NO_ABOVE)
    log.info(
        f"Dictionary filtered: {vocab_size_before:,} → {len(dictionary):,} tokens "
        f"(no_below={DICT_NO_BELOW}, no_above={DICT_NO_ABOVE*100:.0f}%)"
    )

    log.info("Building BoW corpus ...")
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    log.info(f"Corpus ready: {len(corpus):,} documents.")
    return dictionary, corpus


def train_lda(corpus, dictionary, num_topics: int) -> LdaModel:
    """Train a single LDA model with the given number of topics."""
    log.info(f"  Training LDA with k={num_topics} ...")
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=LDA_PASSES,
        iterations=LDA_ITERATIONS,
        random_state=LDA_RANDOM_STATE,
        # alpha='auto' lets the model learn document-topic density from data
        alpha="auto",
        # eta='auto' lets the model learn topic-word density from data
        eta="auto",
        per_word_topics=True,   # required for pyLDAvis compatibility
    )
    return model


def compute_coherence(model: LdaModel, token_lists, dictionary) -> float:
    """
    Compute the c_v coherence score for a trained LDA model.

    c_v coherence correlates well with human judgements of topic quality
    (Röder et al., 2015). Higher is better; typical good range is 0.4–0.7.
    """
    coherence_model = CoherenceModel(
        model=model,
        texts=token_lists,
        dictionary=dictionary,
        coherence="c_v",
    )
    return coherence_model.get_coherence()


def print_topics(model: LdaModel, num_words: int = 10):
    """Print the top N words for every topic in the model."""
    log.info(f"\nTop {num_words} words per topic:")
    print("-" * 60)
    for topic_id in range(model.num_topics):
        words = model.show_topic(topic_id, topn=num_words)
        word_str = "  |  ".join(f"{w} ({p:.3f})" for w, p in words)
        print(f"  Topic {topic_id + 1:>2}: {word_str}")
    print("-" * 60)


def save_best_model(model: LdaModel, dictionary, best_k: int):
    """Save the best LDA model and its dictionary to outputs/topics/."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lda_k{best_k}")
    model.save(model_path)
    dict_path = os.path.join(MODEL_DIR, "dictionary.gensim")
    dictionary.save(dict_path)
    log.info(f"Best model (k={best_k}) saved to {model_path}.*")
    log.info(f"Dictionary saved to {dict_path}")


def generate_visualization(model: LdaModel, corpus, dictionary):
    """
    Generate a pyLDAvis interactive HTML visualization.

    pyLDAvis uses a dimensionality-reduction technique (Jensen-Shannon
    divergence + MDS) to position topics in 2D space. Topics that are
    closer together share more vocabulary. The right-hand panel shows the
    top terms for each selected topic.
    """
    log.info("Generating pyLDAvis visualization (this may take a moment) ...")
    os.makedirs(os.path.dirname(VIS_OUTPUT_PATH), exist_ok=True)

    vis_data = gensimvis.prepare(
        model,
        corpus,
        dictionary,
        sort_topics=False,   # keep topic IDs stable for cross-referencing
    )
    pyLDAvis.save_html(vis_data, VIS_OUTPUT_PATH)
    log.info(f"Visualization saved to {VIS_OUTPUT_PATH}")


# ── Main modeling logic ────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load token lists ───────────────────────────────────────────────
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run 02_preprocess.py first to generate the cleaned data."
        )

    token_lists = load_tokens(INPUT_PATH)

    # ── Step 2: Build dictionary and corpus ───────────────────────────────────
    dictionary, corpus = build_corpus(token_lists)

    # ── Step 3: Train LDA models for each k, record coherence scores ──────────
    log.info(f"\nTraining LDA models for k in {K_VALUES} ...")

    results = []   # list of (k, model, coherence_score)

    for k in K_VALUES:
        model     = train_lda(corpus, dictionary, num_topics=k)
        coherence = compute_coherence(model, token_lists, dictionary)
        results.append((k, model, coherence))
        log.info(f"  k={k:>2}  coherence (c_v) = {coherence:.4f}")

    # ── Step 4: Print coherence summary table ─────────────────────────────────
    print("\n" + "=" * 40)
    print(f"  {'k (topics)':<14} {'Coherence (c_v)':>15}")
    print("=" * 40)
    for k, _, coherence in results:
        marker = " ← best" if k == max(results, key=lambda r: r[2])[0] else ""
        print(f"  {k:<14} {coherence:>15.4f}{marker}")
    print("=" * 40 + "\n")

    # ── Step 5: Identify the best model ───────────────────────────────────────
    best_k, best_model, best_coherence = max(results, key=lambda r: r[2])
    log.info(f"Best model: k={best_k} with coherence = {best_coherence:.4f}")

    # ── Step 6: Save the best model ───────────────────────────────────────────
    save_best_model(best_model, dictionary, best_k)

    # ── Step 7: Print top words for each topic in the best model ──────────────
    print_topics(best_model, num_words=10)

    # ── Step 8: Generate interactive HTML visualization ───────────────────────
    generate_visualization(best_model, corpus, dictionary)

    # ── Step 9: Generate Obsidian-style topic graph ────────────────────────────
    generate_topic_graph(best_model, corpus, dictionary, best_k)

    log.info("\nAll done!")
    log.info("  pyLDAvis  → outputs/figures/lda_vis.html")
    log.info("  Topic graph → outputs/figures/topic_graph.html")


if __name__ == "__main__":
    main()
