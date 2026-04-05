# Mohs Surgery Reddit — LDA Topic Modeling

Collects posts and comments from r/MohsSurgery and identifies latent topics using Latent Dirichlet Allocation (LDA).

## Project structure

```
mohs-reddit-lda/
├── data/
│   ├── raw/               # Raw CSV from Reddit API (gitignored)
│   └── processed/         # Cleaned, tokenized CSV (gitignored)
├── outputs/
│   ├── topics/            # Saved LDA model files
│   └── figures/           # lda_vis.html interactive visualization
├── scripts/
│   ├── 01_collect.py      # Fetch posts + comments from Reddit
│   ├── 02_preprocess.py   # Clean, tokenize, lemmatize
│   └── 03_model.py        # Train LDA, evaluate, visualize
├── .env.example           # Credential template
├── .gitignore
└── requirements.txt
```

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd mohs-reddit-lda
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Reddit API credentials

```bash
cp .env.example .env
```

Edit `.env` with your credentials from https://www.reddit.com/prefs/apps  
(create a **script** type app).

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=mohs_research_bot/1.0 by u/your_username
```

## Running the pipeline

Run each script in order from the **project root**:

```bash
python scripts/01_collect.py      # ~10–30 min depending on subreddit size
python scripts/02_preprocess.py   # ~1–2 min
python scripts/03_model.py        # ~5–15 min depending on corpus size
```

### Outputs

| File | Description |
|------|-------------|
| `data/raw/mohs_raw.csv` | All posts and comments with metadata |
| `data/processed/mohs_clean.csv` | Cleaned data with `tokens` column |
| `outputs/topics/lda_k{N}.*` | Best LDA model (Gensim format) |
| `outputs/topics/dictionary.gensim` | Gensim vocabulary dictionary |
| `outputs/figures/lda_vis.html` | Interactive topic visualization |

Open `outputs/figures/lda_vis.html` in any browser — no server required.

## Adjusting parameters

| Script | Variable | Default | What it controls |
|--------|----------|---------|-----------------|
| `03_model.py` | `K_VALUES` | `[5,10,15,20]` | Topic counts to compare |
| `03_model.py` | `LDA_PASSES` | `10` | Training thoroughness (higher = slower but better) |
| `03_model.py` | `DICT_NO_BELOW` | `5` | Min document frequency to keep a token |
| `03_model.py` | `DICT_NO_ABOVE` | `0.5` | Max document frequency (fraction) to keep a token |
| `02_preprocess.py` | `MIN_TOKENS` | `5` | Min tokens required to keep a document |
