# SubredditLDA

A cross-platform desktop application built with Electron and Python to collect Reddit data and perform advanced LDA topic modeling, sentiment analysis, and topic summarization on subreddits.

## Key Features

- **Reddit Data Collection:** Harvest subreddit posts and comments efficiently while cleanly complying with Reddit API rate limits.
- **Advanced Topic Modeling:** Employs a dynamic, corpus-aware LDA topic optimization algorithm to prevent over-partitioning.
- **Local AI Summarization:** Integrates WebLLM for client-side, context-aware, privacy-first topic summarization directly on your machine.
- **RAG-Powered Sentiment Reports:** A Retrieval-Augmented Generation (RAG) pipeline grounds AI-generated sentiment reports in actual Reddit source excerpts, distinctly separating original posts from community responses.
- **Automated Deployments:** GitHub Actions automatically builds and releases code-signed Windows and macOS executables.

## Setup for Development

1. **Install Node.js** (v18+) and **Python** (3.10+)
2. **Clone the repo**
3. **Install JS dependencies:**
   `npm install`
4. **Set up Python environment:**
   Create a virtual environment: `python -m venv venv`
   Activate the virtual environment:
     - Windows: `venv\Scripts\activate`
     - Mac/Linux: `source venv/bin/activate`
   Install Python dependencies: `pip install -r requirements.txt`
5. **Set up credentials:**
   Rename `.env.example` to `.env` and insert your Reddit API credentials.
6. **Run development version:**
   `npm run dev`

## Setup for End Users

Automated builds are available via GitHub Actions. Navigate to the Releases page of the repository to download and run the provided `.dmg` (Mac) or `.exe` installer (Windows). No need to manually install Python or Node.js.
