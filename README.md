# Mohs Reddit LDA

Full-stack Reddit NLP analysis app for Mohs surgery discussions. The workflow collects Reddit data from public `.json` endpoints, preprocesses text, trains a gensim LDA model, runs VADER sentiment analysis, generates publication-style Plotly figures, and exports the full analysis bundle.

## Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

If port `8000` is already occupied on your machine, use another port:

```bash
uvicorn main:app --reload --port 8010
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`.

When the backend is not on port `8000`, start the frontend with:

```bash
set NEXT_PUBLIC_API_BASE=http://127.0.0.1:8010&& npm run dev
```

## Notes

- Reddit collection uses only public `.json` endpoints and sends a User-Agent header.
- Enable sample data mode in the UI to test the full workflow without making Reddit requests.
- Exports are written under `backend/exports/{run_id}` and exposed by FastAPI at `/exports/...`.
- Topic summaries use `OPENAI_API_KEY` when available. Without it, the app falls back to deterministic local summaries and still exports `final_topics_comparison.xlsx`.
