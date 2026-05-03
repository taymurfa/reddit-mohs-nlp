from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
import uuid
import zipfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from plotly.utils import PlotlyJSONEncoder
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parent
load_dotenv(APP_ROOT / ".env")
EXPORT_ROOT = APP_ROOT / "exports"
EXPORT_ROOT.mkdir(exist_ok=True)

REDDIT_BASE = "https://www.reddit.com"
USER_AGENT = "mohs-reddit-lda/1.0 academic analysis app"
POST_LISTING_DELAY_SECONDS = 2.5
COMMENT_DELAY_SECONDS = 2.0
LARGE_RUN_COMMENT_DELAY_SECONDS = 3.5
MAX_COMMENT_FETCHES_PER_RUN = 250
MOHS_STOPWORDS = {
    "mohs",
    "surgery",
    "skin",
    "cancer",
    "dermatologist",
    "doctor",
    "surgeon",
    "procedure",
    "thing",
    "things",
    "people",
    "person",
    "someone",
    "anyone",
    "week",
    "weeks",
    "day",
    "days",
    "month",
    "months",
    "time",
    "really",
    "just",
    "like",
    "know",
    "think",
    "feel",
    "felt",
    "went",
    "going",
    "got",
    "get",
    "did",
    "does",
    "healing",
    "heal",
    "healed",
    "recovery",
    "recover",
    "post",
    "op",
    "pre",
    "nose",
    "face",
    "forehead",
    "cheek",
    "would",
    "could",
    "should",
    "also",
    "take",
    "took",
    "look",
    "looks",
    "looking",
    "line",
    "first",
    "lot",
    "lots",
    "minimal",
    "pretty",
    "right",
    "still",
    "back",
    "year",
    "old",
    "new",
    "way",
    "area",
    "good",
    "great",
    "best",
    "better",
    "want",
    "need",
    "needed",
    "work",
    "worked",
    "use",
    "used",
    "using",
    "help",
    "helped",
    "care",
    "much",
    "long",
    "full",
    "young",
    "never",
    "normal",
    "bad",
    "even",
    "three",
    "hour",
    "ask",
    "apply",
    "applied",
    "scar",
    "scars",
    "scarring",
    "little",
    "medical",
    "reason",
    "kind",
    "similar",
    "true",
    "bacterial",
    "keep",
    "kept",
    "well",
    "one",
    "two",
    "put",
    "advice",
    "treatment",
    "longer",
    "hope",
    "cause",
    "ended",
    "say",
    "wish",
    "see",
    "tiny",
    "positive",
    "mine",
    "ago",
    "notice",
    "experience",
    "sometimes",
    "give",
    "guess",
    "tip",
    "able",
    "sorry",
    "speedy",
    "photo",
    "glad",
    "wishing",
    "trying",
    "small",
    "removed",
    "every",
    "body",
    "lumpy",
    "try",
    "others",
    "taking",
    "hard",
    "else",
    "lol",
    "patience",
    "result",
    "spot",
    "definitely",
    "scary",
    "life",
    "profile",
    "black",
    "looked",
    "okay",
    "hang",
    "five",
    "friend",
    "late",
    "vain",
    "unnoticeable",
    "moh",
    "morning",
    "college",
    "photo",
    "picture",
    "quite",
    "may",
    "nothing",
    "maybe",
    "option",
    "attractive",
    "honestly",
    "said",
    "told",
    "make",
    "sure",
}
DEFAULT_LDA_STOPWORDS = [
    "mohs",
    "surgery",
    "skin",
    "cancer",
    "dermatologist",
    "doctor",
    "basal",
    "squamous",
    "carcinoma",
]
DEFAULT_CATEGORIES = [
    "Management/recovery",
    "Clinical presentation",
    "Procedure/reconstruction",
    "Emotion/anxiety",
    "Cost/access",
    "Information appraisal",
]
TREATMENT_TERMS = [
    "Aquaphor",
    "Vaseline",
    "silicone gel",
    "silicone sheets",
    "scar cream",
    "sunscreen",
    "antibiotics",
    "bandage",
    "stitches",
]
OFFICIAL_PRACTICE_AREAS = {
    "petrolatum_ointment": "Petrolatum or bland ointment use; compare with official wound-moisture and dressing instructions.",
    "dressings_bandages": "Dressing changes and wound coverage; compare with clinic guidance for keeping wounds covered and protected.",
    "moist_wound_care": "Moist wound healing; compare with official advice on avoiding dry scab formation when applicable.",
    "wound_coverage": "Wound coverage; compare with timing and type of dressing recommended by the surgeon.",
    "cleansing": "Cleansing and showering; compare with instructions for washing, patting dry, and avoiding soaking.",
    "silicone_scar_therapy": "Silicone gel/sheets; compare with official timing after epithelialization or stitch removal.",
    "sun_protection": "Sun protection; compare with official scar photoprotection and sunscreen recommendations.",
    "scar_massage": "Scar massage; compare with guidance on when massage is safe after wound closure.",
    "cosmetic_camouflage": "Makeup/camouflage; compare with official timing for applying cosmetics near the wound.",
    "pain_control": "Pain control; compare with recommended analgesics and cautions around NSAIDs/bleeding risk.",
    "numbing_lidocaine": "Peri-procedure numbing/anxiolytic discussion; compare with clinician-directed medication guidance.",
    "swelling_ice_elevation": "Swelling control; compare with official advice on ice, elevation, and activity.",
    "bleeding_pressure": "Bleeding management; compare with official pressure instructions and thresholds for calling clinic.",
    "infection_warning": "Infection warning signs; compare with official guidance on redness, drainage, warmth, and fever.",
    "antibiotic_ointment": "Antibiotic use; compare with official instructions and avoidance of unsupervised antibiotic ointment use.",
    "call_clinic": "Escalation to clinic; compare with official reasons to call after surgery.",
    "stitch_care": "Suture care/removal; compare with official stitch care and removal timing.",
    "reconstruction_repair": "Reconstruction/closure concerns; compare with surgeon-specific reconstruction aftercare.",
    "graft_flap_care": "Flap/graft care; compare with official graft/flap protection and follow-up instructions.",
    "activity_restriction": "Activity restriction; compare with official limits on exercise, bending, and heavy lifting.",
    "sleep_rest_positioning": "Rest and positioning; compare with official advice on sleeping position and elevation.",
}
MANAGEMENT_RECOVERY_TERMS = [
    "advice",
    "recommend",
    "recommended",
    "helped",
    "worked",
    "use",
    "using",
    "apply",
    "ointment",
    "vaseline",
    "aquaphor",
    "silicone",
    "gel",
    "sheet",
    "scar",
    "scarring",
    "sunscreen",
    "bandage",
    "dressing",
    "wound",
    "care",
    "stitches",
    "suture",
    "antibiotic",
    "infection",
    "bleeding",
    "swelling",
    "pain",
    "ice",
    "clean",
    "wash",
    "shower",
    "moist",
    "moisturize",
    "healing",
    "heal",
    "recovery",
    "recover",
    "aftercare",
    "post op",
    "post-op",
    "reconstruction",
    "flap",
    "graft",
]
AFTERCARE_RECOMMENDATION_TERMS = [
    "ointment",
    "vaseline",
    "aquaphor",
    "petrolatum",
    "silicone",
    "gel",
    "sheet",
    "scar cream",
    "sunscreen",
    "spf",
    "bandage",
    "bandaid",
    "dressing",
    "gauze",
    "tape",
    "nonstick",
    "wound",
    "stitches",
    "suture",
    "antibiotic",
    "infection",
    "infected",
    "redness",
    "drainage",
    "bleeding",
    "swelling",
    "pain",
    "tylenol",
    "ibuprofen",
    "nsaid",
    "lidocaine",
    "numbing",
    "ice",
    "elevated",
    "elevation",
    "pillow",
    "clean",
    "wash",
    "shower",
    "soap",
    "moist",
    "moisturize",
    "massage",
    "aftercare",
    "exercise",
    "lifting",
    "workout",
    "sleep",
    "flap",
    "graft",
]
ADVICE_CUE_TERMS = [
    "recommend",
    "recommended",
    "recommends",
    "suggest",
    "suggested",
    "should",
    "shouldn't",
    "shouldnt",
    "don't",
    "dont",
    "do not",
    "make sure",
    "helped",
    "worked",
    "use",
    "used",
    "using",
    "apply",
    "applied",
    "keep",
    "kept",
    "avoid",
    "change",
    "changed",
    "clean",
    "wash",
    "cover",
    "covered",
    "call",
    "ask",
    "take",
    "took",
    "put",
    "try",
    "tried",
    "best",
    "tip",
]
RECOVERY_TOPIC_LABELS = {
    "Ointments and dressings": {"petrolatum_ointment", "dressings_bandages", "moist_wound_care", "wound_coverage"},
    "Scar care and sun protection": {"silicone_scar_therapy", "sun_protection", "scar_massage", "cosmetic_camouflage"},
    "Cleaning and wound care": {"cleansing", "moist_wound_care", "dressings_bandages"},
    "Pain, swelling, and bleeding": {"pain_control", "swelling_ice_elevation", "bleeding_pressure", "numbing_lidocaine"},
    "Stitches and reconstruction": {"stitch_care", "reconstruction_repair", "graft_flap_care"},
    "Antibiotics and infection concerns": {"infection_warning", "antibiotic_ointment", "call_clinic"},
    "Activity and recovery restrictions": {"activity_restriction", "sleep_rest_positioning"},
}

RECOMMENDATION_VOCAB = {
    "petrolatum_ointment": ["vaseline", "aquaphor", "petrolatum", "ointment"],
    "dressings_bandages": ["bandage", "bandaid", "dressing", "gauze", "nonstick", "tape"],
    "moist_wound_care": ["moist", "moisturize", "dry out", "keep moist"],
    "wound_coverage": ["cover", "covered", "wrap"],
    "cleansing": ["clean", "wash", "shower", "soap", "water", "rinsed", "patted"],
    "silicone_scar_therapy": ["silicone", "silicon", "cica", "mederma", "scar sheet", "scar gel"],
    "sun_protection": ["sunscreen", "spf", "sun", "hat"],
    "scar_massage": ["massage", "massaging"],
    "cosmetic_camouflage": ["makeup", "concealer", "color corrector", "foundation"],
    "pain_control": ["pain", "tylenol", "acetaminophen", "ibuprofen", "nsaid", "codiene", "pain reliever", "pain killer"],
    "numbing_lidocaine": ["numbing", "lidocaine", "tranquilizer"],
    "swelling_ice_elevation": ["swelling", "bruise", "bruising", "ice", "elevated", "pillow"],
    "bleeding_pressure": ["bleeding", "bleed", "pressure"],
    "infection_warning": ["infection", "infected", "redness", "red", "warm", "drainage"],
    "antibiotic_ointment": ["antibiotic", "antibiotics", "polysporin"],
    "call_clinic": ["call", "office", "clinic"],
    "stitch_care": ["stitch", "stitches", "suture", "removal"],
    "reconstruction_repair": ["reconstruction", "plastic", "closure"],
    "graft_flap_care": ["graft", "flap"],
    "activity_restriction": ["exercise", "workout", "lifting", "bend"],
    "sleep_rest_positioning": ["sleep", "rest", "position", "side"],
}

def cors_origins() -> list[str]:
    configured = [
        origin.strip().rstrip("/")
        for origin in os.getenv("CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    return ["http://localhost:3000", "http://127.0.0.1:3000", *configured]


app = FastAPI(title="Mohs Reddit LDA API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/exports", StaticFiles(directory=str(EXPORT_ROOT)), name="exports")
ANALYSIS_STEPS = [
    "Collecting Reddit posts",
    "Collecting Reddit comments",
    "Cleaning text",
    "Running LDA",
    "Running sentiment",
    "Generating figures",
    "Exporting results",
]
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class RedditRateLimitError(RuntimeError):
    pass


class AnalysisRequest(BaseModel):
    subreddit: str = Field(min_length=1)
    # Backward-compatible field name: the UI now uses this list as extra LDA stopwords.
    keywords: list[str] = Field(default_factory=list)
    start_date: date
    end_date: date
    k: int = Field(default=10, ge=2, le=50)
    max_results: int = Field(default=1000, ge=10, le=5000)
    include_comments: bool = True
    sample_mode: bool = False


class AnalysisJobCreated(BaseModel):
    job_id: str


def reddit_get(path: str, params: dict[str, Any] | None = None, delay_seconds: float = POST_LISTING_DELAY_SECONDS) -> dict[str, Any] | list[Any]:
    url = f"{REDDIT_BASE}{path}"
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(6):
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            try:
                wait_seconds = int(retry_after) if retry_after else 8 * (attempt + 1)
            except ValueError:
                wait_seconds = 8 * (attempt + 1)
            time.sleep(min(wait_seconds, 45))
            continue
        if response.status_code >= 400:
            raise HTTPException(response.status_code, f"Reddit request failed: {response.text[:200]}")
        time.sleep(delay_seconds)
        return response.json()
    raise RedditRateLimitError("Reddit rate limit persisted after retries")


def normalize_subreddit_name(value: str) -> str:
    return value.strip().removeprefix("r/").removeprefix("/r/").strip("/")


def clean_reddit_text(value: Any) -> str:
    if not isinstance(value, str) or value in {"[deleted]", "[removed]"}:
        return ""
    return value.strip()


def normalize_post(post: dict[str, Any], query: str, subreddit: str) -> dict[str, Any] | None:
    data = post.get("data", {})
    body = clean_reddit_text(data.get("selftext", ""))
    title = clean_reddit_text(data.get("title", ""))
    if not title and not body:
        return None
    created = int(data.get("created_utc", 0))
    return {
        "id": data.get("id"),
        "type": "post",
        "subreddit": data.get("subreddit", subreddit),
        "author": data.get("author") or "",
        "created_utc": created,
        "date": datetime.fromtimestamp(created, timezone.utc).date().isoformat(),
        "title": title,
        "body": body,
        "score": data.get("score", 0),
        "num_comments": data.get("num_comments", 0),
        "permalink": f"https://reddit.com{data.get('permalink', '')}",
        "url": data.get("url", ""),
        "parent_id": "",
        "thread_id": data.get("id"),
        "query_used": query,
    }


def flatten_comments(node: dict[str, Any], thread_id: str, subreddit: str, query: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if node.get("kind") == "more":
        return rows
    data = node.get("data", {})
    if node.get("kind") == "t1":
        body = clean_reddit_text(data.get("body", ""))
        if body:
            created = int(data.get("created_utc", 0))
            rows.append({
                "id": data.get("id"),
                "type": "comment",
                "subreddit": data.get("subreddit", subreddit),
                "author": data.get("author") or "",
                "created_utc": created,
                "date": datetime.fromtimestamp(created, timezone.utc).date().isoformat(),
                "title": "",
                "body": body,
                "score": data.get("score", 0),
                "num_comments": 0,
                "permalink": f"https://reddit.com{data.get('permalink', '')}",
                "url": "",
                "parent_id": data.get("parent_id", ""),
                "thread_id": thread_id,
                "query_used": query,
            })
    replies = data.get("replies")
    children = []
    if isinstance(replies, dict):
        children = replies.get("data", {}).get("children", [])
    for child in children:
        rows.extend(flatten_comments(child, thread_id, subreddit, query))
    return rows


def collect_reddit_data(req: AnalysisRequest, progress: Any | None = None) -> pd.DataFrame:
    if req.sample_mode:
        rows = sample_dataset(req)
        if progress:
            progress({
                "phase": "complete",
                "posts_collected": int((rows["type"] == "post").sum()),
                "comments_collected": int((rows["type"] == "comment").sum()),
                "max_posts": req.max_results,
                "comment_fetches": 0,
                "comments_target": 0,
                "eta_seconds": 0,
                "elapsed_seconds": 0,
                "collection_status": "sample data loaded",
            })
        return rows
    subreddit = normalize_subreddit_name(req.subreddit)
    start_ts = int(datetime.combine(req.start_date, datetime.min.time(), timezone.utc).timestamp())
    end_ts = int(datetime.combine(req.end_date, datetime.max.time(), timezone.utc).timestamp())
    rows: list[dict[str, Any]] = []
    posts: list[dict[str, Any]] = []
    seen: set[str] = set()
    after = None
    query_used = "subreddit_new"
    comment_fetches = 0
    comments_target = 0
    current_phase = "posts"
    collection_started = time.time()

    def post_count() -> int:
        return len([r for r in rows if r["type"] == "post"])

    def comment_count() -> int:
        return len([r for r in rows if r["type"] == "comment"])

    def report(status: str = "collecting") -> None:
        if not progress:
            return
        elapsed = max(0.1, time.time() - collection_started)
        posts = post_count()
        if current_phase == "comments":
            rate = comment_fetches / elapsed if comment_fetches else 0
            remaining = max(0, comments_target - comment_fetches)
        else:
            rate = posts / elapsed if posts else 0
            remaining = max(0, req.max_results - posts)
        eta_seconds = int(remaining / rate) if rate > 0 else None
        progress({
            "phase": current_phase,
            "posts_collected": posts,
            "comments_collected": comment_count(),
            "max_posts": req.max_results,
            "comment_fetches": comment_fetches,
            "comments_target": comments_target,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": int(elapsed),
            "collection_status": status,
        })

    report("starting Reddit collection")
    while post_count() < req.max_results:
        params = {"limit": 100}
        if after:
            params["after"] = after
        try:
            payload = reddit_get(f"/r/{subreddit}/new.json", params, delay_seconds=POST_LISTING_DELAY_SECONDS)
        except RedditRateLimitError:
            if rows:
                break
            raise HTTPException(429, "Reddit rate limit persisted before any posts could be collected. Try again later, reduce Max posts, turn off comments, or use sample mode.")
        listing = payload.get("data", {}) if isinstance(payload, dict) else {}
        children = listing.get("children", [])
        if not children:
            break
        for post in children:
            normalized = normalize_post(post, query_used, subreddit)
            if not normalized:
                continue
            created = normalized["created_utc"]
            if created > end_ts:
                continue
            if created < start_ts:
                continue
            if normalized["id"] in seen:
                continue
            seen.add(normalized["id"])
            posts.append(normalized)
            rows.append(normalized)
            report("collecting posts")
            if post_count() >= req.max_results:
                break
        after = listing.get("after")
        if not after:
            break
    if req.include_comments and posts:
        current_phase = "comments"
        comments_target = min(len(posts), MAX_COMMENT_FETCHES_PER_RUN)
        collection_started = time.time()
        report("starting comment collection")
        for normalized in posts:
            if comment_fetches >= MAX_COMMENT_FETCHES_PER_RUN:
                report("comment request budget reached; continuing with collected comments")
                break
            try:
                delay = LARGE_RUN_COMMENT_DELAY_SECONDS if req.max_results > 100 else COMMENT_DELAY_SECONDS
                comments_payload = reddit_get(f"/r/{subreddit}/comments/{normalized['id']}.json", delay_seconds=delay)
                comment_fetches += 1
            except RedditRateLimitError:
                report("rate limited while collecting comments; continuing with partial comments")
                return pd.DataFrame(rows)
            if isinstance(comments_payload, list) and len(comments_payload) > 1:
                for child in comments_payload[1].get("data", {}).get("children", []):
                    for comment in flatten_comments(child, normalized["id"], subreddit, query_used):
                        if comment["id"] not in seen:
                            seen.add(comment["id"])
                            if start_ts <= comment["created_utc"] <= end_ts:
                                rows.append(comment)
            report("collecting comments")
    current_phase = "complete"
    report("Reddit collection complete")
    return pd.DataFrame(rows)


def sample_dataset(req: AnalysisRequest) -> pd.DataFrame:
    subreddit = normalize_subreddit_name(req.subreddit)
    examples = [
        ("post", "Mohs surgery next week", "I am anxious about basal cell carcinoma on my nose and reconstruction scarring.", 14),
        ("comment", "", "I recommend keeping the wound moist with Vaseline and changing the bandage daily for the first week.", 6),
        ("comment", "", "The pressure bandage helped with bleeding; I used clean gauze and called the clinic when spotting continued.", 9),
        ("post", "Cost of Mohs", "Insurance covered most of my squamous cell carcinoma procedure but access was confusing.", 3),
        ("comment", "", "Silicone gel made the scar flatter after several weeks, and I used sunscreen every morning.", 11),
        ("post", "How reliable is Reddit advice", "I found mixed information about antibiotics after Mohs surgery and called my clinic.", 5),
        ("comment", "", "Aquaphor irritated my skin, but plain Vaseline worked well under the dressing.", 2),
        ("comment", "", "For a flap reconstruction, sleeping elevated and icing around the area helped swelling.", 7),
        ("comment", "", "My best advice is avoid exercise, bending, and heavy lifting until the surgeon clears you.", 8),
        ("comment", "", "I washed gently in the shower, patted dry, then applied ointment before covering it again.", 10),
        ("comment", "", "Tylenol controlled my pain better than expected, but I avoided ibuprofen because they warned me about bleeding.", 4),
        ("comment", "", "When the wound became red and warm, I called the office and they prescribed antibiotics.", 5),
        ("comment", "", "Silicone sheets helped the raised scar after the stitches came out; sunscreen kept it from darkening.", 12),
        ("comment", "", "Paper tape and a nonstick bandage kept the graft from rubbing on my mask.", 6),
        ("comment", "", "Do not let it dry out early; moist wound care with petrolatum was the biggest tip from my nurse.", 13),
        ("comment", "", "I changed the dressing after every shower and watched for drainage or spreading redness.", 5),
        ("comment", "", "For nose reconstruction, I used ice packs on and off and slept on extra pillows.", 8),
        ("comment", "", "Scar massage only started after my doctor said the incision was fully closed.", 7),
        ("comment", "", "Ask for written aftercare instructions because Reddit advice varies a lot.", 4),
        ("comment", "", "A wide brim hat plus mineral sunscreen helped protect the new scar outside.", 9),
        ("comment", "", "Stitches pulled a little, so I used the ointment and bandage routine until removal day.", 6),
    ]
    rows = []
    base = datetime.combine(req.start_date, datetime.min.time(), timezone.utc)
    for i in range(max(32, min(req.max_results, 160))):
        typ, title, body, score = examples[i % len(examples)]
        created = int((base + pd.DateOffset(days=i % max(1, (req.end_date - req.start_date).days + 1))).timestamp())
        rows.append({
            "id": f"sample_{i}",
            "type": typ,
            "subreddit": subreddit,
            "author": f"user_{i % 13}",
            "created_utc": created,
            "date": datetime.fromtimestamp(created, timezone.utc).date().isoformat(),
            "title": title,
            "body": f"{body} https://example{i % 5}.org/resource",
            "score": score + i % 4,
            "num_comments": 4 if typ == "post" else 0,
            "permalink": f"https://reddit.com/r/{subreddit}/comments/sample_{i}",
            "url": "",
            "parent_id": "" if typ == "post" else f"sample_{max(0, i-1)}",
            "thread_id": f"sample_thread_{i // 4}",
            "query_used": "sample_subreddit_new",
        })
    return pd.DataFrame(rows)


def normalize_stopword_terms(terms: list[str]) -> set[str]:
    normalized: set[str] = set()
    for term in terms:
        for token in re.sub(r"[^a-zA-Z\s]", " ", term.lower()).split():
            if len(token) > 1:
                normalized.add(token)
    return normalized


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def terms_pattern(terms: list[str]) -> re.Pattern[str]:
    parts = []
    for term in sorted(terms, key=len, reverse=True):
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        parts.append(rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])")
    return re.compile("|".join(parts), re.IGNORECASE)


def extract_recovery_advice_text(text: str) -> tuple[str, int]:
    if not text:
        return "", 0
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    recovery_pattern = terms_pattern(AFTERCARE_RECOMMENDATION_TERMS)
    advice_pattern = terms_pattern(ADVICE_CUE_TERMS)
    selected = []
    for sentence in split_sentences(text):
        has_recovery = bool(recovery_pattern.search(sentence))
        has_advice = bool(advice_pattern.search(sentence))
        if has_recovery and has_advice:
            selected.append(sentence)
    if not selected and recovery_pattern.search(text) and advice_pattern.search(text):
        selected = [text]
    passage = " ".join(selected)
    return passage, len(selected)


def add_recovery_phrases(token_lists: list[list[str]]) -> list[list[str]]:
    if len(token_lists) < 3:
        return token_lists
    bigram = Phraser(Phrases(token_lists, min_count=2, threshold=4))
    trigram = Phraser(Phrases(bigram[token_lists], min_count=2, threshold=4))
    return [list(trigram[bigram[tokens]]) for tokens in token_lists]


def recommendation_vocab_tokens(text: str) -> list[str]:
    tokens = []
    lowered = str(text).lower()
    for canonical, terms in RECOMMENDATION_VOCAB.items():
        pattern = terms_pattern(terms)
        if pattern.search(lowered):
            tokens.append(canonical)
    return tokens


def apply_recommendation_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vocab_tokens = out["analysis_text"].where(out["analysis_text"].str.len() > 0, out["combined_text"]).apply(recommendation_vocab_tokens)
    out["free_text_tokens"] = out["tokens"]
    out["tokens"] = vocab_tokens
    out["clean_text"] = out["tokens"].apply(lambda tokens: " ".join(tokens))
    return out[out["tokens"].apply(len) >= 1].reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame, extra_stopwords: set[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        stops = set(stopwords.words("english")) | MOHS_STOPWORDS
    except LookupError:
        stops = set(ENGLISH_STOP_WORDS) | MOHS_STOPWORDS
    stops |= extra_stopwords or set()
    lemmatizer = WordNetLemmatizer()
    out = df.copy()
    out["combined_text"] = (out["title"].fillna("") + " " + out["body"].fillna("")).str.strip()
    out["raw_urls"] = out["combined_text"].apply(lambda x: re.findall(r"https?://\S+", x))
    extracted = out["combined_text"].apply(extract_recovery_advice_text)
    out["analysis_text"] = extracted.apply(lambda x: x[0])
    out["advice_sentence_count"] = extracted.apply(lambda x: x[1])
    out["model_text"] = out["analysis_text"].where(out["analysis_text"].str.len() > 0, out["combined_text"])

    def clean(text: str) -> tuple[str, list[str]]:
        text = text.lower()
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = []
        for token in text.split():
            if len(token) <= 2 or token in stops:
                continue
            try:
                token = lemmatizer.lemmatize(token, wordnet.NOUN)
            except LookupError:
                pass
            tokens.append(token)
        return " ".join(tokens), tokens

    cleaned = out["model_text"].apply(clean)
    out["clean_text"] = cleaned.apply(lambda x: x[0])
    out["tokens"] = cleaned.apply(lambda x: x[1])
    out["tokens"] = add_recovery_phrases(out["tokens"].tolist())
    out["clean_text"] = out["tokens"].apply(lambda tokens: " ".join(tokens))
    out = out[out["tokens"].apply(len) >= 3].drop_duplicates(subset=["clean_text"])
    return out.reset_index(drop=True)


def compute_corpus_stats(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    if raw.empty:
        return {"total_docs": 0, "posts": 0, "comments": 0, "unique_users": 0, "date_range": "", "avg_length": 0}
    lengths = clean["combined_text"].str.split().apply(len) if not clean.empty else pd.Series(dtype=float)
    return {
        "total_docs": int(len(clean)),
        "raw_docs": int(len(raw)),
        "posts": int((raw["type"] == "post").sum()),
        "comments": int((raw["type"] == "comment").sum()),
        "unique_users": int(raw["author"].nunique()),
        "date_range": f"{raw['date'].min()} to {raw['date'].max()}",
        "avg_length": round(float(lengths.mean() if len(lengths) else 0), 1),
    }


def focus_management_recovery(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pattern = terms_pattern(AFTERCARE_RECOMMENDATION_TERMS)
    advice_focused = df[df["advice_sentence_count"] > 0].copy()
    comment_advice = advice_focused[advice_focused["type"] == "comment"].copy()
    if len(comment_advice) >= 10:
        focused = comment_advice
    elif len(advice_focused) >= 5:
        focused = advice_focused
    else:
        mask = (
            df["combined_text"].apply(lambda text: bool(pattern.search(str(text))))
            | df["clean_text"].apply(lambda text: bool(pattern.search(str(text))))
        )
        focused = df[mask].copy()
    focused["analysis_focus"] = "management_recovery_advice"
    return focused.reset_index(drop=True)


def train_lda(df: pd.DataFrame, k: int) -> tuple[LdaModel, corpora.Dictionary, list[list[tuple[int, int]]], pd.DataFrame]:
    dictionary = corpora.Dictionary(df["tokens"])
    no_below = 2 if len(df) >= 100 else 1
    dictionary.filter_extremes(no_below=no_below, no_above=0.85, keep_n=3000)
    corpus_all = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]
    keep_indexes = [index for index, bow in enumerate(corpus_all) if bow]
    df = df.iloc[keep_indexes].reset_index(drop=True)
    corpus = [corpus_all[index] for index in keep_indexes]
    if not dictionary or not any(corpus):
        raise HTTPException(422, "Not enough usable text after preprocessing to train LDA.")
    effective_k = min(k, len(df), max(2, round(len(df) ** 0.5) + 1))
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=effective_k,
        passes=35,
        iterations=300,
        alpha=0.1,
        eta=0.01,
        random_state=42,
        minimum_probability=0,
    )
    return model, dictionary, corpus, df


def assign_topics(df: pd.DataFrame, model: LdaModel, corpus: list[list[tuple[int, int]]]) -> pd.DataFrame:
    out = df.copy()
    assignments = []
    strengths = []
    topic_probabilities = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0)
        topic, score = max(dist, key=lambda x: x[1])
        assignments.append(int(topic))
        strengths.append(float(score))
        topic_probabilities.append({int(topic_id): float(probability) for topic_id, probability in dist})
    out["topic"] = assignments
    out["topic_score"] = strengths
    out["topic_probabilities"] = topic_probabilities
    return out


def infer_recovery_label(keywords: list[str]) -> str:
    scores: dict[str, int] = {}
    for label, terms in RECOVERY_TOPIC_LABELS.items():
        score = 0
        for rank, keyword in enumerate(keywords[:8]):
            keyword_parts = set(keyword.split("_"))
            if keyword in terms:
                score += 12 if rank == 0 else 6 if rank == 1 else 3
            score += len(keyword_parts & terms)
        scores[label] = score
    label, score = max(scores.items(), key=lambda item: item[1])
    if score > 0:
        return label
    return "Management/recovery advice"


def topic_distinctiveness(topic_keywords: list[str], all_topic_keywords: list[list[str]]) -> float:
    current = set(topic_keywords[:10])
    if not current:
        return 0
    overlaps = []
    for other in all_topic_keywords:
        other_set = set(other[:10])
        if other_set == current:
            continue
        overlaps.append(len(current & other_set) / len(current | other_set) if current | other_set else 0)
    if not overlaps:
        return 1
    return round(1 - max(overlaps), 3)


def display_example_text(row: pd.Series) -> str:
    text = row.get("analysis_text") or row.get("combined_text") or ""
    text = re.sub(r"https?://\S+", "", str(text)).strip()
    return text[:420]


def build_topics(df: pd.DataFrame, model: LdaModel) -> list[dict[str, Any]]:
    topics = []
    total = max(1, len(df))
    all_keywords = [[word for word, _ in model.show_topic(topic_id, topn=5)] for topic_id in range(model.num_topics)]
    for topic_id in range(model.num_topics):
        keywords = all_keywords[topic_id]
        subset = df[df["topic"] == topic_id]
        rep = ""
        examples: list[dict[str, Any]] = []
        ranked_source = subset
        if ranked_source.empty:
            ranked_source = df.copy()
            ranked_source["_topic_probability"] = ranked_source["topic_probabilities"].apply(lambda probs: probs.get(topic_id, 0))
            ranked = ranked_source.sort_values("_topic_probability", ascending=False).head(3)
        else:
            ranked = ranked_source.sort_values("topic_score", ascending=False).head(3)
        if not ranked.empty:
            rep = ranked.iloc[0]["combined_text"][:320]
            examples = [
                {
                    "id": row["id"],
                    "type": row["type"],
                    "date": row["date"],
                    "score": int(row.get("score", 0)),
                    "permalink": row.get("permalink", ""),
                    "text": display_example_text(row),
                }
                for _, row in ranked.iterrows()
            ]
        topics.append({
            "topic": topic_id,
            "keywords": keywords,
            "doc_count": int(len(subset)),
            "percentage": round(len(subset) / total * 100, 2),
            "distinctiveness": topic_distinctiveness(keywords, all_keywords),
            "representative_document": rep,
            "example_documents": examples,
            "label": infer_recovery_label(keywords),
            "category": "Management/recovery",
        })
    non_empty_topics = [topic for topic in topics if topic["doc_count"] > 0]
    sorted_topics = sorted(non_empty_topics or topics, key=lambda item: item["doc_count"], reverse=True)
    label_counts = Counter(topic["label"] for topic in sorted_topics)
    for topic in sorted_topics:
        if label_counts[topic["label"]] > 1:
            terms = [term.replace("_", " ") for term in topic["keywords"][:2]]
            topic["label"] = f"{topic['label']} ({', '.join(terms)})"
    return sorted_topics


def topic_official_practice_notes(keywords: list[str]) -> str:
    notes = [OFFICIAL_PRACTICE_AREAS[keyword] for keyword in keywords if keyword in OFFICIAL_PRACTICE_AREAS]
    return " ".join(notes[:3]) if notes else "Compare with official Mohs post-operative wound care instructions."


def fallback_topic_summary(topic: dict[str, Any], reason: str = "OpenAI API key not configured") -> dict[str, str]:
    readable_keywords = [keyword.replace("_", " ") for keyword in topic["keywords"]]
    examples = topic.get("example_documents", [])
    example_text = examples[0]["text"] if examples else topic.get("representative_document", "")
    summary = (
        f"This topic captures patient-to-patient recommendations around {topic['label'].lower()}, "
        f"especially {', '.join(readable_keywords[:4])}."
    )
    explanation = (
        f"Representative comments suggest practical recovery advice rather than clinical diagnosis. "
        f"Example evidence: {example_text[:220]}"
    )
    return {
        "llm_topic_title": topic["label"],
        "llm_summary": summary,
        "llm_explanation": explanation,
        "evidence_source_ids": json.dumps([f"T{topic['topic'] + 1}-E1"] if examples else [], ensure_ascii=False),
        "notable_recommendations": json.dumps(readable_keywords[:4], ensure_ascii=False),
        "cautions_or_uncertainties": "Local fallback summary; review retrieved examples directly for nuance.",
        "official_practice_area": topic_official_practice_notes(topic["keywords"]),
        "comparison_guidance": "Use this row to compare Reddit advice against official post-op instructions from dermatology or Mohs surgery practices.",
        "llm_summary_source": "local_fallback",
        "llm_error": reason,
    }


def parse_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def topic_sentiment_distribution(df: pd.DataFrame, topic_id: int) -> dict[str, int]:
    subset = df[df["topic"] == topic_id]
    return {str(key): int(value) for key, value in subset["sentiment"].value_counts().to_dict().items()}


def topic_treatment_mentions(df: pd.DataFrame, topic_id: int) -> list[dict[str, Any]]:
    subset = df[df["topic"] == topic_id]
    rows = []
    for term in TREATMENT_TERMS:
        count = int(subset["combined_text"].str.contains(re.escape(term), case=False, na=False).sum())
        if count:
            rows.append({"term": term, "count": count})
    return rows


def topic_source_packet(topic: dict[str, Any], sentiment_df: pd.DataFrame | None = None) -> dict[str, Any]:
    topic_id = topic["topic"]
    examples = []
    for index, example in enumerate(topic.get("example_documents", [])[:5], start=1):
        examples.append({
            "source_id": f"T{topic_id + 1}-E{index}",
            "document_id": example.get("id", ""),
            "type": example.get("type", ""),
            "date": example.get("date", ""),
            "score": example.get("score", ""),
            "permalink": example.get("permalink", ""),
            "text": example.get("text", ""),
        })
    return {
        "topic": topic_id,
        "label": topic["label"],
        "keywords": topic["keywords"],
        "doc_count": topic["doc_count"],
        "percentage": topic["percentage"],
        "distinctiveness": topic.get("distinctiveness"),
        "sentiment_distribution": topic_sentiment_distribution(sentiment_df, topic_id) if sentiment_df is not None else {},
        "treatment_mentions": topic_treatment_mentions(sentiment_df, topic_id) if sentiment_df is not None else [],
        "official_practice_notes": topic_official_practice_notes(topic["keywords"]),
        "retrieved_sources": examples,
    }


def llm_interpret_topics(topics: list[dict[str, Any]], sentiment_df: pd.DataFrame | None = None) -> list[dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    fallback = [fallback_topic_summary(topic) for topic in topics]
    if not api_key:
        return fallback

    topic_payload = [
        topic_source_packet(topic, sentiment_df)
        for topic in topics
    ]
    prompt = {
        "task": "Use the retrieved Reddit source snippets to interpret topic-model outputs about lay post-Mohs surgery recovery recommendations.",
        "instructions": [
            "Return JSON only, as an array with one object per input topic in the same order.",
            "Ground every interpretation in the retrieved_sources text and topic keywords. Do not infer advice that is not supported by the snippets.",
            "Use cautious academic language. Do not claim the Reddit advice is medically correct.",
            "Summaries should explain what laypeople are recommending, what problem the advice addresses, and any variation or uncertainty visible in the sources.",
            "Mention source IDs such as T1-E1 when describing evidence.",
            "Add comparison guidance for reviewing the lay advice against official post-operative Mohs recovery instructions.",
            "If sources are thin or mixed, say so explicitly.",
        ],
        "required_fields": [
            "llm_topic_title",
            "llm_summary",
            "llm_explanation",
            "evidence_source_ids",
            "notable_recommendations",
            "cautions_or_uncertainties",
            "official_practice_area",
            "comparison_guidance",
        ],
        "topics": topic_payload,
    }
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": [
                    {
                        "role": "system",
                        "content": "You are assisting with qualitative interpretation of topic-model outputs for academic research.",
                    },
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        text = parse_openai_text(response.json()).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        interpreted = json.loads(text)
        if not isinstance(interpreted, list) or len(interpreted) != len(topics):
            return [fallback_topic_summary(topic, "OpenAI response did not contain one interpretation per topic") for topic in topics]
        merged = []
        for item in interpreted:
            merged.append({
                "llm_topic_title": str(item.get("llm_topic_title", "")),
                "llm_summary": str(item.get("llm_summary", "")),
                "llm_explanation": str(item.get("llm_explanation", "")),
                "evidence_source_ids": json.dumps(item.get("evidence_source_ids", []), ensure_ascii=False),
                "notable_recommendations": json.dumps(item.get("notable_recommendations", []), ensure_ascii=False),
                "cautions_or_uncertainties": str(item.get("cautions_or_uncertainties", "")),
                "official_practice_area": str(item.get("official_practice_area", "")),
                "comparison_guidance": str(item.get("comparison_guidance", "")),
                "llm_summary_source": f"openai:{model}",
                "llm_error": "",
            })
        return merged
    except requests.HTTPError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        return [fallback_topic_summary(topic, f"OpenAI request failed: {detail}") for topic in topics]
    except Exception as exc:
        return [fallback_topic_summary(topic, f"OpenAI interpretation failed: {exc}") for topic in topics]


def attach_topic_interpretations(topics: list[dict[str, Any]], sentiment_df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    interpretations = llm_interpret_topics(topics, sentiment_df)
    enriched = []
    for topic, interpretation in zip(topics, interpretations):
        enriched.append({**topic, **interpretation})
    return enriched


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    out = df.copy()
    scores = out["combined_text"].apply(analyzer.polarity_scores)
    out["sentiment_compound"] = scores.apply(lambda x: x["compound"])
    out["sentiment"] = out["sentiment_compound"].apply(lambda v: "positive" if v >= 0.05 else "negative" if v <= -0.05 else "neutral")
    return out


def treatment_sentiment(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for term in TREATMENT_TERMS:
        mask = df["combined_text"].str.contains(re.escape(term), case=False, na=False)
        subset = df[mask]
        count = len(subset)
        rows.append({
            "term": term,
            "mention_count": int(count),
            "positive_pct": round(float((subset["sentiment"] == "positive").mean() * 100), 2) if count else 0,
            "negative_pct": round(float((subset["sentiment"] == "negative").mean() * 100), 2) if count else 0,
        })
    return rows


def shared_domains(df: pd.DataFrame) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for urls in df.get("raw_urls", []):
        for url in urls:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if domain:
                counter[domain] += 1
    return [{"domain": d, "count": c} for d, c in counter.most_common(25)]


def make_figures(df: pd.DataFrame, topics: list[dict[str, Any]], treatment: list[dict[str, Any]], domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figures = []
    chart_theme = dict(template="plotly_dark", paper_bgcolor="#10131a", plot_bgcolor="#10131a", font_color="#e5e7eb")

    def add_figure(fig_id: str, title: str, fig: go.Figure) -> None:
        figures.append({"id": fig_id, "title": title, "spec": json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))})

    monthly = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("M").astype(str)).groupby(["month", "type"]).size().reset_index(name="count")
    fig = go.Figure()
    for typ in ["post", "comment"]:
        subset = monthly[monthly["type"] == typ]
        fig.add_bar(x=subset["month"], y=subset["count"], name=typ.title())
    fig.update_layout(title="Posts and Comments Over Time", barmode="stack", **chart_theme)
    add_figure("over_time", "Posts/comments over time", fig)

    fig = go.Figure(go.Bar(x=[f"T{t['topic'] + 1}" for t in topics], y=[t["percentage"] for t in topics]))
    fig.update_layout(title="Topic Distribution", yaxis_title="% of corpus", **chart_theme)
    add_figure("topic_distribution", "Topic distribution", fig)

    trends = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("M").astype(str)).groupby(["month", "topic"]).size().reset_index(name="count")
    fig = go.Figure()
    for topic in sorted(df["topic"].unique()):
        subset = trends[trends["topic"] == topic]
        fig.add_scatter(x=subset["month"], y=subset["count"], mode="lines+markers", name=f"T{topic + 1}")
    fig.update_layout(title="Topic Trends Over Time", **chart_theme)
    add_figure("topic_trends", "Topic trends over time", fig)

    sentiment_counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    fig = go.Figure(go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.45))
    fig.update_layout(title="Sentiment Distribution", **chart_theme)
    add_figure("sentiment_distribution", "Sentiment distribution", fig)

    by_topic = df.groupby(["topic", "sentiment"]).size().reset_index(name="count")
    fig = go.Figure()
    for sentiment in ["positive", "neutral", "negative"]:
        subset = by_topic[by_topic["sentiment"] == sentiment]
        fig.add_bar(x=[f"T{int(t) + 1}" for t in subset["topic"]], y=subset["count"], name=sentiment.title())
    fig.update_layout(title="Sentiment by Topic", barmode="stack", **chart_theme)
    add_figure("sentiment_by_topic", "Sentiment by topic", fig)

    fig = go.Figure()
    fig.add_bar(x=[t["term"] for t in treatment], y=[t["positive_pct"] for t in treatment], name="Positive %")
    fig.add_bar(x=[t["term"] for t in treatment], y=[t["negative_pct"] for t in treatment], name="Negative %")
    fig.update_layout(title="Treatment/Product Sentiment", barmode="group", **chart_theme)
    add_figure("treatment_sentiment", "Treatment/product sentiment", fig)

    fig = go.Figure(go.Bar(x=[d["count"] for d in domains], y=[d["domain"] for d in domains], orientation="h"))
    fig.update_layout(title="Shared Domains Frequency", **chart_theme)
    add_figure("shared_domains", "Shared domains frequency", fig)
    return figures


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    normalized = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if isinstance(value, (list, dict)):
                item[key] = json.dumps(value, ensure_ascii=False)
        normalized.append(item)
    pd.DataFrame(normalized).to_csv(path, index=False)


def topic_rows_for_excel(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for topic in topics:
        rows.append({
            "topic_number": topic["topic"] + 1,
            "topic_label": topic["label"],
            "llm_topic_title": topic.get("llm_topic_title", ""),
            "keywords": ", ".join(keyword.replace("_", " ") for keyword in topic["keywords"]),
            "doc_count": topic["doc_count"],
            "percentage": topic["percentage"],
            "distinctiveness": topic.get("distinctiveness", ""),
            "llm_summary": topic.get("llm_summary", ""),
            "llm_explanation": topic.get("llm_explanation", ""),
            "evidence_source_ids": topic.get("evidence_source_ids", ""),
            "notable_recommendations": topic.get("notable_recommendations", ""),
            "cautions_or_uncertainties": topic.get("cautions_or_uncertainties", ""),
            "official_practice_area": topic.get("official_practice_area", topic_official_practice_notes(topic["keywords"])),
            "comparison_guidance": topic.get("comparison_guidance", ""),
            "official_recommendation_notes": "",
            "alignment_assessment": "",
            "reviewer_notes": "",
            "llm_summary_source": topic.get("llm_summary_source", ""),
            "llm_error": topic.get("llm_error", ""),
        })
    return rows


def topic_example_rows(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for topic in topics:
        for example in topic.get("example_documents", []):
            rows.append({
                "topic_number": topic["topic"] + 1,
                "topic_label": topic["label"],
                "example_id": example.get("id", ""),
                "type": example.get("type", ""),
                "date": example.get("date", ""),
                "score": example.get("score", ""),
                "permalink": example.get("permalink", ""),
                "example_text": example.get("text", ""),
            })
    return rows


def write_topic_comparison_excel(path: Path, topics: list[dict[str, Any]]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(topic_rows_for_excel(topics)).to_excel(writer, sheet_name="Final Topics", index=False)
        pd.DataFrame(topic_example_rows(topics)).to_excel(writer, sheet_name="Example Documents", index=False)
        pd.DataFrame([
            {"recommendation_concept": key, "official_practice_comparison_note": value}
            for key, value in OFFICIAL_PRACTICE_AREAS.items()
        ]).to_excel(writer, sheet_name="Practice Comparison Guide", index=False)
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                column_letter = column_cells[0].column_letter
                max_length = min(70, max(len(str(cell.value or "")) for cell in column_cells) + 2)
                worksheet.column_dimensions[column_letter].width = max(12, max_length)


def export_results(raw: pd.DataFrame, clean: pd.DataFrame, topics: list[dict[str, Any]], sentiment: pd.DataFrame, category_percentages: list[dict[str, Any]], treatment: list[dict[str, Any]], domains: list[dict[str, Any]], figures: list[dict[str, Any]], stats: dict[str, Any]) -> list[dict[str, str]]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EXPORT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "raw_reddit_dataset.csv", raw)
    cleaned_export = clean.drop(columns=["tokens"], errors="ignore")
    write_csv(out_dir / "cleaned_dataset.csv", cleaned_export)
    write_csv(out_dir / "topic_assignments.csv", sentiment.drop(columns=["tokens"], errors="ignore"))
    write_csv(out_dir / "topic_summary.csv", topics)
    write_topic_comparison_excel(out_dir / "final_topics_comparison.xlsx", topics)
    write_csv(out_dir / "category_percentages.csv", category_percentages)
    write_csv(out_dir / "sentiment_results.csv", sentiment[["id", "type", "date", "topic", "sentiment", "sentiment_compound"]])
    write_csv(out_dir / "treatment_sentiment.csv", treatment)
    write_csv(out_dir / "shared_domains.csv", domains)
    for fig in figures:
        Path(out_dir / f"{fig['id']}.json").write_text(json.dumps(fig["spec"]), encoding="utf-8")
        html = go.Figure(fig["spec"]).to_html(include_plotlyjs="cdn", full_html=True)
        Path(out_dir / f"{fig['id']}.html").write_text(html, encoding="utf-8")
    report = ["# Reddit Mohs NLP Analysis", "", "## Corpus Statistics", ""]
    report += [f"- **{k}**: {v}" for k, v in stats.items()]
    report += ["", "## Topics", ""]
    report += [
        f"- Topic {t['topic'] + 1}: {t.get('llm_topic_title') or t['label']} ({t['percentage']}%). "
        f"{t.get('llm_summary', '')}"
        for t in topics
    ]
    (out_dir / "research_summary.md").write_text("\n".join(report), encoding="utf-8")
    zip_path = out_dir / "all_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in out_dir.iterdir():
            if file != zip_path:
                zf.write(file, file.name)
    return [{"name": p.name, "url": f"/exports/{run_id}/{p.name}"} for p in out_dir.iterdir()]


def update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def run_analysis_pipeline(req: AnalysisRequest, progress: Any | None = None) -> dict[str, Any]:
    def step(index: int) -> None:
        if progress:
            progress(index, ANALYSIS_STEPS[index])

    if req.end_date < req.start_date:
        raise HTTPException(422, "end_date must be on or after start_date")
    step(0)
    def collection_progress(details: dict[str, Any]) -> None:
        if progress:
            phase = details.get("phase")
            if phase == "comments":
                progress(1, ANALYSIS_STEPS[1], details)
            else:
                progress(0, ANALYSIS_STEPS[0], details)

    raw = collect_reddit_data(req, collection_progress)
    if raw.empty:
        raise HTTPException(404, "No Reddit documents found for the requested subreddit and date range.")
    step(2)
    clean_all = preprocess_dataframe(raw, normalize_stopword_terms(req.keywords or DEFAULT_LDA_STOPWORDS))
    if clean_all.empty:
        raise HTTPException(422, "Documents were found, but none had enough analyzable text after preprocessing.")
    clean = focus_management_recovery(clean_all)
    if clean.empty:
        raise HTTPException(422, "Documents were found, but none matched the management/recovery advice focus after preprocessing.")
    clean = apply_recommendation_vocabulary(clean)
    if clean.empty:
        raise HTTPException(422, "Management/recovery advice documents were found, but not enough concrete aftercare recommendations were detected for topic modeling.")
    step(3)
    model, _, corpus, lda_input = train_lda(clean, req.k)
    assigned = assign_topics(lda_input, model, corpus)
    topics = build_topics(assigned, model)
    step(4)
    sentiment_df = run_sentiment(assigned)
    sentiment_summary = {
        "overall_distribution": sentiment_df["sentiment"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "by_topic": sentiment_df.groupby(["topic", "sentiment"]).size().reset_index(name="count").to_dict("records"),
    }
    category_percentages = (
        pd.DataFrame(topics).groupby("category")["doc_count"].sum().reset_index().assign(
            percentage=lambda d: (d["doc_count"] / max(1, d["doc_count"].sum()) * 100).round(2)
        ).to_dict("records")
    )
    treatment = treatment_sentiment(sentiment_df)
    domains = shared_domains(sentiment_df)
    topics = attach_topic_interpretations(topics, sentiment_df)
    step(5)
    figures = make_figures(sentiment_df, topics, treatment, domains)
    stats = compute_corpus_stats(raw, clean_all)
    stats["analysis_focus"] = "Management/recovery advice"
    stats["lda_docs"] = int(len(clean))
    stats["lda_doc_percentage"] = round(len(clean) / max(1, len(clean_all)) * 100, 2)
    step(6)
    links = export_results(raw, clean_all, topics, sentiment_df, category_percentages, treatment, domains, figures, stats)
    return {
        "corpus_stats": stats,
        "topics": topics,
        "topic_percentages": [{"topic": t["topic"], "percentage": t["percentage"]} for t in topics],
        "category_percentages": category_percentages,
        "sentiment_summary": sentiment_summary,
        "treatment_sentiment": treatment,
        "shared_domains": domains,
        "figures": figures,
        "export_links": links,
    }


def run_job(job_id: str, req: AnalysisRequest) -> None:
    def progress(step_index: int, step_name: str, details: dict[str, Any] | None = None) -> None:
        updates = {"status": "running", "step_index": step_index, "step": step_name}
        if details is not None:
            updates["collection_progress"] = details
        update_job(job_id, **updates)

    try:
        result = run_analysis_pipeline(req, progress)
        update_job(job_id, status="completed", step_index=len(ANALYSIS_STEPS), step="Complete", result=result)
    except HTTPException as exc:
        update_job(job_id, status="failed", error=str(exc.detail), step="Failed")
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), traceback=traceback.format_exc(), step="Failed")


@app.post("/analysis-jobs", response_model=AnalysisJobCreated)
def create_analysis_job(req: AnalysisRequest) -> AnalysisJobCreated:
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "step_index": -1,
            "step": "Queued",
            "steps": ANALYSIS_STEPS,
            "collection_progress": {
                "phase": "queued",
                "posts_collected": 0,
                "comments_collected": 0,
                "max_posts": req.max_results,
                "comment_fetches": 0,
                "comments_target": 0,
                "eta_seconds": None,
                "elapsed_seconds": 0,
                "collection_status": "queued",
            },
            "result": None,
            "error": None,
        }
    thread = threading.Thread(target=run_job, args=(job_id, req), daemon=True)
    thread.start()
    return AnalysisJobCreated(job_id=job_id)


@app.get("/analysis-jobs/{job_id}")
def get_analysis_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Analysis job not found")
        return dict(job)


@app.post("/run-analysis")
def run_analysis(req: AnalysisRequest) -> dict[str, Any]:
    return run_analysis_pipeline(req)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
