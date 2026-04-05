"""
01_collect.py
=============
Collect all posts and comments from r/MohsSurgery using the Reddit API (PRAW).

What this script does, step by step:
  1. Loads Reddit API credentials from a .env file (never hardcoded).
  2. Connects to Reddit via PRAW.
  3. Fetches every available post from r/MohsSurgery using the "new" listing
     (PRAW caps a single listing call at ~1,000; the script pages through all
     available results automatically via a generator).
  4. For each post it collects:
       - The post itself (title + body text)
       - Every comment in the thread, including nested replies, by calling
         submission.comments.replace_more(limit=None) followed by
         submission.comments.list() to fully flatten the comment tree.
  5. Writes everything to data/raw/mohs_raw.csv with columns:
       id, type, parent_id, body, created_utc, score

Run from the project root:
    python scripts/01_collect.py
"""

import os
import csv
import time
import logging

import praw
from dotenv import load_dotenv
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

# Output path (relative to project root)
OUTPUT_PATH = os.path.join("data", "raw", "mohs_raw.csv")

# Subreddit to collect from
SUBREDDIT_NAME = "MohsSurgery"

# How many API requests to batch before writing to disk
# (keeps memory usage low for large subreddits)
WRITE_BATCH_SIZE = 500

# Seconds to wait between retries on rate-limit errors
RETRY_WAIT = 60

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helper functions ───────────────────────────────────────────────────────────

def load_credentials():
    """Load Reddit API credentials from the .env file in the project root."""
    # Walk up from scripts/ to find the .env at the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, ".env")

    if not os.path.exists(dotenv_path):
        raise FileNotFoundError(
            f".env file not found at {dotenv_path}. "
            "Copy .env.example to .env and fill in your Reddit credentials."
        )

    load_dotenv(dotenv_path)

    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = os.getenv("REDDIT_USER_AGENT")

    missing = [k for k, v in {
        "REDDIT_CLIENT_ID":     client_id,
        "REDDIT_CLIENT_SECRET": client_secret,
        "REDDIT_USER_AGENT":    user_agent,
    }.items() if not v]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Check your .env file."
        )

    return client_id, client_secret, user_agent


def build_reddit_client(client_id, client_secret, user_agent):
    """Create and return a read-only PRAW Reddit instance."""
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        # read_only prevents accidental write operations
        ratelimit_seconds=300,
    )


def post_to_row(submission):
    """Convert a PRAW Submission object into a CSV-ready dict."""
    # Combine title and body so both are searchable in later analysis
    body = f"{submission.title} {submission.selftext}".strip()
    return {
        "id":          submission.id,
        "type":        "post",
        "parent_id":   "",               # posts have no parent
        "body":        body,
        "created_utc": int(submission.created_utc),
        "score":       submission.score,
    }


def comment_to_row(comment):
    """Convert a PRAW Comment object into a CSV-ready dict."""
    return {
        "id":          comment.id,
        "type":        "comment",
        "parent_id":   comment.parent_id,   # e.g. "t3_abc123" or "t1_xyz789"
        "body":        comment.body,
        "created_utc": int(comment.created_utc),
        "score":       comment.score,
    }


# ── Main collection logic ──────────────────────────────────────────────────────

def collect(reddit):
    """
    Fetch all posts and their comments from the target subreddit.

    Yields one dict per row (post or comment) suitable for CSV writing.
    """
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    log.info(f"Connected to r/{SUBREDDIT_NAME}")

    # PRAW's .new() generator pages through ALL available posts automatically.
    # "new" ordering is used so we get a consistent, reproducible crawl.
    # limit=None tells PRAW to keep fetching until Reddit returns no more posts.
    submissions = subreddit.new(limit=None)

    post_count    = 0
    comment_count = 0

    for submission in tqdm(submissions, desc="Posts collected", unit="post"):
        # ── Yield the post itself ──────────────────────────────────────────────
        yield post_to_row(submission)
        post_count += 1

        # ── Expand and flatten the comment tree ───────────────────────────────
        # replace_more(limit=None) fetches all "load more comments" stubs.
        # This can make additional API calls, so we wrap it in a retry loop.
        try:
            submission.comments.replace_more(limit=None)
        except Exception as exc:
            log.warning(f"Could not fully expand comments for {submission.id}: {exc}")

        for comment in submission.comments.list():
            # Skip deleted/removed comments that have no text
            if not comment.body or comment.body in ("[deleted]", "[removed]"):
                continue
            yield comment_to_row(comment)
            comment_count += 1

    log.info(f"Collection complete: {post_count} posts, {comment_count} comments")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # Step 1: Load credentials
    log.info("Loading credentials from .env ...")
    client_id, client_secret, user_agent = load_credentials()

    # Step 2: Connect to Reddit
    log.info("Connecting to Reddit API ...")
    reddit = build_reddit_client(client_id, client_secret, user_agent)

    # Step 3: Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Step 4: Collect posts and comments, writing to CSV in batches
    fieldnames = ["id", "type", "parent_id", "body", "created_utc", "score"]

    log.info(f"Writing output to {OUTPUT_PATH} ...")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch  = []
        total  = 0

        for row in collect(reddit):
            batch.append(row)
            if len(batch) >= WRITE_BATCH_SIZE:
                writer.writerows(batch)
                total += len(batch)
                batch = []

        # Write any remaining rows
        if batch:
            writer.writerows(batch)
            total += len(batch)

    log.info(f"Done. {total} total rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
