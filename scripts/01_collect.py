"""
01_collect.py
=============
Collect all posts and comments from r/MohsSurgery using Reddit's public JSON API.
No API credentials required.

What this script does, step by step:
  1. Paginates through r/MohsSurgery using the /new.json endpoint (100 posts/page).
  2. For each post it collects:
       - The post itself (title + body text)
       - Every comment in the thread by fetching /comments/{id}.json
         and recursively walking the reply tree.
  3. Writes everything to data/raw/mohs_raw.csv with columns:
       id, type, parent_id, body, created_utc, score

Run from the project root:
    python scripts/01_collect.py
"""

import os
import csv
import time
import logging
import requests

from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_PATH    = os.path.join("data", "raw", "mohs_raw.csv")
SUBREDDIT_NAME = "MohsSurgery"
WRITE_BATCH_SIZE = 500
HEADERS = {"User-Agent": "windows:mohs-reddit-lda:v1.0 (academic research project)"}

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helper functions ───────────────────────────────────────────────────────────

def _walk_comments(children, out):
    """Recursively walk a Reddit comment tree, collecting comment dicts."""
    for child in children:
        if child.get("kind") == "more":
            continue
        d = child.get("data", {})
        body = d.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            out.append({
                "id":          d.get("id", ""),
                "type":        "comment",
                "parent_id":   d.get("parent_id", ""),
                "body":        body,
                "created_utc": int(d.get("created_utc", 0)),
                "score":       d.get("score", 0),
            })
        replies = d.get("replies", "")
        if isinstance(replies, dict):
            _walk_comments(replies["data"]["children"], out)


def fetch_comments(post_id):
    """Fetch and flatten comments for a single post. Returns list of row dicts."""
    url = f"https://www.reddit.com/r/{SUBREDDIT_NAME}/comments/{post_id}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning(f"Could not fetch comments for {post_id}: {exc}")
        return []

    rows = []
    _walk_comments(data[1]["data"]["children"], rows)
    return rows


# ── Main collection logic ──────────────────────────────────────────────────────

def collect():
    """
    Page through r/MohsSurgery/new.json, yielding one dict per post or comment.
    """
    base_url = f"https://www.reddit.com/r/{SUBREDDIT_NAME}/new.json"
    after = None
    post_count = 0
    comment_count = 0

    with tqdm(desc="Posts collected", unit="post") as pbar:
        while True:
            params = {"limit": 100, "raw_json": 1}
            if after:
                params["after"] = after

            try:
                resp = requests.get(base_url, headers=HEADERS, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.error(f"Request failed: {exc}")
                break

            children = data["data"]["children"]
            if not children:
                break

            for child in children:
                post = child["data"]
                body = f"{post.get('title', '')} {post.get('selftext', '')}".strip()

                yield {
                    "id":          post.get("id", ""),
                    "type":        "post",
                    "parent_id":   "",
                    "body":        body,
                    "created_utc": int(post.get("created_utc", 0)),
                    "score":       post.get("score", 0),
                }
                post_count += 1
                pbar.update(1)

                comments = fetch_comments(post["id"])
                for row in comments:
                    yield row
                    comment_count += 1

                time.sleep(1)

            after = data["data"].get("after")
            if not after:
                break

            time.sleep(1)

    log.info(f"Collection complete: {post_count} posts, {comment_count} comments")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fieldnames = ["id", "type", "parent_id", "body", "created_utc", "score"]
    log.info(f"Writing output to {OUTPUT_PATH} ...")

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch = []
        total = 0

        for row in collect():
            batch.append(row)
            if len(batch) >= WRITE_BATCH_SIZE:
                writer.writerows(batch)
                total += len(batch)
                batch = []

        if batch:
            writer.writerows(batch)
            total += len(batch)

    log.info(f"Done. {total} total rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
