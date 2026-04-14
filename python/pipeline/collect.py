import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HEADERS = {"User-Agent": "windows:mohs-reddit-lda:v1.0 (academic research project)"}
REQUEST_DELAY = 0.5   # seconds between requests
MAX_WORKERS   = 5     # parallel comment fetches


def _fetch_comments(subreddit_name, post_id):
    """
    Fetch and flatten comments for a single post using the Reddit JSON API.
    Returns a list of comment body strings.
    """
    url = f"https://www.reddit.com/r/{subreddit_name}/comments/{post_id}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    comments = []
    _walk_comments(data[1]["data"]["children"], comments)
    return comments


def _walk_comments(children, out):
    """Recursively walk the comment tree, collecting body text."""
    for child in children:
        if child.get("kind") == "more":
            continue
        body = child.get("data", {}).get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            out.append(body)
        replies = child.get("data", {}).get("replies", "")
        if isinstance(replies, dict):
            _walk_comments(replies["data"]["children"], out)


def collect_data(subreddit_name, date_from_str, date_to_str):
    """
    Collect posts and comments from a subreddit within the specified date range.
    Uses Reddit's public JSON API — no credentials required.

    Post listing is paginated sequentially; comments are fetched in parallel.
    """
    dt_from = datetime.strptime(date_from_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ts_from = dt_from.timestamp()

    dt_to = datetime.strptime(date_to_str, "%Y-%m-%d")
    dt_to = dt_to.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    ts_to = dt_to.timestamp()

    # --- Phase 1: paginate post listings to collect posts in range ---
    posts_in_range = []   # list of (post_body, post_id)
    after = None
    base_url = f"https://www.reddit.com/r/{subreddit_name}/new.json"

    while True:
        params = {"limit": 100, "raw_json": 1}
        if after:
            params["after"] = after

        try:
            resp = requests.get(base_url, headers=HEADERS, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch posts from r/{subreddit_name}: {e}")

        children = data["data"]["children"]
        if not children:
            break

        done = False
        for child in children:
            post = child["data"]
            created = post["created_utc"]

            if created < ts_from:
                done = True
                break

            if created <= ts_to:
                body = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                posts_in_range.append((body, post["id"]))

        if done:
            break

        after = data["data"].get("after")
        if not after:
            break

        time.sleep(REQUEST_DELAY)

    # --- Phase 2: fetch comments for all posts in parallel ---
    collected_texts = [body for body, _ in posts_in_range]
    post_ids = [pid for _, pid in posts_in_range]

    def fetch_with_delay(post_id):
        result = _fetch_comments(subreddit_name, post_id)
        time.sleep(REQUEST_DELAY)
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_with_delay, pid): pid for pid in post_ids}
        for future in as_completed(futures):
            try:
                collected_texts.extend(future.result())
            except Exception:
                pass

    return collected_texts
