import sqlite3
from config.config_loader import get_config
from datetime import datetime, UTC
from pathlib import Path

config = get_config()
DB_PATH = config["database"]["path"]

_conn = None

def _get_connection():
    global _conn
    if _conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
    return _conn

def insert_post(post: dict, community_type: str = "primary"):
    conn = _get_connection()
    try:
        conn.execute("""
        INSERT OR IGNORE INTO posts (
            id, url, title,title_id, body, subreddit, created_utc, last_active,
            processed_at, community_type, type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post["id"],
            post["url"],
            post["title"],
            post["title_id"],
            post.get("body", ""),
            post["subreddit"],
            post["created_utc"],
            post["created_utc"],
            datetime.now(UTC).date().isoformat(),
            community_type,
            post.get("type", "post")
        ))

        conn.execute("""
        INSERT OR IGNORE INTO history (id, processed_at)
        VALUES (?, ?)
        """, (
            post["id"],
            datetime.utcnow().isoformat()
        ))

        conn.commit()
    except sqlite3.Error as e:
        print(f"[SQLite Insert Error] {e}")

def update_post_filter_scores(post_id: str, scores: dict):
    """Update filtering phase scores only (relevance, emotion, pain)."""
    conn = _get_connection()
    try:
        conn.execute("""
        UPDATE posts SET
            relevance_score = ?,
            emotion_score = ?,
            pain_score = ?,
            processed_at = ?
        WHERE id = ?
        """, (
            scores.get("relevance_score"),
            scores.get("emotional_intensity"),
            scores.get("pain_point_clarity"),
            datetime.now(UTC).date().isoformat(),
            post_id
        ))
        conn.commit()
    except sqlite3.Error as e:
        print(f"[SQLite update_post_filter_scores Error] {e}")

def update_post_insight(post_id: str, insight: dict):
    """Update deeper insights (lead_type, tags, roi_weight). Safe from overwriting with nulls."""
    conn = _get_connection()
    cursor = conn.cursor()

    fields = {
        "lead_type": insight.get("lead_type"),
        "tags": ", ".join(insight["tags"]) if "tags" in insight else None,
        "roi_weight": insight.get("roi_weight"),
        "pain_point": insight.get("pain_point"),
        "potential_solution": insight.get("potential_solution"),
    }

    updates = [f"{key} = ?" for key, value in fields.items() if value is not None]
    values = [value for value in fields.values() if value is not None]

    if not updates:
        return  # No valid fields to update

    updates.append("insight_processed = 1")
    # updates.append("insight_processed_at = ?")
    # values.append(datetime.utcnow().isoformat())

    query = f"""
        UPDATE posts SET {', '.join(updates)}
        WHERE id = ?
    """
    try:
        cursor.execute(query, values + [post_id])
        conn.commit()
    except sqlite3.Error as e:
        print(f"[SQLite update_post_insight Error] {e}")
def update_post_cluster(post_id: str, cluster: str):
    """Update deeper insights (lead_type, tags, roi_weight). Safe from overwriting with nulls."""
    conn = _get_connection()
    cursor = conn.cursor()

    fields = {
        "pain_point": cluster,
    }

    updates = [f"{key} = ?" for key, value in fields.items() if value is not None]
    values = [value for value in fields.values() if value is not None]

    if not updates:
        return  # No valid fields to update

    updates.append("insight_processed = 1")
    # updates.append("insight_processed_at = ?")
    # values.append(datetime.utcnow().isoformat())

    query = f"""
        UPDATE posts SET {', '.join(updates)}
        WHERE id = ?
    """
    try:
        cursor.execute(query, values + [post_id])
        conn.commit()
    except sqlite3.Error as e:
        print(f"[SQLite update_post_insight Error] {e}")


def mark_insight_processed(post_id: str):
    """Mark a post as having been processed for deep insight."""
    conn = _get_connection()
    try:
        conn.execute("""
        UPDATE posts SET insight_processed = 1
        WHERE id = ?
        """, (post_id,))
        conn.commit()
    except sqlite3.Error as e:
        print(f"[SQLite mark_insight_processed Error] {e}")
