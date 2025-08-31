# scheduler/runner.py

import json
import uuid
import os
import glob
from datetime import datetime
import time
from reddit.scraper import scrape_all_configured_subreddits
from db.writer import insert_post, update_post_filter_scores, update_post_insight, mark_insight_processed,update_post_cluster
from db.reader import get_top_insights_from_today, get_posts_by_ids
from db.schema import create_tables
from gpt.filters import prepare_batch_payload as prepare_filter_batch, estimate_batch_cost as estimate_filter_cost
from gpt.insights import prepare_insight_batch, estimate_insight_cost,prepare_cluster_batch
from gpt.batch_api import generate_batch_payload, process_batch_async, download_batch_results, add_estimated_batch_cost
from db.cleaner import clean_old_entries
from scheduler.cost_tracker import initialize_cost_tracking, can_process_batch
from config.config_loader import get_config
from utils.logger import setup_logger
from utils.helpers import ensure_directory_exists, sanitize_text
import asyncio
log = setup_logger()
config = get_config()
async def submit_with_backoff(batch_items, model, generate_file_fn=None, label="filter") -> str | None:
    """
    提交 Ark batch 请求，带退避重试。
    注意：generate_file_fn 在 Ark 模式下已无意义，这里保留参数只是为了兼容调用。
    """
    delay = 10
    max_retries = 20

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"[Retry {attempt}/{max_retries}] Submitting {label} batch with {len(batch_items)} items...")

            # 生成 batch_id
            requests = generate_batch_payload(batch_items, model)

            # 提交任务
            results, errors = await process_batch_async(requests, model,max_workers=len(requests)//10 or 10)

            if errors:
                log.warning(f"Batch contains {len(errors)} errors. Retrying...")
                if attempt >= max_retries:
                    save_failed_batch(errors, label)
                    return None
                await asyncio.sleep(delay)
                delay = min(delay * 2, 3600)
                continue
            # 保存结果
            batch_id = uuid.uuid4().hex
            result_path = f"data/batch_responses/{label}_result_{batch_id}.jsonl"
            download_batch_results(results, result_path)
            log.info(f"{label.capitalize()} batch completed. Results saved to {result_path}")
            return result_path
        except Exception as e:
            log.error(f"Error in {label} batch retry #{attempt}: {str(e)}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 3600)

    # 全部失败
    log.error(f"❌ {label.capitalize()} batch failed after {max_retries} retries. Deferring.")
    save_failed_batch(batch_items, label)
    return None

def save_failed_batch(batch_items, label, folder="data/deferred"):
    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, f"failed_{label}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in batch_items:
            f.write(json.dumps(item) + "\n")
    log.warning(f"Deferred {len(batch_items)} {label} items to {out_path}")

def is_valid_post(post):
    """Ensure post has valid title and body after sanitization."""
    title = sanitize_text(post.get("title", ""))
    body = sanitize_text(post.get("body", ""))
    return bool(title and body)

def split_batch_by_token_limit(payload, model: str, token_limit: int = 200_000):
    batches = []
    current_batch = []
    current_tokens = 0

    for item in payload:
        tokens = item.get("meta", {}).get("estimated_tokens", 300)
        if current_tokens + tokens > token_limit:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(item)
        current_tokens += tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def clean_old_batch_files(folder="data/batch_responses", days_old=None):
    """Delete .jsonl files older than `days_old`. Defaults to config value."""
    days_old = days_old or config.get("cleanup", {}).get("batch_response_retention_days", 3)
    cutoff = time.time() - (days_old * 86400)

    deleted = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.endswith(".jsonl") and os.path.isfile(path):
            if os.path.getmtime(path) < cutoff:
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    log.warning(f"Failed to delete old file {path}: {e}")
    log.info(f"Cleaned up {deleted} old batch response files older than {days_old} days.")

def get_high_potential_ids_from_filter_results(score_threshold=7.0):
    high_ids = set()
    weights = config["scoring"]
    for path in glob.glob("data/batch_responses/filter_result_*.jsonl"):
        with open(path, "r") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    post_id = result["custom_id"]
                    content = result["response"]["choices"][0]["message"]["content"]
                    scores = json.loads(content)
                    weighted_score = (
                        scores["relevance_score"] * weights["relevance_weight"] +
                        scores["emotional_intensity"] * weights["emotion_weight"] +
                        scores["pain_point_clarity"] * weights["pain_point_weight"]
                    )
                    if weighted_score >= score_threshold:
                        high_ids.add(post_id)
                        update_post_filter_scores(post_id, scores)
                    # log.info(f"Path {path} Post {post_id} scored {weighted_score:.2f} (Threshold: {score_threshold})")
                except Exception as e:
                    log.error(f"Error parsing filter result line: {e}")
                    continue
    return high_ids

def run_daily_pipeline():
    log.info("\U0001F680 Starting Reddit scraping and analysis pipeline")

    ensure_directory_exists("data/deferred")
    ensure_directory_exists("data")
    ensure_directory_exists("data/batch_responses")
    clean_old_batch_files()
    create_tables()
    initialize_cost_tracking()

    log.info("Step 1: Cleaning old database entries...")
    clean_old_entries()

    log.info("Step 2: Scraping Reddit posts...")
    scraped_posts = scrape_all_configured_subreddits()
    # 这里可以改成 从数据库中获取
    if not scraped_posts:
        log.warning("No posts found to analyze. Exiting pipeline.")
        return

    log.info(f"Found {len(scraped_posts)} posts before filtering invalid entries...")
    scraped_posts = [p for p in scraped_posts if is_valid_post(p)]
    log.info(f"{len(scraped_posts)} posts remain after sanitization/validation.")

    if not scraped_posts:
        log.warning("No valid posts after sanitization. Exiting pipeline.")
        return

    log.info("Step 3: Preparing posts for filtering...")
    filter_batch = prepare_filter_batch(scraped_posts)
    filter_cost = estimate_filter_cost(scraped_posts)
    log.info(f"Estimated cost for filtering: ${filter_cost:.2f}")

    if not can_process_batch(filter_cost):
        log.error("Insufficient budget for filtering. Exiting pipeline.")
        return

    model_filter = config["openai"]["model_filter"]
    filter_batches = split_batch_by_token_limit(filter_batch, model_filter)

    for i, batch in enumerate(filter_batches):
        log.info(f"Submitting sub-batch {i + 1}/{len(filter_batches)} with {len(batch)} entries...")
        add_estimated_batch_cost(batch, model_filter)

        results_path = asyncio.run(submit_with_backoff(
            batch_items=batch,
            model=model_filter,
            generate_file_fn=generate_batch_payload,
            label="filter"
        ))

        if not results_path:
            continue  # move on to next batch

    log.info("Step 4: Selecting high-potential posts from filter results...")
    high_potential_ids = get_high_potential_ids_from_filter_results()
    if not high_potential_ids:
        log.info("No high-value posts found. Exiting pipeline.")
        return

    deep_posts = get_posts_by_ids(high_potential_ids, require_unprocessed=True)
    if not deep_posts:
        log.info("No new posts left for deep insight. Exiting pipeline.")
        return

    insight_batch = prepare_insight_batch(deep_posts)
    insight_cost = estimate_insight_cost(insight_batch)
    log.info(f"Estimated cost for insight analysis: ${insight_cost:.2f}")

    if not can_process_batch(insight_cost):
        log.error("Insufficient budget for insight analysis. Exiting pipeline.")
        return

    log.info(f"Submitting batch of {len(insight_batch)} posts for deep analysis...")
    log.info(f"Preparing {len(insight_batch)} posts for deep insight...")
    model_deep = config["openai"]["model_deep"]
    insight_batches = split_batch_by_token_limit(insight_batch, model_deep)
    all_insight_paths = []

    for i, batch in enumerate(insight_batches):
        log.info(f"Submitting insight sub-batch {i + 1}/{len(insight_batches)} with {len(batch)} entries...")
        add_estimated_batch_cost(batch, model_deep)

        insight_path = asyncio.run(submit_with_backoff(
            batch_items=batch,
            model=model_deep,
            generate_file_fn=generate_batch_payload,
            label="insight"
        ))

        if not insight_path:
            continue

        all_insight_paths.append(insight_path)

    log.info("Step 5: Updating posts with deep insights...")
    insight_post_id = []
    try:
        for insight_path in all_insight_paths:
            with open(insight_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    post_id = result["custom_id"]
                    content = result["response"]["choices"][0]["message"]["content"]
                    try:
                        insight = json.loads(content)
                        update_post_insight(post_id, insight)
                        mark_insight_processed(post_id)
                        insight_post_id.append(post_id)
                    except Exception as e:
                        log.error(f"Error parsing insight for post {post_id}: {str(e)}")
    except Exception as e:
        log.error(f"Error reading insight results: {str(e)}")

    log.info("Step 6: Clustering similar insights...")
    # 聚合相同title下的痛点，全部放在 Post Pain point 下面
    # 1. 获取所有已经 insight_processed 的 comment 和 post
    insight_post = get_posts_by_ids(insight_post_id, require_unprocessed=False)
    # 2. 按 title 聚合同一个帖子下的所有 pain_point
    cluster_batch = prepare_cluster_batch(insight_post)
    model_deep = config["openai"]["model_deep"]
    cluster_batchs = split_batch_by_token_limit(cluster_batch, model_deep)
    all_cluster_paths = []

    for i, batch in enumerate(cluster_batchs):
        log.info(f"Submitting insight sub-batch {i + 1}/{len(cluster_batchs)} with {len(batch)} entries...")
        cluster_path = asyncio.run(submit_with_backoff(
            batch_items=batch,
            model=model_deep,
            generate_file_fn=None,
            label="cluster"
        ))
        if not cluster_path:
            continue

        all_cluster_paths.append(cluster_path)
    # 写入到数据库中
    try:
        for cluster_path in all_cluster_paths:
            with open(cluster_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    post_id = result["custom_id"]
                    content = result["response"]["choices"][0]["message"]["content"]
                    try:                        
                        update_post_cluster(post_id, content)
                        mark_insight_processed(post_id)
                    except Exception as e:
                        log.error(f"Error parsing insight for post {post_id}: {str(e)}")
    except Exception as e:
        log.error(f"Error reading insight results: {str(e)}")


    output_limit = config["scoring"]["output_top_n"]
    top_posts = get_top_insights_from_today(limit=output_limit)      
    log.info(f"✅ Pipeline finished. Found {len(top_posts)} qualified leads.")
 
    for i, post in enumerate(top_posts):
        log.info(f"{i}. [{post['subreddit']}] {post['title']} — Pain Point: {post['pain_point']} | ROI: {post['roi_weight']} | Tags: {post['tags']} - {post['url']}")


if __name__ == "__main__":
    run_daily_pipeline()
