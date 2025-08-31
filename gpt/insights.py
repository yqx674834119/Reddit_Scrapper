# gpt/insights.py

import os
from typing import List, Dict, Any

from utils.helpers import estimate_tokens, sanitize_text
from utils.logger import setup_logger
from config.config_loader import get_config
# Ark OpenAI compatible client
from openai import OpenAI
from volcenginesdkarkruntime import Ark


log = setup_logger()
config = get_config()
PROMPT_PATH = "gpt/prompts/insight_prompt.txt"
CLUSTER_PROMPT_PATH = "gpt/prompts/cluster_prompt.txt"
# Initialize Ark OpenAI client
client = Ark(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=os.environ.get("ARK_API_KEY"),
)


def load_insight_prompt_template() -> str:
    """Load the insight prompt template from file."""
    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log.error(f"Error loading insight prompt template: {str(e)}")
        return (
            "Extract:\n"
            "1. The core pain point\n"
            "2. Lead type\n"
            "3. 1–3 relevant marketing tags\n"
            "4. ROI weight (1–5)\n"
            "5. Justification\n\n"
            "Respond with JSON."
        )


INSIGHT_PROMPT_TEMPLATE = load_insight_prompt_template()


def build_insight_prompt(post: dict) -> List[Dict[str, str]]:
    """Constructs the GPT-4.1 prompt for extracting deeper insights using template."""
    return [
        {
            "role": "system",
            "content": "You are a SaaS strategist extracting pain points and marketing tags from Reddit posts."
        },
        {
            "role": "user",
            "content": f"Post title: {post['title']}\nPost body: {post['body']}\n\n{INSIGHT_PROMPT_TEMPLATE}"
        }
    ]



def load_cluster_prompt_template() -> str:
    """Load the cluster prompt template from file."""
    try:
        with open(CLUSTER_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log.error(f"Error loading cluster prompt template: {str(e)}")
        return (
            "Extract:\n"
            "1. The core pain point\n"
            "2. Lead type\n"
            "3. 1–3 relevant marketing tags\n"
            "4. ROI weight (1–5)\n"
            "5. Justification\n\n"
            "Respond with JSON."
        )
    
CLUSTER_PROMPT_TEMPLATE = load_cluster_prompt_template()

def build_cluster_prompt(cluster: dict) -> List[Dict[str, str]]:
    """Constructs the GPT-4.1 prompt for extracting deeper clusters using template."""
    return [
        {
            "role": "system",
            "content": "You are a SaaS strategist specializing in clustering user pain points from Reddit posts."
        },
        {
            "role": "user",
            "content": f"Post title: {cluster['title']}\nCollected pain points:\n{cluster['pain_points']}\n\n{CLUSTER_PROMPT_TEMPLATE}"
        }
    ]

def prepare_insight_batch(posts: List[dict]) -> List[Dict[str, Any]]:
    """Prepares GPT-4.1 insight batch payload."""
    model = config["openai"].get("model_deep", "gpt-4.1")
    payload = []

    for post in posts:
        raw_title = post.get("title", "")
        raw_body = post.get("body", "")
        title = sanitize_text(raw_title)
        body = sanitize_text(raw_body)

        if not title or not body:
            continue  # skip malformed posts

        messages = build_insight_prompt({"title": title, "body": body})
        payload.append({
            "id": post["id"],
            "messages": messages,
            "meta": {
                "estimated_tokens": estimate_tokens(title + body, model)
            }
        })
    return payload
def prepare_cluster_batch(posts: List[dict]) -> List[Dict[str, Any]]:
    model = config["openai"].get("model_deep", "gpt-4.1")
    payload = []
    grouped_posts: Dict[str, Dict[str, Any]] = {}
    for post in posts:
        title_id = post.get("title_id", "")                
        raw_title = post.get("title", "")  # 真实标题
        raw_pain_point = post.get("pain_point", "")  
        title = sanitize_text(raw_title)
        pain_point = sanitize_text(raw_pain_point)

        if not pain_point:
            continue  # skip malformed posts
        if title_id not in grouped_posts:            
            grouped_posts[title_id] = {
                "pain_points": [],
                "post_title_id": title_id,
                "post_title": title 
            }

        grouped_posts[title_id]["pain_points"].append(pain_point)

    # Step 2: 为每个 title 构建聚合任务
    # 构建 payload
    for title_id, data in grouped_posts.items():
        pain_points = data["pain_points"]
        post_title = data["post_title"]
        post_title_id = data["post_title_id"]

        merged_pain_points = "\n- " + "\n- ".join(pain_points)
        messages = build_cluster_prompt({
            "title": post_title,
            "pain_points": merged_pain_points
        })

        payload.append({
            "id": post_title_id,  # ✅ 用主 post 的 id
            "messages": messages,
            "meta": {
                "estimated_tokens": estimate_tokens(title + merged_pain_points, model),
                "title": post_title,
                "title_id": post_title_id,
                "num_pain_points": len(pain_points),
            }
        })
    return payload

def estimate_insight_cost(batch: List[Dict]) -> float:
    """Estimate GPT-4.1 cost using real input token metadata."""
    cost_per_1k_input = 0.0020
    cost_per_1k_output = 0.0080
    discount = 0.05  # Batch API discount

    input_tokens = sum(item.get("meta", {}).get("estimated_tokens", 700) for item in batch)
    output_tokens = len(batch) * 300  # assume output tokens

    return (
        (input_tokens / 1_000_000 * cost_per_1k_input) +
        (output_tokens / 1_000_000 * cost_per_1k_output)
    ) * discount
