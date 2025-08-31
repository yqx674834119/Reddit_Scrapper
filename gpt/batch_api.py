import json
import uuid
import time
import os
import openai
import requests
from config.config_loader import get_config
from scheduler.cost_tracker import add_cost
from utils.logger import setup_logger
import os
import asyncio
import sys
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk
from tqdm.asyncio import tqdm 
log = setup_logger()
config = get_config()

BATCH_FILE_PATH = "data/batch_requests.jsonl"
RESPONSE_DIR = "data/batch_responses"

def generate_batch_payload(requests: list[dict], model: str) -> list[dict]:
    """生成符合AsyncArk格式的请求列表"""
    return [{
        "custom_id": prompt.get("id", str(uuid.uuid4())),
        "model": model,
        "messages": prompt["messages"],
        "temperature": 0,
        "thinking": {"type": "disabled"}
    } for prompt in requests]
async def process_batch_async(
    requests: list[dict], 
    model: str, 
    max_workers: int = 1000
) -> tuple[list[dict], list[dict]]:
    #使用AsyncArk worker队列处理批量请求"
    client = AsyncArk(
        api_key=os.getenv("ARK_API_KEY"),
        timeout=24 * 3600
    )
    request_queue = asyncio.Queue()
    results = []
    errors = []
    total_tasks = len(requests)
    # 初始化进度条
    # pbar = tqdm(total=total_tasks, desc=f"Processing {model} batch", unit="task")

    # 填充请求队列
    for req in requests:
        await request_queue.put((req["custom_id"], req))

    # 定义worker协程
    async def worker(worker_id: int):
        nonlocal results, errors
        while True:
            try:
                custom_id, req = await request_queue.get()
                response = await client.chat.completions.create(
                    model=model,
                    messages=req["messages"],
                    temperature=req["temperature"],
                    thinking=req["thinking"]
                )
                print(custom_id)
                results.append({
                    "custom_id": custom_id,
                    "response": response.dict()
                })
                # pbar.update(1)
                # pbar.set_postfix_str(f"Completed: {len(results)}/{total_tasks}")
            except Exception as e:
                errors.append({
                    "custom_id": custom_id,
                    "error": str(e)
                })
                log.error(f"Worker {worker_id} failed: {str(e)}")
                # pbar.update(1)  # 即使出错也更新进度
            finally:
                request_queue.task_done()

    # 创建并启动worker
    workers = [asyncio.create_task(worker(i)) for i in range(max_workers)]
    await request_queue.join()

    # 取消worker任务
    # pbar.close()
    for worker_task in workers:
        worker_task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    await client.close()

    return results, errors


def download_batch_results(results: list[dict], save_path: str):
    # """保存批量处理结果到JSONL文件"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    log.info(f"Saved {len(results)} results to {save_path}")

def add_estimated_batch_cost(requests: list[dict], model: str):
    """Estimate and record the cost of the batch job using accurate pricing."""
    # Per 1M token pricing in USD
    pricing = {
        "gpt-4.1": {"input": 0.0020, "output": 0.0080},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    discount_factor = 0.5  # 50% discount for batch jobs

    # Default fallback pricing
    model_pricing = pricing.get(model, {"input": 0.0010, "output": 0.0010})

    input_tokens = sum(req.get("meta", {}).get("estimated_tokens", 300) for req in requests)
    output_tokens = len(requests) * 300  # Assumes ~300 output tokens per completion

    input_cost = (input_tokens / 200_000) * model_pricing["input"] * discount_factor
    output_cost = (output_tokens / 500_000) * model_pricing["output"] * discount_factor
    estimated_cost = input_cost + output_cost

    log.info(f"Estimated cost for batch (input + output): ${estimated_cost:.4f}")
    add_cost(estimated_cost)
