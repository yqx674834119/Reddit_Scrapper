import asyncio
import uuid
import json
from gpt.batch_api import generate_batch_payload, submit_batch_job, poll_batch_status, download_batch_results

def make_dummy_post(i):
    return {
        "id": f"test_{i}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Say hello world {i}"}
        ],
        "meta": {"estimated_tokens": 10}
    }

async def test_batch():
    # 构造10条测试数据
    posts = [make_dummy_post(i) for i in range(10)]
    model = "ep-bi-20250825173518-q44dq"  # Ark模型名，可根据实际情况修改
    batch_id = await generate_batch_payload(posts, model)
    await submit_batch_job(batch_id)
    batch_info = await poll_batch_status(batch_id, timeout_seconds=300)
    print(f"Batch status: {batch_info['status']}")
    save_path = f"data/batch_responses/test_batch_{uuid.uuid4().hex}.jsonl"
    await download_batch_results(batch_id, save_path)
    print(f"Results saved to {save_path}")
    # 打印部分结果
    with open(save_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(json.loads(line))

if __name__ == "__main__":
    asyncio.run(test_batch())
