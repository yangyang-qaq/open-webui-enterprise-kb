"""Mock OpenAI 兼容 LLM / Embedding Server，用于压测时挡住外部 AI 调用。

只挡三个外部接口，让压测聚焦在 OWUI 自身的鉴权 / DB / Chroma / RAG 链路：

    GET  /v1/models
    POST /v1/chat/completions   （SSE 流式 + 非流式）
    POST /v1/embeddings         （固定 384 维向量，对齐 all-MiniLM-L6-v2）

启动（用 backend venv 的 python，已含 fastapi/uvicorn）：

    cd open-webui/backend
    ./.venv/Scripts/python.exe ../../loadtest/mock/llm_mock.py
    # 或：uvicorn loadtest.mock.llm_mock:app --port 9099

可调参数（环境变量 / 请求 header）：
    MOCK_EMB_DIM=384       向量维度（必须与已建 Chroma collection 一致）
    MOCK_MODEL=mock-chat   返回的模型名
    x-mock-delay           首 token 前延迟（秒），默认 1.5，Locust 可用它模拟不同思考时长
    x-mock-chunk-delay     每 chunk 间隔（秒），默认 0.1
    x-mock-n-chunks        chunk 数量，默认 15
"""

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM Server")

EMB_DIM = int(os.getenv("MOCK_EMB_DIM", "384"))
MODEL = os.getenv("MOCK_MODEL", "mock-chat")
BASE_DELAY = float(os.getenv("MOCK_BASE_DELAY", "1.5"))
CHUNK_DELAY = float(os.getenv("MOCK_CHUNK_DELAY", "0.1"))
N_CHUNKS = int(os.getenv("MOCK_N_CHUNKS", "15"))


def _chunk(i: int) -> dict:
    """构造一个 OpenAI 兼容的流式 chunk。"""
    return {
        "id": "cmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"tok{i} "} if i == 0 else {"content": f"tok{i} "},
                "finish_reason": None,
            }
        ],
    }


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    # header 可覆盖默认延迟，供 Locust 模拟不同思考时长（低/高延迟两档对比）
    base = float(request.headers.get("x-mock-delay", BASE_DELAY))
    chunk_delay = float(request.headers.get("x-mock-chunk-delay", CHUNK_DELAY))
    n = int(request.headers.get("x-mock-n-chunks", N_CHUNKS))

    stream = bool(body.get("stream", False))

    if not stream:
        # 非流式：一次性返回完整 JSON（Agent 工作流 / 标题生成等内部调用会用到）
        content = " ".join(f"tok{i}" for i in range(n))
        return JSONResponse(
            {
                "id": "cmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
            }
        )

    async def gen():
        await asyncio.sleep(base)
        for i in range(n):
            yield f"data: {json.dumps(_chunk(i), ensure_ascii=False)}\n\n"
            await asyncio.sleep(chunk_delay)
        done = {
            "id": "cmpl-mock",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    inp = body.get("input", [])
    if isinstance(inp, str):
        inp = [inp]
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.0123456] * EMB_DIM}
                for i in range(len(inp))
            ],
            "model": "mock-embedding",
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9099)
