"""full 档检索：HTTP 调 /evaluate/query 拿真实「混合检索 + 重排」片段。

复用 loadtest 的 signin 模式（登录一次拿 Bearer token）。
仅 full 档需要后端在跑；judge-only 档不 import 本模块。
"""

import requests

import config


def signin() -> str:
    url = f"{config.BACKEND_URL}{config.API}/auths/signin"
    r = requests.post(
        url,
        json={"email": config.EVAL_USER_EMAIL, "password": config.EVAL_USER_PASSWORD},
        timeout=30,
    )
    r.raise_for_status()
    token = r.json().get("token", "")
    if not token:
        raise RuntimeError(f"signin 未返回 token: {r.text[:200]}")
    return token


def retrieve_chunks(question: str, kb_id: str, token: str, k: int | None = None) -> list[dict]:
    """调 /evaluate/query（已定制返回完整 k 条），映射成统一 context_chunk 结构。"""
    k = k or config.RETRIEVAL_K
    url = f"{config.BACKEND_URL}{config.API}/knowledge/{kb_id}/evaluate/query"
    r = requests.post(
        url,
        json={"query": question, "k": k},
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    r.raise_for_status()
    chunks = []
    for item in r.json().get("results", []):
        meta = item.get("metadata") or {}
        chunks.append(
            {
                "text": item.get("text", ""),
                "source": meta.get("name") or meta.get("source") or meta.get("file_id", ""),
                "score": item.get("score"),
                "rank": item.get("rank"),
            }
        )
    return chunks
