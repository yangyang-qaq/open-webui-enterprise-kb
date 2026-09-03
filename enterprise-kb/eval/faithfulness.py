"""Faithfulness 忠实度判定（LLM-as-judge，RAGAS 风格）。

把生成答案拆成原子主张，逐条判断是否被检索片段支持：
  faithfulness = supported / total（no / insufficient 视为不支持）。
拒答（空主张）视为忠实（faithfulness=1.0），因为没有编造。
"""

import json
import re

from openai import OpenAI

import config
from generate import build_context_string

JUDGE_SYSTEM = (
    "You are a strict fact-checker evaluating the faithfulness of a RAG answer. "
    "Judge whether each claim in the answer is supported by the provided context. "
    "Use ONLY the context, never outside knowledge. Output ONLY a valid JSON object, no markdown fences."
)

JUDGE_PROMPT = """Given a question, the retrieved context (inside <source> tags), and a generated answer:

1. Decompose the answer into a list of atomic claims (each is a single verifiable statement).
2. For each claim, set "supported":
   - "yes"         → the context directly supports the claim;
   - "no"          → the context contradicts the claim;
   - "insufficient" → the context does not mention enough to verify the claim.
3. If the answer is a refusal (e.g. says the information is not found), output an empty claims list.

Output a JSON object exactly in this shape:
{{"claims": [{{"text": "...", "supported": "yes|no|insufficient", "evidence": "..."}}]}}

Question: {question}

Context:
{context}

Answer:
{answer}
"""


def _extract_json(text: str) -> dict:
    """从 LLM 输出稳健抽取 JSON（容忍 markdown 围栏 / 前后缀）。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in judge output: {text[:200]}")
    return json.loads(text[start : end + 1])


def _judge_call(client: OpenAI, question: str, context: str, answer: str) -> dict:
    """调 judge，JSON 解析失败重试一次。"""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_PROMPT.format(question=question, context=context, answer=answer)},
    ]
    last_err = None
    for _ in range(2):
        resp = client.chat.completions.create(
            model=config.EVAL_MODEL, messages=messages, temperature=0, stream=False
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            return _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": "That was not valid JSON. Output ONLY the JSON object as requested."}
            )
    raise last_err


def judge_faithfulness(question: str, answer: str, context_chunks: list[dict]) -> dict:
    """返回 {"faithfulness", "claims", "supported", "total", "refusal"}。"""
    if not answer.strip():
        return {"faithfulness": 0.0, "claims": [], "supported": 0, "total": 0, "refusal": False}

    context = build_context_string(context_chunks)
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    data = _judge_call(client, question, context, answer)
    claims = data.get("claims", [])

    if not claims:
        # 无主张（拒答 / 空输出）→ 视为忠实，没有编造
        return {"faithfulness": 1.0, "claims": [], "supported": 0, "total": 0, "refusal": True}

    supported = sum(1 for c in claims if c.get("supported") == "yes")
    total = len(claims)
    return {
        "faithfulness": round(supported / total, 4),
        "claims": claims,
        "supported": supported,
        "total": total,
        "refusal": False,
    }
