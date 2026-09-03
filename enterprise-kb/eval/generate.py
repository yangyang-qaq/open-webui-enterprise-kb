"""生成答案：渲染 RAG 模板（含引用约束）+ 调 DeepSeek（非流式，temperature=0）。

对齐 open-webui 生产链路的两处行为：
- get_source_context（utils/middleware.py:798）→ 拼 `<source id=...>...</source>` 块；
- rag_template（utils/task.py:265）→ 替换 {{CONTEXT}} / {{QUERY}} 占位符。
"""

from openai import OpenAI

import config


def build_context_string(context_chunks: list[dict]) -> str:
    """把检索片段拼成 <source> 块，id 从 1 递增，对齐生产 get_source_context。"""
    parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        text = chunk.get("text", "") or chunk.get("content", "")
        src_name = chunk.get("source") or chunk.get("name") or ""
        attrs = f' name="{src_name}"' if src_name else ""
        parts.append(f'<source id="{i}"{attrs}>{text}</source>')
    return "\n".join(parts)


def render_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """渲染 RAG 模板：注入 context 与 query。"""
    template = config.load_rag_template()
    context = build_context_string(context_chunks)
    prompt = template.replace("{{CONTEXT}}", context).replace("[context]", context)
    prompt = prompt.replace("{{QUERY}}", question).replace("[query]", question)
    return prompt


def _client() -> OpenAI:
    return OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """给定问题 + 检索片段，用 RAG 模板生成答案（system=渲染后的模板, user=问题）。"""
    client = _client()
    resp = client.chat.completions.create(
        model=config.EVAL_MODEL,
        messages=[
            {"role": "system", "content": render_rag_prompt(question, context_chunks)},
            {"role": "user", "content": question},
        ],
        temperature=0,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip()
