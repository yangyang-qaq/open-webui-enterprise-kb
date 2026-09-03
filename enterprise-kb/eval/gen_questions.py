"""从文档批量生成黄金集草稿（问答对），供人工核实后作为 golden_set.json。

用途：把知识库里的原始文档一键变成 50~200 条候选问答对，省去逐条手写。
生成的 golden_answer 严格基于原文，但**仍需逐条人工核实**（见 README「黄金集整理流程」）。

用法：
  python gen_questions.py --docs docs.json --kb-id <KB_ID> --per-doc 10

--docs 指向一个 JSON 文件，结构 { "<文件名>": "<文档全文>", ... }。
示例（从 SQLite 读 KB 文档导出）：
  python - <<'PY'
  import sqlite3, json
  db = sqlite3.connect("open-webui/backend/data/webui.db")
  rows = db.execute("SELECT f.filename, json_extract(f.data,'$.content') FROM knowledge_file kf JOIN file f ON f.id=kf.file_id WHERE kf.knowledge_id=?", ("<KB_ID>",)).fetchall()
  json.dump({fn: c for fn, c in rows if c}, open("docs.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
  PY
"""

import argparse
import json
import sys

from openai import OpenAI

import config


def generate_for_doc(client: OpenAI, title: str, content: str, per_doc: int) -> list[dict]:
    system = (
        "你是一名知识库评测集整理助手。你要从一个技术文档中提取事实性的问答对，"
        "用于 RAG 系统的忠实度（Faithfulness）评测。"
    )
    prompt = """请从下面这份文档中提取 {n} 条事实性问答对。要求：

1. 每条「question」是一个自然的中文提问，答案必须能在文档中**直接找到**（不要问文档没写的内容）。
2. 每条「golden_answer」是简洁、准确的中文陈述句，内容**逐字忠实于文档原文**，不得发挥、不得补充文档没有的信息。
3. 覆盖文档的关键事实点（数字、时间、专有名词、定义、流程、对比等），问题彼此不重复。
4. 输出**只包含**一个 JSON 数组，不要任何额外文字或 markdown 围栏，格式：
[{{"question":"...","golden_answer":"..."}}, ...]

文档标题：{title}
文档内容：
{content}
""".format(n=per_doc, title=title, content=content)

    resp = client.chat.completions.create(
        model=config.EVAL_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0,
        stream=False,
    )
    raw = (resp.choices[0].message.content or "").strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"未从输出解析到 JSON 数组: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="从文档批量生成黄金集问答草稿")
    parser.add_argument("--docs", required=True, help="JSON 文件：{文件名: 文档全文}")
    parser.add_argument("--kb-id", default=None, help="知识库 id（缺省用 EVAL_KB_ID）")
    parser.add_argument("--per-doc", type=int, default=10, help="每个文档提取 N 条（默认 10）")
    parser.add_argument("--output", default=config.GOLDEN_SET_PATH)
    args = parser.parse_args()

    kb_id = args.kb_id or config.DEFAULT_KB_ID
    with open(args.docs, "r", encoding="utf-8") as f:
        docs = json.load(f)

    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    entries = []
    idx = 0
    for title, content in docs.items():
        qa = generate_for_doc(client, title, content, args.per_doc)
        for item in qa:
            idx += 1
            entries.append(
                {
                    "id": f"q{idx:03d}",
                    "question": item["question"],
                    "golden_answer": item["golden_answer"],
                    "kb_id": kb_id,
                    "source_file": title,
                }
            )
        print(f"{title}: {len(qa)} 条", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"共 {len(entries)} 条 → {args.output}（context_chunks 待 full 档 --save-context 回填）", file=sys.stderr)


if __name__ == "__main__":
    main()
