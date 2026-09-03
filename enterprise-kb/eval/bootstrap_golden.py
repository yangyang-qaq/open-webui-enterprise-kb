"""辅助人工整理黄金集：从一组 (question, golden_answer) 生成 golden_set.json 骨架。

context_chunks 留空，之后用 `run_eval.py --mode full --save-context` 回填真实检索片段。

输入格式（--input 指向的 JSON 文件）：
  [{"question": "...", "golden_answer": "...", "source_file": "..."}, ...]

用法：
  python bootstrap_golden.py --input my_qa.json --kb-id <KB_ID> [--output golden_set.json]
"""

import argparse
import json

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="从问答对生成 golden_set.json 骨架")
    parser.add_argument("--input", required=True, help="输入 JSON 文件：[{question, golden_answer, source_file?}]")
    parser.add_argument("--kb-id", default=None, help="知识库 id（缺省用 EVAL_KB_ID）")
    parser.add_argument("--output", default=config.GOLDEN_SET_PATH)
    args = parser.parse_args()

    kb_id = args.kb_id or config.DEFAULT_KB_ID

    with open(args.input, "r", encoding="utf-8") as f:
        qa = json.load(f)

    entries = []
    for i, item in enumerate(qa, start=1):
        entry = {
            "id": f"q{i:03d}",
            "question": item["question"],
            "golden_answer": item["golden_answer"],
            "kb_id": kb_id,
        }
        if item.get("source_file"):
            entry["source_file"] = item["source_file"]
        entries.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"已生成 {len(entries)} 条骨架 → {args.output}（context_chunks 待 full 档回填）")


if __name__ == "__main__":
    main()
