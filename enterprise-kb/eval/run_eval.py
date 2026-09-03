"""主入口：跑黄金集 → 算聚合 faithfulness → 低于门槛则退出码非 0（CI 失败）。

用法：
  python run_eval.py --mode judge-only                 # PR 档：用 golden_set.json 预存 context_chunks
  python run_eval.py --mode full --save-context        # full 档：真实检索 + 回填 context_chunks
  python run_eval.py --mode judge-only --limit 5 --threshold 0.8
"""

import argparse
import json
import os
import sys
import time

import config
from generate import generate_answer
from faithfulness import judge_faithfulness


def load_golden_set(path: str | None = None) -> list[dict]:
    path = path or config.GOLDEN_SET_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _eval_one(entry: dict, chunks: list[dict]) -> dict:
    q = entry["question"]
    answer = generate_answer(q, chunks)
    verdict = judge_faithfulness(q, answer, chunks)
    return {
        "id": entry.get("id"),
        "question": q,
        "answer": answer,
        "faithfulness": verdict["faithfulness"],
        "claims": verdict.get("claims", []),
        "refusal": verdict.get("refusal", False),
    }


def run_judge_only(entries: list[dict], limit: int) -> list[dict]:
    results = []
    for entry in entries[:limit]:
        chunks = entry.get("context_chunks", [])
        r = _eval_one(entry, chunks)
        results.append(r)
        print(
            f"[{r['id']}] faithfulness={r['faithfulness']} "
            f"({'refusal' if r['refusal'] else str(len(r['claims'])) + ' claims'})"
        )
    return results


def run_full(entries: list[dict], limit: int, save_context: bool) -> list[dict]:
    from retrieve import signin, retrieve_chunks  # 延迟 import，judge-only 档无需后端

    token = signin()
    results = []
    for entry in entries[:limit]:
        kb_id = entry.get("kb_id") or config.DEFAULT_KB_ID
        if not kb_id:
            raise RuntimeError(f"条目 {entry.get('id')} 缺少 kb_id，且 EVAL_KB_ID 未设置")
        chunks = retrieve_chunks(entry["question"], kb_id, token)
        if save_context:
            entry["context_chunks"] = chunks
        r = _eval_one(entry, chunks)
        results.append(r)
        print(f"[{r['id']}] faithfulness={r['faithfulness']} (retrieved {len(chunks)} chunks)")

    if save_context:
        with open(config.GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"已回填 context_chunks 到 {config.GOLDEN_SET_PATH}")
    return results


def aggregate(results: list[dict]) -> float:
    scores = [r["faithfulness"] for r in results]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def write_report(results: list[dict], avg: float, report_dir: str) -> str:
    os.makedirs(report_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(report_dir, f"report-{stamp}.json")
    md_path = os.path.join(report_dir, f"report-{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate_faithfulness": avg, "results": results}, f, ensure_ascii=False, indent=2)

    lines = ["# RAG Faithfulness 报告", "", f"- 聚合 Faithfulness: **{avg:.4f}**", f"- 题目数: {len(results)}", ""]
    for r in results:
        status = "refusal" if r["refusal"] else f"{len(r['claims'])} claims"
        lines.append(f"- `{r['id']}`: **{r['faithfulness']}** ({status})")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Faithfulness 离线评测")
    parser.add_argument("--mode", choices=["judge-only", "full"], required=True)
    parser.add_argument("--threshold", type=float, default=None, help="覆盖 EVAL_THRESHOLD")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--save-context", action="store_true", help="full 档：回填 context_chunks 到 golden_set.json")
    parser.add_argument("--report-dir", default=config.REPORT_DIR)
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else config.THRESHOLD
    entries = load_golden_set()
    limit = args.limit if args.limit else len(entries)

    if args.mode == "judge-only":
        results = run_judge_only(entries, limit)
    else:
        results = run_full(entries, limit, args.save_context)

    avg = aggregate(results)
    report = write_report(results, avg, args.report_dir)
    print(f"\n报告: {report}")
    print(f"聚合 Faithfulness = {avg:.4f}（门槛 {threshold:.2f}）")

    if avg < threshold:
        print(f"FAIL: 聚合 Faithfulness {avg:.4f} 低于门槛 {threshold:.2f}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
