# RAG Faithfulness 离线评测

用「黄金问答集」测量 RAG 生成答案的 **Faithfulness（忠实度）**：生成答案中的每条主张是否真的被检索片段支持。接入 GitHub Actions 后，每个 PR 自动跑，低于门槛即失败。

## 目录

```
eval/
├── golden_set.json          # 黄金集（当前 40 条，来自 4 个真实技术文档，待人工核实后扩到 50~200）
├── golden_set.schema.json   # JSON Schema 校验
├── rag_template.txt         # 固化的 RAG 模板（含引用约束，需与 config.py 同步）
├── config.py                # 集中配置（env 可覆盖）
├── generate.py              # 渲染 RAG 模板 + DeepSeek 生成答案
├── faithfulness.py          # LLM-as-judge：拆主张 + 判定 + 打分
├── retrieve.py              # full 档：HTTP 调真实混合检索+重排
├── run_eval.py              # 主入口（--mode judge-only|full，门槛退出码）
├── bootstrap_golden.py      # 从问答对生成 golden_set 骨架
├── gen_questions.py         # 从文档自动生成问答草稿（人工核实后入库）
└── requirements.txt
```

## 两档模式

| 档 | 片段来源 | 依赖 | 触发 |
|---|---|---|---|
| `judge-only`（轻量） | `golden_set.json` 里预存的 `context_chunks` | 仅 DeepSeek key | 每个 PR |
| `full`（完整） | 真实「混合检索+重排」（HTTP 调 `/evaluate/query`） | 后端 + 模型 + 登录账号 | nightly / 手动 |

两档共用 `generate.py`（生成）+ `faithfulness.py`（判定），差异只在片段来源。

## Faithfulness 算法

1. 用 RAG 模板（含「严格基于上下文回答，不足则拒答」约束）生成答案（`temperature=0`）。
2. LLM-as-judge 把答案拆成原子主张，逐条判 `supported ∈ {yes,no,insufficient}`。
3. `faithfulness = #yes / #claims`；拒答（空主张）视为 1.0。

## 本地运行

```bash
# 安装依赖（建议复用 backend venv，或新建轻量 venv）
pip install -r requirements.txt

# 1) judge-only 档（需 DEEPSEEK_API_KEY）
export DEEPSEEK_API_KEY=sk-...
python run_eval.py --mode judge-only --limit 5

# 2) full 档（需后端在 127.0.0.1:8080 跑 + 登录账号 + KB id）
export EVAL_USER_EMAIL=eval@example.com
export EVAL_USER_PASSWORD=xxx
export EVAL_KB_ID=<your-kb-id>
python run_eval.py --mode full --limit 5 --save-context   # --save-context 回填 context_chunks
```

报告输出到 `eval/reports/report-<时间戳>.json` / `.md`。

## 黄金集整理流程

1. 拿到原始文档内容：可从知识库导出（见 `gen_questions.py` 文件头示例）或直接写 `{文件名: 全文}` 的 JSON。
2. 生成草稿（二选一）：
   - 自动：`python gen_questions.py --docs docs.json --kb-id <ID> --per-doc 10`（从文档提取问答对）；
   - 手动：`python bootstrap_golden.py --input my_qa.json --kb-id <ID>`（从已有问答对生成骨架）。
3. 逐条人工核实 `golden_answer` 正确、且在 KB 内**可答**（否则忠实度无意义）。
4. 跑 `run_eval.py --mode full --save-context` 一次性回填真实检索片段（`context_chunks`）。
5. 提交 `golden_set.json`。之后 PR 走 judge-only 档即用这些预存片段。

> 当前 `golden_set.json` 的 40 条来自知识库「新的test1」的 4 个真实技术文档（AI 医疗 / 微服务 / Python 后端 / LLM 综述），由 DeepSeek 依据原文生成、`source_file` 已标注出处。**上线前务必逐条人工核实 `golden_answer` 与原文一致**，再据此扩到 50~200 条。

## 模板同步（重要）

`rag_template.txt` 是 `open-webui/backend/open_webui/config.py` 里 `DEFAULT_RAG_TEMPLATE` 的固话副本。若后者改动，须同步更新这里，否则 judge-only 档测的不是生产行为。生产环境可用 `RAG_TEMPLATE` env 覆盖（`config.py:load_rag_template` 优先读 env）。

## CI 门禁

- `rag-eval.yml`：PR / push master → judge-only 档，低于 `EVAL_THRESHOLD`（默认 0.8）`exit 1` 使 build 失败。
- `rag-eval-full.yml`：`schedule`（每天 03:17 UTC）+ `workflow_dispatch` → full 档（自托管 runner，需后端）。

**仓库需配置的 Secrets**（Settings → Secrets and variables → Actions）：

| Name | 档位 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 两档 | DeepSeek 密钥（生成 + judge 共用） |
| `EVAL_USER_EMAIL` | full | 评测专用登录账号邮箱 |
| `EVAL_USER_PASSWORD` | full | 该账号密码 |

**仓库需配置的 Variables**（可选，均有默认值）：

| Name | 默认 | 说明 |
|---|---|---|
| `EVAL_THRESHOLD` | `0.8` | 聚合 Faithfulness 门槛 |
| `EVAL_KB_ID` | — | full 档目标知识库 id |

> 建议单独建一个只读的 `eval` 账号跑 full 档，避免频繁 signin 触发限流（15 次/180s/email）。
