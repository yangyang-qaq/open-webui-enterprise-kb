# 基于 Open WebUI 的企业级知识库管理后台

> 在开源项目 [Open WebUI](https://github.com/open-webui/open-webui)（128K+ Star）的 FastAPI + SvelteKit 架构上进行二次开发，独立设计并实现了可视化的 RAG 知识库管理仪表板。
> **全 11 个 Phase 已完成 ✅** ｜ 开发周期 2026-07-20 ~ 2026-08-14 ｜ 独立排查解决 32 个技术问题 ｜ 向 Open WebUI 上游提交 2 个 bug fix PR

本仓库是**完整可运行的项目代码**（Open WebUI fork + 全部二次开发功能，默认分支即为知识库后台版本），同时收纳了二次开发的精华摘录、开发文档与评测/压测体系，做到**一个仓库完整代表整个项目**。

---

## 📁 仓库结构导览

```
仓库根目录（本 fork 完整代码，可运行）
├── backend/              # FastAPI 后端（二次开发代码已内嵌）
│   └── open_webui/
│       ├── routers/knowledge.py          # 知识库主路由（chunks/进度/评估/快照/工作流/自主式Agent）
│       ├── utils/chunking_strategies.py  # 7 种分块策略 + 关键词提取 + 问题生成
│       ├── utils/agent_autonomous.py     # 自主式 LangChain Agent（LangGraph create_agent）
│       └── models/knowledge.py           # KnowledgeChunk 等数据模型 + Agent 角色定义
├── src/                  # SvelteKit 前端
│   └── lib/components/workspace/Knowledge/
│       ├── ChunkManager / ProcessingDashboard / EvaluatePanel / SnapshotManager
│       └── AgentsPanel / AgentWorkflowEditor / AutonomousAgentPanel
├── enterprise-kb/        # 【精华层】二次开发摘录 + 文档 + 评测 + 压测（独立成册，便于快速理解）
│   ├── backend/ frontend/                # 只含新增/改动代码的摘录副本
│   ├── eval/                             # RAG Faithfulness 离线评测（40 条黄金集 + LLM-as-judge + 两档 CI）
│   ├── loadtest/                         # Locust 并发压测（Mock LLM/Embedding，隔离变量）
│   ├── 开发文档.md / 问题记录.md / 项目总结.md / DESIGN.md
│   └── read.md                           # 项目 README（面向展示）
└── README.md             # 本文件
```

> **阅读建议**：想快速理解「加了什么」→ 看 [enterprise-kb/](enterprise-kb/)；想看到处运行的全部代码 → 本仓库根目录的 `backend/`、`src/`。

---

## 项目简介

Open WebUI 是一个自托管的 AI 对话平台，支持接入 DeepSeek、Ollama 等大模型。本人在其既有 RAG 能力之上，独立实现了一套可视化知识库管理后台，核心模块：

| 模块 | 说明 |
|---|---|
| 🧩 **分块预览与手动调整** | 上传后自动分块展示，支持合并/拆分，调整后重建向量（`knowledge_chunk` 表） |
| ⏳ **向量化进度可视化** | 每个文件独立状态 + 进度条，SSE 实时推送 + 轮询兜底 |
| 📊 **检索质量评估面板** | 查询→Top-K→人工标注→自动算 recall@K / precision@K / MRR |
| 📸 **知识库版本管理** | 快照 / 回滚 / 快照间差异对比（Unlink Only 保留源文件） |
| 📝 **Prompt 模板配置** | 知识库级 RAG Prompt 模板，变量 `{query}/{context}/{kb_name}`，已接入聊天管道 |
| 🔪 **多策略分块引擎** | 7 种分块算法 + jieba 关键词提取 + 问题自动生成（纯函数、32 个 pytest 用例） |
| 🤖 **多 Agent 工作流编排** | Agent 角色预设（检索/分析/汇报/校验/翻译）、LLM 真实调用、SSE 流式、Word 报告下载与溯源 |
| ⚙️ **自主式 LangChain Agent** | 与编排式可切换：LangGraph function-calling Agent 自主决定调用哪些角色工具、几轮，SSE 展示逐步轨迹 + 结论 |
| 🎯 **Faithfulness 离线评测** | 40 条黄金集 + LLM-as-judge 原子主张判定 + CI 门禁（judge-only / full 两档） |
| 🔥 **并发压力测试** | Locust 100 并发 + Mock LLM/Embedding + 多 worker，定位 ChromaDB HNSW 写锁 |

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 后端 | Python / FastAPI（异步 ASGI）/ SQLAlchemy async ORM / Alembic |
| 前端 | SvelteKit / TypeScript / Tailwind CSS |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） |
| 向量库 | ChromaDB（Hybrid：BM25 + 向量加权 RRF，Cross-Encoder 重排 top3） |
| AI | DeepSeek API / LangChain / LangGraph / Prompt Engineering |
| 实时 | SSE (Server-Sent Events) |

---

## 快速开始

```bash
# 后端（open-webui 依赖需满足 requirements：langchain/langgraph 已升至新版 Agent 栈）
cd backend
uvicorn open_webui.main:app --port 8080 --host 0.0.0.0 --forwarded-allow-ips '*'

# 前端（另一终端）
npm install && npm run dev     # vite dev，端口 5173
```

访问 http://127.0.0.1:8080 → 知识库详情页可见 `Files│Chunks│Processing│Evaluate│Snapshots│Agents` 六个 Tab，Agents Tab 右上可切换「编排式 ⇄ 自主式」。

> 环境变量见 `backend/.env`（`OPENAI_API_KEY` 等，未纳入版本控制）。

---

## 关键实现与经验

- **分块合并/拆分撞 `UNIQUE(file_id, chunk_index)`**：中间态用「+10000 偏移停车区 + 插新块 + 重新编号」解决。
- **检索量 k=15 增强**：ChromaDB 单 chunk 平均 ~200 字符，k=3 太碎 → 召回扩到 k=15、按 `file_id` 分组 + `chunk_index` 排序拼成连贯文档。
- **SSE 双通道**：实时推送为主 + 3s 轮询兜底。
- **FastAPI 路由顺序**：下划线前缀具体路由（`_workflows`、`_agents`）置于 `GET /{id}` 参数路由之前。
- 完整排查记录见 [enterprise-kb/问题记录.md](enterprise-kb/问题记录.md)（32 个问题）、设计文档见 [enterprise-kb/开发文档.md](enterprise-kb/开发文档.md)。

---

## 上游贡献

| PR | 内容 |
|---|---|
| [#27222](https://github.com/open-webui/open-webui/pull/27222) | fix: knowledge_fs grep splits on literal backslash-n instead of newline |
| [#27249](https://github.com/open-webui/open-webui/pull/27249) | fix: mutable default argument in generate_function_chat_completion |

> 本仓库 fork 自 [open-webui/open-webui](https://github.com/open-webui/open-webui)，仅用于项目展示与二次开发，不用于对上游的提交。
