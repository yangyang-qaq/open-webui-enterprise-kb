# 企业级知识库管理后台 — Enterprise Knowledge Base Dashboard

> 基于 [Open WebUI](https://github.com/open-webui/open-webui)（128K+ Stars）二次开发
>
> **全 11 个 Phase 已完成** | 33+ 个 API | 7 张数据表 | 7 种分块策略 | 5 种 Agent 角色（编排式 + 自主式 LangChain Agent） | 32 个技术问题已解决

---

## 项目简介

在 Open WebUI 现有 RAG 能力之上，独立设计并实现了企业级知识库管理仪表板，覆盖 RAG 全链路可观测性、质量管理和 Agent 协作编排：

| 模块 | 功能 | 说明 |
|---|---|---|
| 🧩 **分块管理** | 预览 / 合并 / 拆分 / 重建向量 | 可视化 RAG 分块结果，手动修正提升检索准确率 |
| ⏳ **进度监控** | SSE 实时推送 + 轮询兜底 | 文档处理不再黑盒，双通道保证可靠性 |
| 📊 **检索评估** | 查询 → Top-K → 标注 → Recall/Precision/MRR | 把 RAG 从黑盒变成可度量系统 |
| 📸 **版本管理** | 快照 / 回滚 / 差异对比 | 知识库版本可控可回溯 |
| 📝 **Prompt 配置** | 模板编辑 / 变量替换 / 聊天管道注入 | Prompt Engineering 融入 RAG 管道 |
| 🔪 **多策略分块** | 7 种算法 + jieba 关键词 + 问题生成 | 按文档类型自适应分块 |
| 🤖 **Agent 工作流** | 多 Agent 协作 / 真实 LLM 调用 / Word 报告下载 | 检索→分析→汇报→校验，全链路可编排 |
| ⚙️ **自主式 Agent** | LangGraph function-calling，5 角色即工具 | 模型自主决定调用顺序与轮数，与编排式一键切换，SSE 逐步轨迹 + 结论 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────┐
│              浏览器 (SvelteKit + TypeScript)           │
│  6-Tab 导航: Files │ Chunks │ Processing │ Evaluate   │
│              Snapshots │ Agents                       │
├──────────────────────────────────────────────────────┤
│                 FastAPI 异步后端                        │
│  Knowledge Router │ Retrieval Router │ Chat Middleware │
│  33 个新增端点      │ 7 种分块策略       │ Prompt 模板注入 │
│        │                    │                         │
│   ┌────┴────┐  ┌──────────┐  ┌──────────────────┐   │
│   │ SQLite  │  │ ChromaDB │  │  DeepSeek API     │   │
│   │(7 张新表)│  │ (向量存储) │  │  (LLM 推理)       │   │
│   └─────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 技术栈

| 层级 | 技术 |
|---|---|
| **后端框架** | Python / FastAPI（异步 ASGI） |
| **前端框架** | SvelteKit + TypeScript + Tailwind CSS |
| **数据库** | SQLite（开发）/ PostgreSQL（生产），SQLAlchemy ORM + Alembic 迁移 |
| **向量数据库** | ChromaDB |
| **文档处理** | LangChain Text Splitters + 7 种自定义分块策略 |
| **实时通信** | SSE（Server-Sent Events）+ 轮询双通道 |
| **AI / LLM** | DeepSeek API / Prompt Engineering / RAG / Agent Workflow |
| **文档生成** | python-docx（Word 报告导出） |

---

## 功能详解

### 🧩 分块管理（Phase 1, 5-6）
- 上传文档后自动展示分块结果，不写入向量库
- 支持合并相邻分块、拆分单个分块
- 调整后一键重建向量索引（Reindex）
- **7 种分块策略**：Naive / General(段落) / Book(章节) / Paper(论文) / Resume(简历) / Table(CSV表格) / QA(问答对)
- jieba 关键词提取 + 问题自动生成

### ⏳ 进度监控（Phase 2）
- SSE 实时推送文件处理状态（pending → chunking → embedding → completed）
- 前端双通道：SSE 主通道 + 3 秒轮询兜底
- 批量任务进度汇总

### 📊 检索评估（Phase 3）
- 输入测试查询 → 返回 Top-K 检索结果及相似度分数
- 人工标注相关/不相关
- 自动计算 **Recall@K / Precision@K / MRR**
- 标注累积后评估不同配置效果

### 📸 版本管理（Phase 4）
- 创建知识库快照（保存文件关联 + 分块元数据）
- 回滚到指定快照
- **Unlink Only** 按钮（`delete_file=false`）：取消文件关联但不物理删除
- 两个快照差异对比

### 📝 Prompt 模板配置（Phase 5）
- 知识库级别的 RAG Prompt 模板
- 支持变量替换：`{query}` / `{context}` / `{kb_name}`
- 已接入聊天管道中间件，自动注入到 system message

### 🤖 Agent 工作流（Phase 7-8）
- **5 种角色预设**：🔍检索员 / 🧠分析员 / 📝汇报员 / ✅校验员 / 🌐翻译员
- 可视化多步工作流创建（拖拽式步骤编排）
- **真实 LLM 调用**：检索员调 ChromaDB → 分析员/汇报员/校验员调 DeepSeek API
- SSE 流式执行反馈，步骤间变量传递
- **检索增强**：k=15 检索 + 按文档分组排序合并（解决 chunk 碎片化）
- **📥 Word 报告下载**：执行完成后一键下载 .docx 报告（含完整输出）
- **⚙️ 自主式 LangChain Agent（最新）**：改用 `langchain.agents.create_agent`（LangGraph 栈）把 5 个角色做成工具，模型自主决定调用顺序/轮数（检索为空会自动换词重试）；`POST /knowledge/_agents/autonomous/exec` SSE 推送每轮 `round/observation` 轨迹与最终结论；前端 Agents Tab 右上角与编排式一键切换（`AgentsPanel` / `AutonomousAgentPanel`）

---

## 数据库设计（7 张新表）

| 表名 | 用途 | 关键字段 |
|---|---|---|
| `knowledge_chunk` | 分块记录 | file_id, chunk_index, content, token_count, content_hash, meta(JSON) |
| `knowledge_processing_task` | 单文件处理状态 | file_id, status, progress, error_message |
| `knowledge_batch_task` | 批量任务汇总 | file_count, completed_count, failed_count |
| `knowledge_relevance_judgment` | 检索相关性标注 | query_text, chunk_id, relevance(0/1), rank_position |
| `knowledge_snapshot` | 版本快照 | label, file_count, snapshot_data(JSON) |
| `agent_workflow` | 工作流定义 | name, description, user_id |
| `agent_workflow_step` | 工作流步骤 | workflow_id, order_index, agent_role, knowledge_id, prompt_template, output_var |

---

## API 端点（33 个）

### 分块管理
| Method | Path | Description |
|---|---|---|
| `POST` | `/{id}/chunks/preview` | 预览分块结果（支持 method 参数选策略） |
| `GET` | `/{id}/files/{fileId}/chunks` | 查看某文件所有分块 |
| `POST` | `/{id}/chunks/merge` | 合并相邻分块（+10000 偏移解决 UNIQUE 冲突） |
| `POST` | `/{id}/chunks/split` | 拆分分块 |
| `POST` | `/{id}/chunks/reindex` | 重建向量索引 |

### 进度监控
| Method | Path | Description |
|---|---|---|
| `GET` | `/{id}/progress` | 获取所有文件处理状态 |
| `GET` | `/{id}/progress/stream` | SSE 实时推送 |
| `GET` | `/{id}/progress/batch` | 批量任务进度 |

### 检索评估
| Method | Path | Description |
|---|---|---|
| `POST` | `/{id}/evaluate/query` | 执行测试查询，返回 Top-K + 指标 |
| `POST` | `/{id}/evaluate/annotate` | 标注相关/不相关 |
| `GET` | `/{id}/evaluate/judgments` | 查看标注列表 |
| `DELETE` | `/{id}/evaluate/judgments/{q}` | 删除标注 |

### 快照管理
| Method | Path | Description |
|---|---|---|
| `POST` | `/{id}/snapshots` | 创建快照 |
| `GET` | `/{id}/snapshots` | 快照列表 |
| `POST` | `/{id}/snapshots/{sid}/rollback` | 回滚 |
| `POST` | `/{id}/snapshots/compare` | 差异对比 |
| `DELETE` | `/{id}/snapshots/{sid}` | 删除快照 |

### Prompt 配置
| Method | Path | Description |
|---|---|---|
| `GET` | `/{id}/prompt` | 获取 RAG Prompt 模板 |
| `PATCH` | `/{id}/prompt` | 更新模板 |

### Agent 工作流
| Method | Path | Description |
|---|---|---|
| `GET` | `/_workflows` | 列出所有工作流 |
| `POST` | `/_workflows` | 创建工作流 |
| `GET` | `/_workflows/roles` | 获取 Agent 角色列表 |
| `POST` | `/_workflows/exec` | 执行工作流（SSE 流式返回） |
| `POST` | `/_workflows/exec/download` | 下载 Word 执行报告 |
| `DELETE` | `/_workflows/{wfid}` | 删除工作流 |

### 自主式 Agent（新增）
| Method | Path | Description |
|---|---|---|
| `POST` | `/_agents/autonomous/exec` | 自主式 LangChain Agent 执行（SSE 流式：round/observation/answer/done） |

---

## 关键设计决策

1. **SQLite UNIQUE 约束冲突**：merge/split 时用 `+10000` 偏移 + 重新编号解决 chunk_index 中间态冲突（SQLite 无 DEFERRABLE）
2. **双通道可靠性**：SSE 实时推送 + 3 秒轮询兜底
3. **快照设计**：存 file_id 引用不存文件内容，配合 Unlink Only（`delete_file=false`）确保可回滚
4. **ChromaDB chunk 碎片化**：k=15 检索 + 按 file_id 分组 + chunk_index 排序合并，提供连贯原文
5. **Agent 全链路上下文**：固定以「用户问题」开头 + 前序步骤输出 + 任务描述，确保每个 Agent 有完整信息
6. **Prompt 闭环**：KB 模板 → 变量替换 → 聊天管道注入，完整链路打通
7. **路由顺序**：`/_workflows` 必须在 `/{id}` 之前注册，避免 FastAPI 路径匹配陷阱

---

## 踩坑精华

| 问题 | 解决方案 |
|---|---|
| SQLite UNIQUE 约束中间态冲突 | +10000 偏移 + 重新编号 |
| `/{id}` 先于 `/_workflows` 匹配 | 路由顺序调整 |
| Agent 步骤间变量不传递 | 自动收集前序输出注入 prompt |
| 校验员反馈检索结果不完整 | k=15 + 按文档分组合并 chunks + 全链路提权截断 |
| 分析员看不到用户问题 | `vn != "query"` 过滤条件修正 |
| Alembic 多头冲突 | 更新 down_revision 指向合并版本 |
| chunk_id 重复导致 Svelte each_key_duplicate | `result-{i}-{hash[:8]}` 组合唯一 ID |

> 详见 [问题记录.md](问题记录.md)（24 个问题完整记录）

---

## 快速启动

```bash
# 1. 合并代码到 Open WebUI 项目
# backend/  → open-webui/backend/open_webui/
# frontend/ → open-webui/src/

# 2. 安装依赖并构建前端
cd open-webui
npm install --engine-strict=false
npm run build

# 3. 启动后端（国内需 HuggingFace 镜像）
cd backend
HF_ENDPOINT="https://hf-mirror.com" \
WEBUI_SECRET_KEY="your-secret-key" \
python -m uvicorn open_webui.main:app --host 127.0.0.1 --port 8080

# 4. 浏览器打开 http://127.0.0.1:8080
```

---

## 开发统计

| 指标 | 数据 |
|---|---|
| 新增数据库表 | 7 张 |
| 新增 API 端点 | 33 个 |
| 分块策略 | 7 种 |
| Agent 角色预设 | 5 种 |
| 新增前端页面 | 6 个 Tab + 9 个组件 |
| 排查解决问题 | 24 个 |
| 上游 PR | 2 个 |
| 开发周期 | 2026-07-20 ~ 2026-07-31 |

---

## 上游贡献

| PR | 内容 |
|---|---|
| [#27222](https://github.com/open-webui/open-webui/pull/27222) | fix: knowledge_fs grep splits on literal backslash-n |
| [#27249](https://github.com/open-webui/open-webui/pull/27249) | fix: mutable default argument in generate_function_chat_completion |

---

## 文档

- [开发文档.md](开发文档.md) — 详细技术实现与架构设计
- [问题记录.md](问题记录.md) — 24 个技术问题的排查与解决
- [项目总结.md](项目总结.md) — 功能模块总结与简历描述
