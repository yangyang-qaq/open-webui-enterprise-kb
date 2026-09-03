# 知识库管理后台 — 100 并发压测体系

对 Open WebUI 二次开发的知识库管理后台做压力测试，模拟 100 人同时使用，定位瓶颈并给出容量结论。

- **压测工具**：Locust
- **外部 AI 调用**：本地 Mock（FastAPI），聚焦系统自身（鉴权 / DB / Chroma / RAG）瓶颈
- **压测环境**：模拟生产 = PostgreSQL + 多 worker uvicorn

```
Locust ──► OWUI（4 worker，PostgreSQL） ──► Mock Server（127.0.0.1:9099）
              │                                  └─ /v1/models、/v1/chat/completions、/v1/embeddings
              └─► PostgreSQL（docker，127.0.0.1:5433）
              └─► ChromaDB（本地持久化）
```

## 目录结构

```
loadtest/
├── locustfile.py          # Locust 入口，汇总 4 个 User 类
├── config.py              # 目标地址 / KB id / 100 账号 / 上传文件
├── auth.py                # 登录 + token 缓存（on_start 登录一次）
├── sse.py                 # SSE 流式读取 + TTFB 单独上报
├── scenarios/
│   ├── read_user.py       # ① 读接口
│   ├── chat_user.py       # ② 聊天 SSE
│   ├── upload_user.py     # ③ 上传 + 向量化
│   └── mixed_user.py      # ④ 混合全链路
├── mock/llm_mock.py       # Mock LLM/Embedding Server
├── seed.py                # DB 层 seeding：100 用户 + 测试文件
├── seed_kb.py             # HTTP 层 seeding：建库 + 灌文件
├── requirements.txt       # locust
└── data/                  # 上传测试文件（seed.py 生成）
```

## 0. 前置条件

- Docker（跑 PostgreSQL）
- backend venv 已就绪（`open-webui/backend/.venv`），含 fastapi/uvicorn/httpx/asyncpg
- 安装 locust：`cd open-webui/backend && ./.venv/Scripts/python.exe -m pip install -r ../../loadtest/requirements.txt`

## 1. 部署 PostgreSQL

```bash
docker run -d --name owui-pg \
  -e POSTGRES_USER=owui -e POSTGRES_PASSWORD=owui_pwd -e POSTGRES_DB=owui \
  -p 5433:5432 postgres:16-alpine \
  -c max_connections=200 -c shared_buffers=512MB -c log_min_duration_statement=250
```

`log_min_duration_statement=250` 会自动记录 >250ms 的慢查询，是定位瓶颈的关键。

## 2. 改 `.env` 切 PostgreSQL + 指向 mock

编辑 `open-webui/.env`：

```ini
DATABASE_URL=postgresql://owui:owui_pwd@127.0.0.1:5433/owui   # psycopg v3，不是 asyncpg
WEBUI_SECRET_KEY=<固定随机字符串，压测全程不变>   # 多 worker 解码 JWT 一致的关键
OPENAI_API_BASE_URL=http://127.0.0.1:9099/v1      # 指向 mock
OPENAI_API_KEY=mock-key
RAG_EMBEDDING_ENGINE=openai                        # embedding 走外部(mock)
RAG_EMBEDDING_MODEL=mock-embedding
RAG_OPENAI_API_BASE_URL=http://127.0.0.1:9099/v1   # 关键：config.py 会把 OPENAI_API_BASE_URL
RAG_OPENAI_API_KEY=mock-key                        # 强制重置为 api.openai.com，必须用
                                                   # RAG_OPENAI_* 显式覆盖，否则 file/add 连不上
```

> ⚠️ `WEBUI_SECRET_KEY` 必须固定且全程不变，否则多 worker 跨进程解码 JWT 失败会随机 401。
> ⚠️ `DATABASE_URL` 必须用 `postgresql://`（psycopg v3 异步 + psycopg2 同步都认），`postgresql+asyncpg://` 会报 MissingGreenlet。

## 3. 起 Mock Server

```bash
cd open-webui/backend
./.venv/Scripts/python.exe ../../loadtest/mock/llm_mock.py   # 监听 127.0.0.1:9099
# 校验：curl http://127.0.0.1:9099/v1/models  →  返回 mock-chat
```

## 4. 多 worker 起 OWUI

```bash
cd open-webui/backend
UVICORN_WORKERS=4 uvicorn open_webui.main:app --host 0.0.0.0 --port 8080 --workers 4 --ws auto
# 校验：curl http://127.0.0.1:8080/health 与 /health/db 均 200
```

## 5. Seeding

```bash
cd open-webui/backend

# 5.1 插入 100 用户 + 生成测试文件（DB 层，无需 OWUI）
./.venv/Scripts/python.exe ../../loadtest/seed.py

# 5.2 建库 + 灌文件（HTTP 层，需 OWUI + mock 已启动）
./.venv/Scripts/python.exe ../../loadtest/seed_kb.py
```

把 `seed_kb.py` 末尾打印的 `KB_ID_READ / KB_ID_UPLOAD / FILE_ID` 回填到 `loadtest/config.py`（或 `export KB_ID_READ=...`）。

## 6. 跑压测

```bash
cd loadtest

# 冒烟：10 并发 × 2 分钟，0 失败
locust -f locustfile.py --host http://127.0.0.1:8080 --users 10 --run-time 2m --headless ReadUser

# 基线：30 并发 × 3 分钟
locust -f locustfile.py --host http://127.0.0.1:8080 --users 30 --spawn-rate 10 --run-time 3m --headless ReadUser

# 阶梯爬坡找拐点：10 起步，每 60s +10，直到 100
locust -f locustfile.py --host http://127.0.0.1:8080 --headless --step-users 10 --step-time 60 ReadUser

# 保压：100 并发 × 5~10 分钟，输出 CSV 报告
locust -f locustfile.py --host http://127.0.0.1:8080 --users 100 --spawn-rate 20 --run-time 5m \
  --headless --csv=reports/read --csv-full-history ReadUser
```

4 个 User 类：`ReadUser` / `ChatUser` / `UploadUser` / `MixedUser`。上传场景（`UploadUser`）只在 5~10 并发下跑，分「真 MiniLM / mock embedding」两轮。

## 7. 指标与判定

| 关注指标 | 说明 |
|---|---|
| 分 endpoint RPS / P50 / P90 / P99 | Locust 默认统计 |
| 失败率 | 聊天无 502/流中断，上传 0 失败 |
| `SSE_TTFB` | 首字节时间（sse.py 单独上报，区别于流总时长） |
| PG 慢查询 | `docker logs owui-pg` 里 >250ms 语句 |
| 连接数 | `SELECT count(*) FROM pg_stat_activity;` |

**「100 并发达标」判定（保压窗口内同时满足）**：

- 失败率 < 1%
- 读接口 P95 < 300ms、P99 < 600ms
- 聊天 TTFB P95 < 500ms（10 并发基线校准后浮动）
- 上传（5~10 并发）`Files.upload` P95 < 1s
- PG 连接数 < 80% max、无 pool timeout、无持续 CPU 100%、爬坡无指数级上翘

## 8. 已知隐患观察点（压测重点复现）

1. **`_progress_store` 是进程内内存 dict**（`knowledge.py`）：多 worker 下 `file/add` 在 worker A 写进度，`GET /progress` 落到 worker B 就读不到 → 复现「同一 file_id 反复轮询返回空/跳变」。
2. **进度表读后写 race**：无唯一约束，并发写可能重复插入。
3. **ChromaDB PersistentClient 多进程写锁**：上传尾延迟高但 CPU 不高时指向它，生产化建议切远程 Chroma。
4. **`AgentWorkflowStep` 缺索引**：多 Agent 场景数据量上来后查询变慢。
5. **mock 自身成为瓶颈**：先用 Locust 直连 mock 打 200 并发验证其容量远高于被测系统。

## 9. 注意事项

- **多 worker 下 SQLite 不可用**：必须确认 `DATABASE_URL` 已切 PG，否则 `--workers 4` 直接把 SQLite 打挂。
- **聊天 `wait_time` 2~5s**：SSE 长连接，间隔过短会让等效并发远超 100。
- **signin 限流**：100 个不同 email 各登录一次，账号不足会误触发 429。
- **数据累积漂移**：上传场景开跑前重置上传库，避免 chunks 越积越多影响后续检索。
- **上传场景两个坑**：①`Files.upload` 必须带 `process_in_background=false`，否则内容提取在后台跑、紧跟的 `file/add` 读到空 content 报 400 EMPTY_CONTENT；②上传内容必须每次唯一（脚本用 uuid 后缀），否则同一 KB 内相同内容被去重判 DUPLICATE_CONTENT。
- **防打满本机**：任一进程 CPU >85% 即停止加并发并记录拐点。
