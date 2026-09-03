"""压测集中配置：目标地址、预置 KB id、100 个测试账号、上传文件。

所有值都可用环境变量覆盖；KB id / file_id 在 seeding 后回填（或运行时注入）。
"""

import os

# 被测系统（多 worker OWUI）
BASE_URL = os.getenv("OWUI_BASE_URL", "http://127.0.0.1:8080")
API = "/api/v1"

# 预置知识库 id：读场景用已灌数据的库，上传场景用留空的库
# seeding 后用真实 id 覆盖（或 export KB_ID_READ=... 等）
KB_ID_READ = os.getenv("KB_ID_READ", "7d6943b9-b3fe-471d-a14e-efd28a2284a9")
KB_ID_UPLOAD = os.getenv("KB_ID_UPLOAD", "d5dd21f4-4a8f-44e2-aba1-c84d67b3483f")

# 读场景「分块列表」接口需要一个具体 file_id（属于 KB_ID_READ）
FILE_ID = os.getenv("FILE_ID", "bd6cd2cc-a9c6-4566-8de7-5adfd1e3462a")

# 聊天场景使用的模型名：必须与 OWUI 从 mock 注册的模型 id 一致
# （mock /v1/models 返回 MOCK_MODEL，默认 "mock-chat"；可用 GET /api/v1/models 核对）
CHAT_MODEL = os.getenv("CHAT_MODEL", "mock-chat")

# 100 个测试账号（seed.py 生成），Locust 虚拟用户 on_start 随机选一个登录
USER_COUNT = int(os.getenv("USER_COUNT", "100"))
EMAIL_PREFIX = os.getenv("EMAIL_PREFIX", "loadtest")
USER_PASSWORD = os.getenv("USER_PASSWORD", "LoadTest@123")


def credentials() -> list[tuple[str, str]]:
    return [(f"{EMAIL_PREFIX}{i:03d}@example.com", USER_PASSWORD) for i in range(1, USER_COUNT + 1)]


# 上传测试文件（相对 loadtest/data/，由 seed.py 的 gen_test_files 生成）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UPLOAD_FILES = [os.path.join(DATA_DIR, f) for f in ("10kb.txt", "100kb.txt", "1mb.txt")]
