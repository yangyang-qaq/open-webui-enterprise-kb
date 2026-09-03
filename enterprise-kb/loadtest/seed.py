"""压测前置数据 seeding（DB 层，无需 OWUI 运行）。

作用：
  1. 插入 100 个 admin 测试用户（email 各不相同，绕开 signin 限流 15 次/180s）；
  2. 生成上传测试文件（10kb / 100kb / 1mb）。

运行（用 backend venv 的 python，.env 已切 PostgreSQL）：

    cd open-webui/backend
    ./.venv/Scripts/python.exe ../../loadtest/seed.py

说明：
  - 必须在 import open_webui 之前显式加载 open-webui/.env，确保 DATABASE_URL 已切 PG；
  - 密码统一 bcrypt 哈希后写入 auth 表，role='admin' 以绕开 check_model_access。
"""

import argparse
import asyncio
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# 1. 显式加载 open-webui/.env（必须在 import open_webui 之前）
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(HERE, "..", "open-webui", ".env"), override=True)

# 2. 把 backend 加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "open-webui", "backend")))

USER_COUNT = 100
EMAIL_PREFIX = "loadtest"
PASSWORD = "LoadTest@123"
DATA_DIR = os.path.join(HERE, "data")


def gen_test_files():
    """生成不同大小的纯文本上传文件，覆盖不同分块/向量化耗时。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    unit = "这是一个用于压测的中文句子，包含足够的词汇以测试分块与向量化的性能表现。\n"
    unit_len = len(unit.encode("utf-8"))
    for name, size in (("10kb.txt", 10 * 1024), ("100kb.txt", 100 * 1024), ("1mb.txt", 1024 * 1024)):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(unit * (size // unit_len + 1))
        print(f"gen: {path} ({os.path.getsize(path)} bytes)")


async def seed_users():
    # 延后 import：只在真正需要写库时才初始化 open_webui（连 PG）
    from open_webui.models.auths import Auths
    from open_webui.models.users import Users
    from open_webui.utils.auth import get_password_hash

    hashed = await get_password_hash(PASSWORD)
    created = skipped = 0
    for i in range(1, USER_COUNT + 1):
        email = f"{EMAIL_PREFIX}{i:03d}@example.com"
        if await Users.get_user_by_email(email):
            skipped += 1
            continue
        user = await Auths.insert_new_auth(
            email=email,
            password=hashed,
            name=f"LoadTest{i:03d}",
            role="admin",
        )
        if user:
            created += 1
            print(f"created: {email} ({user.id})")
        else:
            print(f"FAILED: {email}")
    print(f"\nusers: created={created}, skipped(existing)={skipped}, target={USER_COUNT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-files", action="store_true", help="只生成上传测试文件")
    args = parser.parse_args()

    gen_test_files()
    if not args.gen_files:
        # Windows 下 psycopg 异步模式需要 SelectorEventLoop（不能用默认 ProactorEventLoop）
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(seed_users())
