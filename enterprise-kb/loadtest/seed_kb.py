"""压测前置数据 seeding（HTTP 层，需 OWUI 已启动且 mock 已就位）。

作用：
  1. 用 loadtest001 登录（admin）；
  2. 创建「读场景库 loadtest-read」与「上传场景库 loadtest-upload」；
  3. 往读场景库灌 10kb + 100kb 两个文件（触发真实分块 + 向量化，走 mock embedding）。

运行（backend venv，OWUI 与 mock 均已启动）：

    cd open-webui/backend
    ./.venv/Scripts/python.exe ../../loadtest/seed_kb.py

脚本末尾会打印 KB id / file_id，回填到 loadtest/config.py 或用环境变量注入。
"""

import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import API, BASE_URL, DATA_DIR, EMAIL_PREFIX, USER_PASSWORD  # noqa: E402

EMAIL = f"{EMAIL_PREFIX}001@example.com"
PASSWORD = USER_PASSWORD
READ_KB_NAME = "loadtest-read"
UPLOAD_KB_NAME = "loadtest-upload"


def main():
    with httpx.Client(base_url=BASE_URL, timeout=300.0) as c:
        r = c.post(f"{API}/auths/signin", json={"email": EMAIL, "password": PASSWORD})
        r.raise_for_status()
        token = r.json()["token"]
        h = {"Authorization": f"Bearer {token}"}

        r = c.post(
            f"{API}/knowledge/create",
            json={"name": READ_KB_NAME, "description": "压测读场景库"},
            headers=h,
        )
        r.raise_for_status()
        read_id = r.json()["id"]

        r = c.post(
            f"{API}/knowledge/create",
            json={"name": UPLOAD_KB_NAME, "description": "压测上传场景库"},
            headers=h,
        )
        r.raise_for_status()
        upload_id = r.json()["id"]

        first_file_id = None
        for fname in ("10kb.txt", "100kb.txt"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "rb") as f:
                up = c.post(f"{API}/files/", files={"file": (fname, f, "text/plain")}, headers=h)
                up.raise_for_status()
                file_id = up.json()["id"]
                if first_file_id is None:
                    first_file_id = file_id
                add = c.post(f"{API}/knowledge/{read_id}/file/add", json={"file_id": file_id}, headers=h)
                add.raise_for_status()
                print(f"added {fname} -> {file_id}")

    print("\n===== 回填到 loadtest/config.py（或 export 环境变量）=====")
    print(f"KB_ID_READ   = {read_id}")
    print(f"KB_ID_UPLOAD = {upload_id}")
    print(f"FILE_ID      = {first_file_id}")


if __name__ == "__main__":
    main()
