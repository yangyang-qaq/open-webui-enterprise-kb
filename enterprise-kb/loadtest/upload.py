"""上传 + 向量化两段式的共享逻辑，供 UploadUser / MixedUser 复用。

关键点（踩坑记录）：
  - Files.upload 必须带 `process_in_background=false`：否则内容提取在后台跑，
    紧跟的 file/add 会读到空 content 报 400 EMPTY_CONTENT；
  - 上传内容每次唯一（uuid 后缀）：否则同一 KB 内相同内容被去重判
    DUPLICATE_CONTENT 报 400，无法反映真实写路径。
"""

import io
import random
import uuid

from config import API, KB_ID_UPLOAD

# 三档大小（字节）：小/中/大文档，覆盖不同分块、向量化耗时
SIZE_TIERS = [10 * 1024, 100 * 1024, 1024 * 1024]
BASE_UNIT = "这是一个用于压测的中文句子，包含足够的词汇以测试分块与向量化的性能表现。\n"


def unique_content(size: int) -> str:
    """生成指定大小的唯一文本，避免同内容触发 DUPLICATE_CONTENT 去重。"""
    unit_len = len(BASE_UNIT.encode("utf-8"))
    n = max(1, size // unit_len + 1)
    return BASE_UNIT * n + f"\n文档唯一标识 {uuid.uuid4().hex}\n"


def upload_and_vectorize(client, headers) -> None:
    """两段式：上传（同步分块+向量化）→ 加入知识库。"""
    size = random.choice(SIZE_TIERS)
    content = unique_content(size)
    fname = f"upload-{uuid.uuid4().hex[:8]}.txt"
    with io.BytesIO(content.encode("utf-8")) as f:
        up = client.post(
            f"{API}/files/",
            params={"process_in_background": "false"},
            files={"file": (fname, f, "text/plain")},
            headers=headers,
            name="Files.upload",
        )
    if up.status_code != 200:
        return
    file_id = up.json().get("id", "")
    if not file_id:
        return
    client.post(
        f"{API}/knowledge/{KB_ID_UPLOAD}/file/add",
        json={"file_id": file_id},
        headers=headers,
        name="KB.file_add",
    )