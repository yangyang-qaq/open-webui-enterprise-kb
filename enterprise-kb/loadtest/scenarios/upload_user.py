"""场景③ 文件上传 + 向量化（两段式）。

两段式（共享逻辑见 ../upload.py）：
  1. POST /api/v1/files/            multipart 上传，`process_in_background=false` 让
                                    「分块 + 向量化」在上传阶段同步完成
  2. POST /api/v1/knowledge/{id}/file/add   复用已建 file-{id} collection，写入 KB

wait_time 5~10s：写路径重（分块 + 嵌入 + 写 Chroma），单用户频率必须压低；
该场景只在小并发（5~10）下跑，不在 100 并发下跑。
"""

from locust import between, task

from auth import AuthedUser
from upload import upload_and_vectorize


class UploadUser(AuthedUser):
    wait_time = between(5, 10)

    @task
    def upload(self):
        upload_and_vectorize(self.client, self.headers)