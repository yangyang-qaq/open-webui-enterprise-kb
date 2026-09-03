"""场景① 知识库读接口：列表 / 搜索 / 详情 / 分块列表，只读不写。

任务权重：列表 4 > 搜索 3 > 详情 2 > 分块 1，贴近「浏览为主」的真实使用。
"""

from locust import between, task

from auth import AuthedUser
from config import API, FILE_ID, KB_ID_READ


class ReadUser(AuthedUser):
    wait_time = between(1, 2)

    @task(4)
    def list_kb(self):
        self.client.get(f"{API}/knowledge/", params={"page": 1}, headers=self.headers, name="KB.list")

    @task(3)
    def search(self):
        # 注意：search 是 GET，query 走 query 参数
        self.client.get(
            f"{API}/knowledge/search",
            params={"query": "压测关键词"},
            headers=self.headers,
            name="KB.search",
        )

    @task(2)
    def kb_detail(self):
        self.client.get(f"{API}/knowledge/{KB_ID_READ}", headers=self.headers, name="KB.detail")

    @task(1)
    def chunks(self):
        self.client.get(
            f"{API}/knowledge/{KB_ID_READ}/files/{FILE_ID}/chunks",
            headers=self.headers,
            name="KB.chunks",
        )
