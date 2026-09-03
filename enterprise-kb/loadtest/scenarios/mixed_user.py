"""场景④ 混合全链路：读 70% / 搜索 15% / 聊天 10% / 上传 5%。

按真实「查询为主」的负载配比，权重用 @task 的整数权重近似。
"""

from locust import between, task

from auth import AuthedUser
from config import API, CHAT_MODEL, KB_ID_READ
from sse import stream_post
from upload import upload_and_vectorize


class MixedUser(AuthedUser):
    wait_time = between(1, 3)

    @task(14)
    def list_kb(self):
        self.client.get(f"{API}/knowledge/", params={"page": 1}, headers=self.headers, name="KB.list")

    @task(3)
    def search(self):
        self.client.get(
            f"{API}/knowledge/search",
            params={"query": "压测关键词"},
            headers=self.headers,
            name="KB.search",
        )

    @task(2)
    def chat(self):
        body = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": "请基于知识库回答这个问题。"}],
            "stream": True,
        }
        stream_post(
            self.client,
            f"{API}/chat/completions",
            body,
            self.headers,
            name="Chat.completions",
            environment=self.environment,
        )

    @task(1)
    def upload(self):
        upload_and_vectorize(self.client, self.headers)
