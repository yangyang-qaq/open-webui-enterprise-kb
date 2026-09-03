"""场景② 聊天对话（SSE 流式）。

wait_time 必须 2~5s：SSE 是长连接，若间隔过短，用户会立刻重开新流，
等效并发远超 100，把结论做假。
"""

from locust import between, task

from auth import AuthedUser
from config import API, CHAT_MODEL
from sse import stream_post


class ChatUser(AuthedUser):
    wait_time = between(2, 5)

    @task
    def chat(self):
        body = {
            "model": CHAT_MODEL,
            "messages": [
                {"role": "user", "content": "请基于知识库总结一下文档的核心内容。"}
            ],
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
