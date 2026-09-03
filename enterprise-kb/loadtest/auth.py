"""登录态复用：每个虚拟用户在 on_start 登录一次，缓存 Bearer token。

为什么只登录一次：
- signin 接口有限流（15 次/180s/email），反复登录会触发 429；
- 登录本身有 bcrypt + 写 DB 开销，不属于目标场景，应排除在压测指标之外。
"""

import random

from locust import HttpUser

from config import API, credentials

CREDENTIALS = credentials()


class AuthedUser(HttpUser):
    """带登录态基类。子类只需定义 @task，请求时带上 self.headers。"""

    abstract = True

    def on_start(self):
        email, password = random.choice(CREDENTIALS)
        with self.client.post(
            f"{API}/auths/signin",
            json={"email": email, "password": password},
            name="auth.signin",
            catch_response=True,
        ) as r:
            if r.status_code == 200:
                self.token = r.json().get("token", "")
            else:
                self.token = ""
                # catch_response=True 才能调用 r.failure()，否则会抛 LocustError
                r.failure(f"signin failed: {r.status_code} {r.text[:200]}")
        self.headers = {"Authorization": f"Bearer {self.token}"}
