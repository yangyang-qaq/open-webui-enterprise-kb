"""Locust 入口：汇总 4 个 User 类。

运行（在 loadtest/ 目录下）：

    locust -f locustfile.py --host http://127.0.0.1:8080

单场景（只跑某个 User 类）：

    locust -f locustfile.py --host http://127.0.0.1:8080 \
        --users 100 --spawn-rate 10 --run-time 5m --headless ReadUser

阶梯爬坡找拐点（10 并发起步，每 60s +10，直到 100）：

    locust -f locustfile.py --host http://127.0.0.1:8080 \
        --headless --step-users 10 --step-time 60 ReadUser
"""

from scenarios.chat_user import ChatUser
from scenarios.mixed_user import MixedUser
from scenarios.read_user import ReadUser
from scenarios.upload_user import UploadUser
