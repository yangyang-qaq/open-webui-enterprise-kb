"""SSE 流式读取辅助：逐 chunk 消费，并把「首字节时间」TTFB 单独上报。

为什么要单独上报 TTFB：
Locust 对 `stream=True` 的请求，默认把「流结束总时长」记为响应时间；
对 SSE 长连接而言，这会把首 token 延迟和流式时长混在一起。压测需要分开看：

    Chat.completions      —— Locust 自动记录的流总时长
    SSE_TTFB              —— 手动上报的首字节时间（定位 RAG/鉴权/DB 慢的关键）
"""

import time


def stream_post(client, path, json_body, headers, name, environment):
    """POST 一个 SSE 接口，逐行消费流，手动上报 TTFB。

    返回 (status_code, ttfb_ms)。流总时长由 Locust 按 `with` 块退出自动记录。
    """
    start = time.perf_counter()
    ttfb = None

    with client.post(
        path,
        json=json_body,
        headers=headers,
        stream=True,
        catch_response=True,
        name=name,
    ) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if ttfb is None:
                ttfb = (time.perf_counter() - start) * 1000
                # 单独上报首字节时间
                environment.events.request.fire(
                    request_type="SSE_TTFB",
                    name=name,
                    response_time=ttfb,
                    response_length=0,
                    exception=None,
                    context={},
                )
            if line and line.startswith("data:") and line.strip() == "data: [DONE]":
                break

        # 若非 200，标记为失败，便于压测报告统计失败率
        if resp.status_code != 200:
            resp.failure(f"chat stream failed: {resp.status_code}")

        return resp.status_code, ttfb
