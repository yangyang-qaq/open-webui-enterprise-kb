"""自主式（LangChain Agent）执行引擎。

与编排式 exec_workflow 相对：编排式把 5 个角色按用户排好的顺序串行跑；
这里用真实的 LangChain function-calling Agent（langchain.agents.create_agent，
底层是 langgraph 编译图），把 5 个角色包装成 5 个工具，由模型自主决定调哪个、调几轮。

设计约束（与 routers/knowledge.py 的 exec_workflow 保持一致）：
- LLM 走 OpenAI 兼容 /chat/completions（DeepSeek），base_url/api_key 从 .env 直读；
- 检索复用 retrieval.utils.query_collection，检索文本的整理格式与 exec_workflow 逐字一致；
- 每个 LLM 角色工具的内部指令复用 models.knowledge.AGENT_ROLES[role].default_prompt；
- 模块顶层不做任何 langchain/langchain_openai 的 import（避免依赖异常拖垮整个 app 启动），
  全部在函数内惰性导入。
"""

import json
import os
from collections import OrderedDict


def _resolve_api() -> tuple[str, str, str]:
    """返回 (base_url, api_key, model)。优先 .env 直读（对齐 exec_workflow），默认 DeepSeek。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except Exception:
        pass
    base = os.getenv("OPENAI_API_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("AGENT_MODEL", "deepseek-chat")
    return base, key, model


def _build_llm(base_url: str, api_key: str, model: str):
    """构造 ChatOpenAI。use_responses_api=False 强制走 chat.completions 兼容路径。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=2048,
        request_timeout=90,
        use_responses_api=False,
    )


async def _role_answer(llm, role_key: str, material: str) -> str:
    """用某个角色的 default_prompt 对 material 做一次独立 LLM 调用（对齐 exec_workflow：
    default_prompt 作为同一条 user 消息尾部的『你的任务』，这样指令里的『上面』指的就是同一消息上文）。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from open_webui.models.knowledge import AGENT_ROLES

    info = AGENT_ROLES.get(role_key, {})
    name = info.get("name", role_key)
    default_prompt = info.get("default_prompt", "")
    resp = await llm.ainvoke(
        [
            SystemMessage(content=f"You are the {name}. Respond in Simplified Chinese."),
            HumanMessage(content=f"{material}\n\n你的任务：{default_prompt}"),
        ]
    )
    content = resp.content
    if isinstance(content, str):
        return content
    return str(content)


def make_tools(request, knowledge_id: str, llm):
    """把 5 个角色包装成工具。用闭包捕获运行时上下文（request / 知识库 / llm），
    名字与 AGENT_ROLES 的 key 一致（retriever/analyst/reporter/validator/translator）。"""
    from langchain_core.tools import tool

    @tool
    async def retriever(query: str) -> str:
        """在当前知识库中检索与问题最相关的内容，返回按文档整理好的连贯文本。需要事实依据时先调用它；参数 query 为要检索的问题或关键词。"""
        try:
            from open_webui.models.files import Files
            from open_webui.retrieval.utils import query_collection

            emb_fn = request.app.state.EMBEDDING_FUNCTION
            r = await query_collection(
                request=request,
                collection_names=[knowledge_id],
                queries=[query],
                embedding_function=emb_fn,
                k=15,
            )
            if not (isinstance(r, dict) and r.get("documents") and r["documents"][0]):
                return "(no documents found in knowledge base)"
            docs = r["documents"][0][:15]
            metas = r.get("metadatas", [[]])[0][:15] if r.get("metadatas") else []
            # Group chunks by file_id, keep doc order
            file_groups: OrderedDict = OrderedDict()
            for i, doc in enumerate(docs):
                if not doc:
                    continue
                fid = ""
                ci = i
                if i < len(metas) and metas[i]:
                    fid = metas[i].get("file_id", "")
                    ci = metas[i].get("chunk_index", i)
                if fid not in file_groups:
                    file_groups[fid] = []
                file_groups[fid].append((ci, doc))
            output = f"[Retrieved {len(docs)} chunks from {len(file_groups)} documents]\n\n"
            doc_num = 0
            for fid, chunks in file_groups.items():
                doc_num += 1
                chunks.sort(key=lambda x: x[0])
                src = fid[:20] if fid else "Unknown"
                if fid:
                    try:
                        f = await Files.get_file_by_id(fid)
                        if f:
                            src = f.filename
                    except Exception:
                        pass
                merged = "\n\n".join(c[1] for c in chunks)
                output += f"\n=== Document {doc_num}: {src} ===\n"
                output += merged[:8000] + "\n"
            return output
        except Exception as e:
            return f"(retrieval error: {str(e)[:150]})"

    @tool
    async def analyst(text: str, question: str) -> str:
        """对给定资料做深度分析，提炼与用户问题最相关的关键信息并要求引用原文。text 应传 retriever 的检索结果，question 为用户原本的问题。"""
        material = f"用户问题：{question}\n\n需要分析的资料：\n{(text or '')[:8000]}"
        try:
            return await _role_answer(llm, "analyst", material)
        except Exception as e:
            return f"(analyst error: {str(e)[:150]})"

    @tool
    async def reporter(text: str) -> str:
        """把给定内容整理成结构化中文报告（摘要 / 核心观点逐条带引用 / 详细说明 / 参考文档列表）。text 为待整理成报告的内容。"""
        try:
            return await _role_answer(llm, "reporter", f"需要整理成报告的内容：\n{(text or '')[:10000]}")
        except Exception as e:
            return f"(reporter error: {str(e)[:150]})"

    @tool
    async def validator(original: str, conclusion: str) -> str:
        """把检索到的原始文档与分析结论逐条对照核查，输出每条『✅一致 / ❌不一致 / ⚠️未提及』及总体一致性百分比。original 为原始检索资料，conclusion 为待核验的分析结论。"""
        material = (
            f"原始文档：\n{(original or '')[:6000]}\n\n分析结论：\n{(conclusion or '')[:6000]}"
        )
        try:
            return await _role_answer(llm, "validator", material)
        except Exception as e:
            return f"(validator error: {str(e)[:150]})"

    @tool
    async def translator(text: str) -> str:
        """把内容翻译为目标语言（保持原意）；若无法判断目标语言，默认译为简体中文。text 为待翻译内容，可在其中附带目标语言说明。"""
        try:
            return await _role_answer(llm, "translator", f"待翻译内容：\n{(text or '')[:8000]}")
        except Exception as e:
            return f"(translator error: {str(e)[:150]})"

    return [retriever, analyst, reporter, validator, translator]


def _system_prompt() -> str:
    return (
        "你是运行在知识库管理系统里的自主研究 Agent。当前绑定了一个企业知识库。\n"
        "你手上有 5 个工具：\n"
        "- retriever(query)：在当前知识库检索资料，返回按文档整理好的连贯文本。回答问题前请先调用它获取事实依据。\n"
        "- analyst(text, question)：对检索资料做深度提炼，输出带原文引用的关键信息。\n"
        "- reporter(text)：把内容整理成结构化中文报告。\n"
        "- validator(original, conclusion)：核查结论是否忠实于原文，逐条给出 一致/不一致/未提及 与一致性百分比。\n"
        "- translator(text)：翻译内容。\n"
        "自主决策规则：\n"
        "1) 先 retriever 检索；若一次不够（资料缺失或冲突），可换检索词再检一轮。\n"
        "2) 需要时再用 analyst / reporter / validator 加工。\n"
        "3) 当证据足够充分时，直接输出最终结论，不要再调用工具。\n"
        "要求：全程使用简体中文；最终结论用 Markdown，能指出出处就指出；不要无意义地反复调用同一工具。"
    )


def _args_preview(args) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args[:600]
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    return s[:600]


async def run_autonomous(request, knowledge_id: str, query: str, max_steps: int = 8):
    """自主式 Agent 执行主流程。async generator，逐个 yield SSE payload dict。

    dict 事件类型：
      {"type": "start"}
      {"type": "round", "index": n, "tool": <角色key>, "args": <参数预览>}
      {"type": "observation", "index": n, "preview": <工具返回预览>}
      {"type": "answer", "content": <最终结论>}
      {"type": "done", "status": "complete" | "error"}
      {"type": "error", "message": ...}
    """
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    base, key, model = _resolve_api()
    llm = _build_llm(base, key, model)
    tools = make_tools(request, knowledge_id, llm)
    agent = create_agent(model=llm, tools=tools, system_prompt=_system_prompt())

    yield {"type": "start"}

    tool_id_to_index: dict = {}
    round_counter = 0
    pending_answer = None
    recursion_limit = 2 * max_steps + 4
    try:
        async for step in agent.astream(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="updates",
        ):
            if not isinstance(step, dict):
                continue
            for _node, chunk in step.items():
                msgs = chunk.get("messages") if isinstance(chunk, dict) else None
                if not msgs:
                    continue
                for msg in msgs:
                    if isinstance(msg, AIMessage):
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                round_counter += 1
                                tc_id = tc.get("id")
                                if tc_id:
                                    tool_id_to_index[tc_id] = round_counter
                                yield {
                                    "type": "round",
                                    "index": round_counter,
                                    "tool": tc.get("name", ""),
                                    "args": _args_preview(tc.get("args")),
                                }
                        elif msg.content:
                            # 最后一次无工具调用的文本 = 最终结论
                            pending_answer = msg.content
                    elif isinstance(msg, ToolMessage):
                        idx = tool_id_to_index.get(msg.tool_call_id)
                        preview = (msg.content or "")[:400]
                        yield {"type": "observation", "index": idx, "preview": preview}
    except Exception as e:
        yield {"type": "error", "message": str(e)[:300]}
        yield {"type": "done", "status": "error"}
        return

    answer = pending_answer if isinstance(pending_answer, str) else str(pending_answer or "")
    yield {"type": "answer", "content": answer}
    yield {"type": "done", "status": "complete"}
