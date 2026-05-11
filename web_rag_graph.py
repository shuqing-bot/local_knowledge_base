import os
from typing import TypedDict, Annotated, List
import operator
import requests
from datetime import datetime
import  json
from pathlib import Path
import sqlite3
import aiofiles
import asyncio
import traceback


from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from langgraph.errors import GraphInterrupt

# 新增：免费网页搜索工具
from ddgs import DDGS
from bs4 import BeautifulSoup
from config import Config

# ================== 本地 LLM ==================
setting_config = Config()
def get_llm():
    return ChatOllama(
        model=setting_config.models["LLM_MODEL"],
        temperature=setting_config.params["TEMPERATURE"],
        num_ctx=setting_config.params["NUM_CTX"],
        num_predict=setting_config.params.get("NUM_PREDICT", setting_config.params["MAX_NEW_TOKENS"]),
        top_p=setting_config.params["TOP_P"],
        repeat_penalty=setting_config.params["REPEAT_PENALTY"]
    )

llm = get_llm()


# ================== 配置 ==================
CHECKPOINT_FILE = setting_config.paths["CHECKPOINT_FILE"]
CHECKPOINT_DB =setting_config.paths["CHECKPOINT_DB"]
SESSIONS_FILE = setting_config.paths["SESSIONS_FILE"]
SCORE_THRESHOLD = setting_config.params["SCORE_THRESHOLD"]
TOP_K = setting_config.params["TOP_K"]
SUMMARY_INTERVAL=setting_config.params.get("SUMMARY_INTERVAL",10)


# ================== prompt设定 ==================
SUMMARY_PROMPT_01 =setting_config.prompts["SUMMARY_PROMPT_01"]
SUMMARY_PROMPT_02 =setting_config.prompts["SUMMARY_PROMPT_02"]
SUMMARY_PROMPT_03 =setting_config.prompts["SUMMARY_PROMPT_03"]
ANSWER_PROMPT = setting_config.prompts["ANSWER_PROMPT"]
FEEDBACK_PROMPT = setting_config.prompts["FEEDBACK_PROMPT"]

# ================== 知识库设定 ==================
def initialize_vectorstore():
    """初始化向量数据库，只需调用一次"""
    embeddings = OllamaEmbeddings(model=setting_config.models["EMBEDDING_MODEL"])

    vectorstore = Chroma(
        persist_directory=setting_config.paths["PERSIST_DIR"],  # 从config读取
        embedding_function=embeddings,
        collection_name="my_local_knowledge"
    )

    doc_count = vectorstore._collection.count() if hasattr(vectorstore, '_collection') else 0
    print(f"✅ 知识库加载完成，共 {doc_count} 个文档片段")

    return vectorstore

vectorstore = initialize_vectorstore()

def get_retriever():
    """获取检索器"""
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",  # 强烈建议使用这个
        search_kwargs={
            "k": TOP_K,
            "score_threshold": SCORE_THRESHOLD,
        }
    )



# ================== 加载会话记录 ==================
def load_sessions():
    """加载 sessions.json，如果不存在则返回空列表"""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_session(session_id: str, summary: str):
    """保存或更新会话记录"""
    sessions = load_sessions()

    found = False
    for s in sessions:
        if s.get("session_id") == session_id:
            s["summary"] = summary.strip()
            s["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            found = True
            break

    if not found:
        sessions.append({
            "session_id": session_id,
            "summary": summary.strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # 按更新时间倒序排序
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)

    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

def print_sessions(sessions):
    """漂亮地打印会话列表"""
    if not sessions:
        print("📭 目前没有任何历史会话。")
        return

    print("\n" + "=" * 60)
    print("📜 历史对话列表（按最近更新排序）")
    print("=" * 60)
    for i, s in enumerate(sessions, 1):
        print(f"{i:2d}. Session ID: {s['session_id']}")
        print(f"    总结: {s['summary'][:120]}{'...' if len(s['summary']) > 120 else ''}")
        print(f"    更新时间: {s['updated_at']}")
        print("-" * 50)

async def generate_summary_to_save(message_list, session_id):
    conversation_text = "\n".join(
        [f"{m.type}: {m.content}" for m in message_list])

    prompt_path = Path(SUMMARY_PROMPT_03)
    summary_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    summary_prompt = summary_prompt_template.format(conversation_text=conversation_text)

    try:
        summary_response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
        short_summary = summary_response.content.strip()
        save_session(session_id, short_summary)
        print(f"💾 会话已保存 | ID: {session_id} | 总结: {short_summary[:80]}...")
    except Exception as e:
        print(f"❌ 生成总结失败: {e}")



# ================== 免费网页搜索工具 ==================
@tool
def web_search_tool(query: str, max_results: int = 5) -> str:
    """当本地知识库没有足够信息时，从互联网搜索最新信息"""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]

        context = ""
        for i, r in enumerate(results, 1):
            context += f"【搜索结果 {i}】标题: {r['title']}\n链接: {r['href']}\n摘要: {r['body']}\n\n"

        return context if context else "网页搜索未找到相关结果。"
    except Exception as e:
        return f"网页搜索出错: {str(e)}"


# ================== LangGraph 状态 ==================
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    context: str
    needs_web_search: bool  # 新增：是否需要上网搜索
    next:str # 新增：下一步动作
    summary: str = ""  # 新增：对话总结（可以是累积的 running summary）
    question: str
    # 新增字段（用于人工反馈循环）
    feedback: str | None  # 人工输入的改善点
    revision_count: int  # 防止无限循环，可选


# ================== 节点 ==================
async def retrieve_node(state: AgentState):
    """检索本地知识库（异步版本）"""
    feedback = state.get("feedback", "")

    if feedback == "":
        question = state["messages"][-1].content
        retriever = get_retriever()

        # 改成异步调用
        docs = await retriever.ainvoke(question)

        # 获取上下文
        context = "\n\n---\n\n".join([
            f"来源: {doc.metadata.get('source', '未知')}\n{doc.page_content}"
            for doc in docs
        ]) if docs else ""

        # 看最高分或平均分是否达标
        if not docs:
            needs_web = True
        else:
            scores = [doc.metadata.get('score', 0) for doc in docs]
            max_score = max(scores) if scores else 0
            avg_score = sum(scores) / len(scores) if scores else 0

            needs_web = max_score < SCORE_THRESHOLD or avg_score < SCORE_THRESHOLD - 0.1

        return {"context": context, "needs_web_search": needs_web, "question": question, \
                "next": "web_search" if needs_web else "answer"}
    else:
        question = state.get("question", "")
        needs_web = state.get("needs_web_search", False)
        context = state.get("context", "")

    return {
        "context": context,
        "needs_web_search": needs_web,
        "question": question,
        "next": "web_search" if needs_web else "answer"
    }


async def web_search_node(state: AgentState):
    """调用网页搜索（异步版本）"""
    feedback = state.get("feedback", "")

    if feedback == "":
        question = state.get("question")
    else:
        question = state.get("question", "") + "用户反馈：" + feedback

    # 改成异步调用
    web_context = await web_search_tool.ainvoke(question)

    # 合并本地 + 网页上下文
    combined_context = state.get("context", "") + "\n\n=== 网页搜索结果 ===\n" + web_context

    return {"context": combined_context}

async def answer_node(state: AgentState):
    context = state.get("context", "")
    question = state.get("question")
    feedback = state.get("feedback", "")
    existing_summary = state.get("summary", "无")

    # 异步读取 prompt 文件
    if feedback:
        prompt_path = Path(FEEDBACK_PROMPT)
        async with aiofiles.open(prompt_path, encoding="utf-8") as f:
            template = (await f.read()).strip()
        prompt = template.format(
            existing_summary=existing_summary,
            context=context,
            question=question,
            feedback=feedback
        )
    else:
        prompt_path = Path(ANSWER_PROMPT)
        async with aiofiles.open(prompt_path, encoding="utf-8") as f:
            template = (await f.read()).strip()
        prompt = template.format(
            existing_summary=existing_summary,
            context=context,
            question=question
        )

    # 使用异步 LLM 调用（LangChain 支持 ainvoke）
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
        "question": question
    }

# 兼容 dict 和 Message 对象
def get_type_and_content(msg):
    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type", "unknown")
        content = msg.get("content", str(msg))
        return role, content
    elif hasattr(msg, "type") and hasattr(msg, "content"):
        return msg.type, msg.content
    else:
        return "unknown", str(msg)


async def summarize_conversation(state: AgentState):
    """每隔一定轮次对历史进行总结，并压缩 messages（异步版本）"""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # 决定是否需要总结
    if len(messages) < 10:   # 可自行调整阈值
        return {"summary": existing_summary}

    # 把所有消息转为字符串
    conversation = "\n".join(
        [f"{role}: {content}" for role, content in
         (get_type_and_content(msg) for msg in messages)]
    )

    # 异步读取 Prompt 文件
    if existing_summary:
        prompt_path = Path(SUMMARY_PROMPT_01)
        template = await asyncio.to_thread(
            prompt_path.read_text, encoding="utf-8"
        )
        summary_prompt = template.strip().format(
            existing_summary=existing_summary,
            conversation=conversation
        )
    else:
        prompt_path = Path(SUMMARY_PROMPT_02)
        template = await asyncio.to_thread(
            prompt_path.read_text, encoding="utf-8"
        )
        summary_prompt = template.strip().format(conversation=conversation)

    # 异步调用 LLM
    summary_message = HumanMessage(content=summary_prompt)
    response = await llm.ainvoke([summary_message])

    new_summary = response.content.strip()

    # 压缩消息历史（保留最近几条）
    recent_messages = messages[-4:] if len(messages) > 4 else messages

    return {
        "summary": new_summary,
        "messages": recent_messages,   # 覆盖旧消息，实现历史压缩
    }


async def build_web_rag_graph(checkpointer=None):
    """构建异步 Web RAG Graph（推荐用于 FastAPI）"""

    workflow = StateGraph(AgentState)

    # 添加异步节点
    workflow.add_node("retrieve", retrieve_node)  # 必须是 async def
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("summarize", summarize_conversation)

    workflow.set_entry_point("retrieve")

    # 条件边
    workflow.add_conditional_edges(
        "retrieve",
        lambda state: state.get("next"),  # 可以保持同步 lambda
        {
            "web_search": "web_search",
            "answer": "answer"
        }
    )

    workflow.add_edge("web_search", "answer")
    workflow.add_edge("answer", "summarize")
    workflow.add_edge("summarize", END)

    # 编译图（支持异步）
    return workflow.compile(checkpointer=checkpointer)


async def test_graph():
    print("🚀 开始测试 Web RAG Graph...\n")

    # 1. 创建 checkpointer
    checkpointer = MemorySaver()

    # 2. 构建图
    print("📦 正在构建 Graph...")
    graph = await build_web_rag_graph(checkpointer=checkpointer)  # 如是同步版则去掉 await

    # 测试问题
    test_question = "请问 LangGraph 是什么？它和 LangChain 有什么关系？"

    initial_state = {
        "messages": [HumanMessage(content=test_question)],
        "question": test_question,
        "context": "",
        "summary": "",
        "feedback": "",
        "needs_web_search": False,
    }

    # 【关键修复】必须传入 config + thread_id
    config = {"configurable": {"thread_id": "test_thread_001"}}

    print(f"❓ 用户问题: {test_question}\n")
    print("⏳ Graph 正在运行，请稍等...\n")

    # 异步调用时传入 config
    result = await graph.ainvoke(initial_state, config=config)

    print("=" * 70)
    print("✅ 最终回答：")
    print("=" * 70)
    print(result.get("messages")[-1].content)
    print("=" * 70)

    print(f"\n📊 总结: {result.get('summary', '无')[:300]}...")


if __name__ == "__main__":
    try:
        asyncio.run(test_graph())
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()