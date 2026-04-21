import os
from typing import TypedDict, Annotated, List
import operator

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver  # 新增：替换 InMemorySaver

# 新增：免费网页搜索工具
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests
from datetime import datetime
import  json

# ================== 配置 ==================
DATA_DIR = "./my_knowledge"
PERSIST_DIR = "./chroma_db"
CHECKPOINT_FILE = "chat_history"
CHECKPOINT_DB =os.path.join(CHECKPOINT_FILE,"chat_checkpoints.db")
SESSIONS_FILE = "./sessions.json"   # 新增：会话索引文件

LLM_MODEL = "qwen3:14b"
EMBEDDING_MODEL = "qwen3-embedding:latest"
# ================== 对话内容的本地化存储 ==================
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

    print(f"sessions:{sessions}")
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
# ================== 加载本地知识库 ==================
def get_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="my_local_knowledge"
    )
    print(f"✅ 知识库加载完成，共 {vectorstore._collection.count()} 个片段")
    return vectorstore.as_retriever(search_kwargs={"k": 6})


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
    summary: str = ""  # 新增：对话总结（可以是累积的 running summary）


# ================== 本地 LLM ==================
llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

retriever = get_retriever()


# ================== 节点 ==================
def retrieve_node(state: AgentState):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)

    context = "\n\n---\n\n".join([
        f"来源: {doc.metadata.get('source', '未知')}\n{doc.page_content}"
        for doc in docs
    ])

    # 简单判断：如果检索内容太短或为空，则需要上网
    needs_web = len(context.strip()) < 100  # 可根据实际情况调整阈值

    return {"context": context, "needs_web_search": needs_web}


def decide_node(state: AgentState):
    """决策节点：决定是否调用网页搜索"""
    if state.get("needs_web_search", False):
        return "web_search"
    else:
        return "answer"


def web_search_node(state: AgentState):
    """调用网页搜索"""
    question = state["messages"][-1].content
    web_context = web_search_tool.invoke(question)
    # 合并本地 + 网页上下文
    combined_context = state.get("context", "") + "\n\n=== 网页搜索结果 ===\n" + web_context
    return {"context": combined_context}

def answer_node(state: AgentState):
    context = state.get("context", "")
    question = state["messages"][-1].content

    prompt = f"""你是一个准确的助手。请优先使用提供的上下文回答问题。
如果上下文来自网页搜索，请注明“根据最新网络信息”。

以下是之前的对话总结（如果有）：
{state.get("summary", "无")}

上下文：
{context}


上下文：
{context}

问题：{question}

请用中文回答，并标注主要来源："""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def should_summarize(state: AgentState):
    """决策：是否需要总结"""
    # 每 5 轮对话（大致 10 条消息）总结一次，可根据实际情况调
    if len(state["messages"]) >= 10 and len(state["messages"]) % 2 == 0:  # 偶数消息时检查
        return "summarize"
    return "retrieve"  # 或直接进入正常流程

def summarize_conversation(state: AgentState):
    """每隔一定轮次对历史进行总结，并压缩 messages"""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # 决定是否需要总结（例如：每 5 轮用户+AI 对话，即消息数达到 10 或更多）
    if len(messages) < 10:  # 可调整阈值，例如 len(messages) // 2 >= 5
        return {"summary": existing_summary}  # 不总结，直接返回

    # 构建总结 prompt
    if existing_summary:
        summary_prompt = (
            f"这是到目前为止的对话总结：{existing_summary}\n\n"
            "请基于下面的新消息，更新并扩展这个总结。保持简洁但保留关键事实、用户偏好和重要上下文。\n"
            "新消息：\n"
        )
    else:
        summary_prompt = "请为下面的对话创建一个简洁的总结，突出关键点、用户需求和重要信息：\n"

    # 把所有消息转为字符串
    conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    # 调用 LLM 生成新总结
    summary_message = HumanMessage(content=summary_prompt + conversation)
    response = llm.invoke([summary_message])  # 或用更强的 prompt

    new_summary = response.content.strip()

    # **关键**：压缩 messages，只保留最近几条 + 把旧的替换为总结
    # 例如：保留最后 4 条消息（最近 2 轮），前面用总结替代
    recent_messages = messages[-4:] if len(messages) > 4 else messages

    return {
        "summary": new_summary,
        "messages": recent_messages,  # 覆盖旧消息列表，实现压缩
    }

# ================== 构建 LangGraph ==================
def build_rag_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("summarize", summarize_conversation)  # 新增总结节点
    workflow.set_entry_point("retrieve")

    # 条件分支：检索后判断是否需要上网
    workflow.add_conditional_edges(
        "retrieve",
        decide_node,
        {
            "web_search": "web_search",
            "answer": "answer"
        }
    )

    workflow.add_edge("web_search", "answer")
    # 添加条件边
    workflow.add_conditional_edges(
        "answer",  # 在回答完后判断是否总结（推荐放在 answer 之后）
        lambda state: "summarize" if len(state["messages"]) >= 10 else END,  # 简单示例
        {"summarize": "summarize", END: END}
    )

    # 总结完成后回到 END 或继续
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)

def generate_summary_to_save(current_state,session_id):
    conversation_text = "\n".join(
        [f"{m.type}: {m.content}" for m in current_state.get("messages", [])[-10:]])

    summary_prompt = f"""请为下面的对话生成一个简短的总结说明（控制在80字以内），适合作为会话标题的描述：
                       {conversation_text}
                       要求：突出主题、主要问题或番号相关内容，用中文。"""

    try:
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        short_summary = summary_response.content.strip()
        print(f"short_summary:{short_summary}")
        save_session(session_id, short_summary)
        print(f"💾 会话已保存 | ID: {session_id} | 总结: {short_summary[:80]}...")
    except Exception as e:
        print(f"❌ 生成总结失败: {e}")

# ================== 运行 ==================
if __name__ == "__main__":
    # 使用 SQLite 持久化（推荐）
    os.makedirs(CHECKPOINT_FILE,exist_ok=True)
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_rag_graph(checkpointer=checkpointer)
        print("🎉 增强版 LangGraph RAG Agent 已启动（支持持久化历史）\n")

        # 启动时显示所有历史会话
        sessions = load_sessions()
        print_sessions(sessions)

        while True:
            print("\n" + "-" * 60)
            user_input = input(
                "操作提示：\n"
                "• 输入 session_id（或番号）继续历史对话\n"
                "• 输入 'new' 或直接回车创建新会话\n"
                "• 输入 'list' 重新显示列表\n"
                "• 输入 'exit' / 'q' 退出\n"
                "请输入："
            ).strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                break
            elif user_input.lower() == "list":
                sessions = load_sessions()
                print_sessions(sessions)
                continue

            # ================== 选择或创建 session_id ==================
            if user_input.lower() == "new" or not user_input:
                # 创建新会话（自动生成或手动输入）
                session_id = input("请输入本次会话的番号 / ID（例如：AV-12345、项目-讨论1）：").strip()
                if not session_id:
                    session_id = f"session_{uuid.uuid4().hex[:8]}"
            else:
                session_id = user_input  # 用户直接输入的 session_id

            config = {"configurable": {"thread_id": session_id}}

            # 尝试加载已有会话的历史
            state_snapshot = graph.get_state(config)
            if state_snapshot and state_snapshot.values and state_snapshot.values.get("messages"):
                print(f"\n✅ 已加载历史会话 → {session_id}")
                # 可选：显示最近几条消息
                msgs = state_snapshot.values["messages"][-6:]  # 显示最近3轮
                for msg in msgs:
                    role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
                    print(f"  {role}: {msg.content}")
            else:
                print(f"\n🆕 创建新会话 → {session_id}")

            # ================== 开始对话循环（支持多轮）==================
            while True:
                user_msg = input("\n👤 你：").strip()
                if user_msg.lower() in ["exit", "quit", "q", "返回", "back"]:
                    current_state = graph.get_state(config).values
                    current_summay=current_state.get("summary", "")
                    print(f"current_state:{current_state}")
                    if current_summay:
                        save_session(session_id,current_summay)
                    else:
                        generate_summary_to_save(current_state, session_id)

                    break  # 返回到选择会话界面

                inputs = {"messages": [HumanMessage(content=user_msg)], "needs_web_search": False}

                # final_answer = ""
                # for output in graph.stream(inputs, config=config, stream_mode="values"):
                #     if "answer" in output or "messages" in output.get("values", {}):
                #         # 取出最新的 AI 回复
                #         messages = output.get("values", {}).get("messages",
                #                                                 []) if "values" in output else output.get(
                #             "messages", [])
                #         if messages and isinstance(messages[-1], AIMessage):
                #             final_answer = messages[-1].content
                #
                # if final_answer:
                #     print(f"\n🤖 AI：{final_answer}")

                print("\n🤖 AI：", end="", flush=True)  # 先打印前缀，不换行

                final_answer = ""

                for chunk in graph.stream(inputs, config=config, stream_mode="messages"):
                    # chunk 的结构通常是 (message_chunk, metadata)
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        message, metadata = chunk
                        if hasattr(message, "content") and message.content:
                            # 实时打印每个 token / 片段
                            print(message.content, end="", flush=True)
                            final_answer += message.content

                # # ================== 每次回答后更新总结（推荐放在这里）==================
                # # 你可以在 answer_node 里返回 summary，或者在这里调用 summarize_conversation 逻辑
                # # 为了简单，这里每次对话后用 LLM 生成一次简短总结
                # current_state = graph.get_state(config).values
                # conversation_text = "\n".join(
                #     [f"{m.type}: {m.content}" for m in current_state.get("messages", [])[-10:]])
                #
                # summary_prompt = f"""请为下面的对话生成一个简短的总结说明（控制在80字以内），适合作为会话标题的描述：
                #    {conversation_text}
                #    要求：突出主题、主要问题或番号相关内容，用中文。"""
                #
                # try:
                #     summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
                #     short_summary = summary_response.content.strip()
                #     title = short_summary[:60]  # 可作为标题
                #
                #     save_session(session_id, short_summary, title)
                #     print(f"💾 会话已保存 | ID: {session_id} | 总结: {short_summary[:80]}...")
                # except:
                #     pass  # LLM 出错时不影响主流程