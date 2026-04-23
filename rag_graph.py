import os
from typing import TypedDict, Annotated, List
import operator
import requests
from datetime import datetime
import  json
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt

# 新增：免费网页搜索工具
from ddgs import DDGS
from bs4 import BeautifulSoup


# ================== 配置 ==================
DATA_DIR = "./my_knowledge"
PERSIST_DIR = "./chroma_db"
CHECKPOINT_FILE = "chat_history"
CHECKPOINT_DB =os.path.join(CHECKPOINT_FILE,"chat_checkpoints.db")
SESSIONS_FILE = "./sessions.json"   # 新增：会话索引文件

LLM_MODEL = "qwen3:14b"
EMBEDDING_MODEL = "qwen3-embedding:latest"

# ================== prompt设定 ==================
SUMMARY_PROMPT_01 ="prompts/summary_conversation_prompt_01.txt"
SUMMARY_PROMPT_02 ="prompts/summary_conversation_prompt_02.txt"
SUMMARY_PROMPT_03 ="prompts/generate_summary_to_save.txt"
ANSWER_PROMPT = "prompts/answer_node.txt"
FEEDBACK_PROMPT = "prompts/feedback_node.txt"

# ================== 工具设定 ==================
def is_meaningless_input(text: str) -> bool:
    """判断输入是否没有实际意义"""
    if len(text) <= 1:  # 单个字符
        return True

    # 只包含标点符号的情况
    import string
    if all(c in string.punctuation + "，。！？；：（）【】「」" for c in text):
        return True

    # 太短且常见无意义词（可自行扩展）
    meaningless = {"嗯", "哦", "啊", "好", "行", "可以", "是的", "恩", "嘿", "hello", "hi"}
    if text.lower() in meaningless or text in meaningless:
        return True

    return False
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
    next:str # 新增：下一步动作
    summary: str = ""  # 新增：对话总结（可以是累积的 running summary）
    question: str
    # 新增字段（用于人工反馈循环）
    feedback: str | None  # 人工输入的改善点
    revision_count: int  # 防止无限循环，可选


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

    return {"context": context, "needs_web_search": needs_web,"question":question,\
            "next": "web_search" if needs_web else "answer"}


def web_search_node(state: AgentState):
    """调用网页搜索"""
    question = state.get("question")
    web_context = web_search_tool.invoke(question)
    # 合并本地 + 网页上下文
    combined_context = state.get("context", "") + "\n\n=== 网页搜索结果 ===\n" + web_context
    return {"context": combined_context}

def answer_node(state: AgentState):
    context = state.get("context", "")
    question = state.get("question")

    existing_summary=state.get("summary", "无")


    prompt_path = Path(ANSWER_PROMPT)
    summary_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    prompt = summary_prompt_template.format(existing_summary=existing_summary, context=context,question=question)

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)],"question":question}

#
def human_review(state: AgentState):
    """人工审查节点：在这里中断，等待人工输入"""
    last_answer = state["messages"][-1].content if state["messages"] else "（无回答）"
    current_question = state.get("question", "（未知问题）")
    revision_count = state.get("revision_count", 0)

    # ================== 打印清晰的人工审查提示 ==================
    print("\n" + "=" * 80)
    print("🔍【人工审查模式】")
    print("=" * 80)
    print(f"📌 **当前问题**：{current_question}")
    print("\n🤖 AI 的回答：")
    print("-" * 60)
    print(last_answer .strip())
    print("-" * 60)
    print("\n💡 请审查上面的回答是否满意？")
    print("操作说明：")
    print("   • 输入 '满意'、'approve'、'yes' 或直接回车  → 通过审查")
    print("   • 输入 '不满意: 你的改善建议'             → 要求AI修改")
    print("     示例：不满意: 回答太简短，请添加更多细节和具体例子")
    print("   • 输入 'skip' 或 '跳过'                    → 直接跳过审查进入总结")

    # 构建清晰的 payload（全部展示给人工）
    interrupt_payload = {
        "task": "review_answer",
        "question": current_question,
        "answer": last_answer,
        "revision_count": revision_count,
        "max_revisions": 3,
        "instruction": (
            "请输入以下之一：\n"
            "• 'approve' 或 '满意' 或直接回车 → 通过审查\n"
            "• 'skip' 或 '跳过' → 跳过审查\n"
            "• '不满意: 你的具体改善建议' → 要求AI修改（推荐）"
        )
    }

    # ================== 执行中断 ==================
    raw_decision = interrupt(interrupt_payload)

    # 安全提取人工输入（支持 str / dict / Message 等常见情况）
    if isinstance(raw_decision, dict):
        decision = raw_decision.get("response") or str(raw_decision)
    elif hasattr(raw_decision, "content"):  # AIMessage / HumanMessage
        decision = raw_decision.content
    else:
        decision = str(raw_decision)

    decision = decision.strip()

    # === 决策逻辑 ===
    decision_lower = decision.lower()

    # 通过审查（包括空输入视为 approve）
    if (not decision or
            decision_lower in ["approve", "yes", "y", "满意", "好", "通过", "ok"] or
            revision_count >= 3):

        if revision_count >= 3:
            print("已达到最大修改次数（3次），强制结束审查")  # 仅调试用

        # 决定下一个节点
        next_node = "summarize" if len(state.get("messages", [])) >= 10 else END
        return Command(goto=next_node, update={
            "feedback": None,
            "revision_count": 0
        })

    # 跳过审查
    if decision_lower in ["skip", "跳过"]:
        return Command(goto="retrieve", update={
            "feedback": None,
            "revision_count": 0
        })

    # 不满意 → 提取反馈
    feedback = decision
    if ":" in decision:
        feedback = decision.split(":", 1)[1].strip()
    elif "不满意" in decision or "dissatisfied" in decision_lower:
        feedback = decision.replace("不满意", "").replace("dissatisfied", "").strip()

    # 返回修改请求 + 计数 + 跳转到反馈/修改节点
    return Command(goto="feedback_node", update={
        "feedback": feedback,
        "context": last_answer,  # 可选：把原回答也存下来
        "revision_count": revision_count + 1
    })



def feedback_node(state: AgentState):
    context = state.get("context", "")
    question = state.get("question")
    existing_summary = state.get("summary", "无")
    feedback = state.get("feedback")

    prompt_path = Path(FEEDBACK_PROMPT)
    summary_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    prompt = summary_prompt_template.format(existing_summary=existing_summary, context=context, question=question, feedback=feedback)

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}


def summarize_conversation(state: AgentState):
    """每隔一定轮次对历史进行总结，并压缩 messages"""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # 决定是否需要总结（例如：每 5 轮用户+AI 对话，即消息数达到 10 或更多）
    if len(messages) < 10:  # 可调整阈值，例如 len(messages) // 2 >= 5
        return {"summary": existing_summary}  # 不总结，直接返回

    # 把所有消息转为字符串
    conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    # 构建总结 prompt
    if existing_summary:
        prompt_path = Path(SUMMARY_PROMPT_01)
        summary_prompt_template=prompt_path.read_text(encoding="utf-8").strip()
        summary_prompt =summary_prompt_template.format(existing_summary=existing_summary,conversation=conversation)
    else:
        prompt_path = Path(SUMMARY_PROMPT_02)
        summary_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        summary_prompt = summary_prompt_template.format(conversation=conversation)


    # 调用 LLM 生成新总结
    summary_message = HumanMessage(content=summary_prompt)
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
    workflow.add_node("human_review",human_review)
    workflow.add_node("feedback_node", feedback_node)
    workflow.add_node("summarize", summarize_conversation)
    workflow.set_entry_point("retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        lambda state: state.get("next"),
        {
            "web_search": "web_search",
            "answer": "answer"
        }
    )

    workflow.add_edge("web_search", "answer")
    workflow.add_edge("answer", "human_review")
    workflow.add_edge("feedback_node", "human_review")
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)

def generate_summary_to_save(current_state,session_id):
    conversation_text = "\n".join(
        [f"{m.type}: {m.content}" for m in current_state.get("messages", [])[-10:]])

    prompt_path = Path(SUMMARY_PROMPT_03)
    summary_prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    summary_prompt = summary_prompt_template.format(conversation_text=conversation_text)

    try:
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        short_summary = summary_response.content.strip()
        save_session(session_id, short_summary)
        print(f"💾 会话已保存 | ID: {session_id} | 总结: {short_summary[:80]}...")
    except Exception as e:
        print(f"❌ 生成总结失败: {e}")


def run_graph_with_human_review(graph, inputs, config):
    """支持多轮 interrupt 的流式运行"""
    command = inputs  # 第一次用原始 inputs，之后用 Command(resume=...)

    while True:
        final_answer = ""
        interrupted = False

        for chunk in graph.stream(command, config=config, stream_mode="updates"):
            # 检测到 interrupt 信号
            if "__interrupt__" in chunk:
                interrupted = True
                interrupt_obj = chunk["__interrupt__"][0]  # 取第一个 Interrupt 对象
                payload = interrupt_obj.value  # 就是你传给 interrupt() 的 dict

                user_input = input("\n>>> 你的决策：").strip()
                command = Command(resume={"response": user_input})
                break
            # 正常节点的输出，提取消息内容流式打印
            for node_name, update in chunk.items():
                if isinstance(update, dict):
                    msgs = update.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            final_answer += msg.content

        if not interrupted:
            break

    return final_answer



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
            session_summary =""
            if user_input.lower() == "new" or not user_input.strip():
                # === 创建新会话 ===
                session_id = input("请输入本次会话的番号 / ID（例如：AV-12345、项目-讨论1）：").strip()

                if not session_id:
                    session_id = f"session_{uuid.uuid4().hex[:8]}"

                print(f"已创建新会话：{session_id}")

            else:
                # === 使用已有会话 ===
                session_id = user_input.strip()

                # 关键判断：检查这个 session 是否已经存在
                sessions = load_sessions()  # 你现有的加载所有 session 的函数
                session_find_flag =False
                for each_session in sessions:
                    if each_session['session_id'] == session_id:
                        session_summary =each_session["summary"]
                        session_find_flag =True
                        break

                if not session_find_flag:
                    print(f"❌ 错误：会话 ID '{session_id}' 不存在！")
                    print("提示：输入 'new' 创建新会话，或输入 'list' 查看已有会话。")
                    continue  # 跳过本次循环，不往下执行

                print(f"已加载现有会话：{session_id}")

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
                return_flag =False
                while True:
                    user_msg = input("\n👤 你：").strip()
                    # 1. 判空
                    if not user_msg:
                        print("❌ 输入不能为空，请输入你的问题。")
                        continue
                    # 2. 判断是否退出
                    if user_msg.lower() in ["exit", "quit", "q", "返回", "back"]:
                        current_state = graph.get_state(config).values
                        current_summay=current_state.get("summary", "")

                        if current_summay:
                            save_session(session_id,current_summay)
                        elif session_summary:
                            save_session(session_id,session_summary)
                        else:
                            generate_summary_to_save(current_state, session_id)

                        return_flag = True
                    if not return_flag and is_meaningless_input(user_msg):
                        print("❌ 请输入有意义的问题或指令。")
                        continue
                    break
                if return_flag:
                        break

                inputs = {"messages": [HumanMessage(content=user_msg)], "needs_web_search": False,\
                          "summary":session_summary,"feedback":None,"revision_count":0}

                final_answer=run_graph_with_human_review(graph, inputs, config)

                print("\n🤖 AI：", end="", flush=True)  # 先打印前缀，不换行
                print(final_answer)