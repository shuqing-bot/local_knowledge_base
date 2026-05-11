from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
from web_rag_graph import build_web_rag_graph, load_sessions,save_session,generate_summary_to_save,llm
from datetime import datetime
from typing import List, Dict,Optional
import os
import json
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from config import Config


# 模板目录（根据你的结构调整）
BASE_DIR = Path(__file__).resolve().parent
setting_config = Config()
templates_path = os.path.join(BASE_DIR ,setting_config.paths["WEB_DIR"])
templates = Jinja2Templates(directory=templates_path)

# ================== 配置 ==================
CHECKPOINT_DB =setting_config.paths["CHECKPOINT_DB"]
SESSIONS_FILE = setting_config.paths["SESSIONS_FILE"]

# ================== 数据模型 ==================
class ChatRequest(BaseModel):
    message: str = ""
    session_id: Optional[str] = None
    action: Optional[str] = None
    question:str = ""



# ================== 路由函数 ==================
def get_sessions_data(sessions: list) -> dict:
    """把会话列表转换成前端友好的 JSON 格式"""
    if not sessions:
        return {
            "sessions": [],
            "total": 0,
            "message": "📭 目前没有任何历史会话。"
        }

    formatted_sessions = []
    for s in sessions:
        # 确保 updated_at 是字符串格式（前端更方便）
        updated_str = (
            s['updated_at'].strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(s['updated_at'], datetime)
            else str(s['updated_at'])
        )

        formatted_sessions.append({
            "session_id": s['session_id'],
            "summary": s['summary'],
            "updated_at": updated_str,
            "message_count": s.get('message_count', 0),
        })

    return {
        "sessions": formatted_sessions,
        "total": len(formatted_sessions),
        "message": f"共 {len(formatted_sessions)} 个历史会话"
    }


# ====================== Lifespan ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 启动中：初始化 Async SQLite Checkpointer...")


    os.makedirs(os.path.dirname(CHECKPOINT_DB) or ".", exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        # 第一次使用建议调用 setup（创建表）
        await checkpointer.setup()

        app.state.checkpointer = checkpointer
        app.state.graph= await build_web_rag_graph(checkpointer=checkpointer)
        app.state.is_new_mode = False  # ← 单个全局布尔变量，用于判断是否是新建会话
        # 新增：初始化会话列表
        app.state.messages_list = []
        app.state.question = ""
        app.state.context = ""
        app.state.summary =""
        app.state.needs_web_search = False
        app.state.session_summary = ""


        print(f"✅ LangGraph 已初始化，持久化文件：{CHECKPOINT_DB}")
        yield

    print("🛑 LangGraph Checkpointer 已关闭")

async def delete_session(checkpointer, session_id: str) -> bool:
    """异步删除某个会话的所有记录"""
    if checkpointer is None:
        print("❌ Checkpointer 未初始化")
        return False

    # 1. 删除 LangGraph 的 checkpoint（状态历史）
    try:
        await checkpointer.adelete_thread(thread_id=session_id)  # ← 必须用 adelete_thread + await
        print(f"✅ 已删除 session {session_id} 的所有 checkpoints 和 writes")
    except Exception as e:
        print(f"删除 checkpoint 时出错: {e}")
        # 可以选择 return False 或继续删除 sessions 文件

    # 2. 删除你自己维护的 sessions 列表（如果有）
    try:
        sessions = load_sessions()
        new_sessions = [s for s in sessions if s.get('session_id') != session_id]

        if len(new_sessions) != len(sessions):
            with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(new_sessions, f, ensure_ascii=False, indent=2)
            print(f"✅ 已从 sessions.json 中移除 session {session_id}")
            return True
        else:
            print(f"ℹ️ sessions.json 中未找到 session {session_id}")
            return False
    except Exception as e:
        print(f"处理 sessions.json 时出错: {e}")
        return False
app = FastAPI(title="本地知识库聊天机器人",lifespan=lifespan)
app.mount("/web/static", StaticFiles(directory=BASE_DIR / "web" / "static"), name="static")
#================== 路由调用 ==================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )
@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    print("🔍 收到完整请求体:", req)  # ← 看这里！

    user_msg = req.message.strip() if hasattr(req, 'message') else ""
    session_id = req.session_id
    action = getattr(req, 'action', None)


    # 安全获取 graph 和 checkpointer
    graph = getattr(request.app.state, "graph", None)
    checkpointer = getattr(request.app.state, "checkpointer", None)
    is_new_mode = getattr(request.app.state, "is_new_mode", False)

    # ==================== 新建会话逻辑 ====================
    messages_list=getattr(request.app.state, "messages_list", [])





    # ==================== 删除会话逻辑 ====================
    if action == "delete_session":
        if not session_id:
            return {"status": "error", "reply": "缺少 session_id"}

        print(f"✅ 收到删除会话请求: {session_id}")

        # 查询对话列表查找是否有对应的session_id,有则删除，并返回删除信息，没有则返回错误信息
        sessions = load_sessions()
        session_ids = [s['session_id'] for s in sessions]

        if session_id not in session_ids:
            return {
                "reply": f"❌ 会话 {session_id} 不存在，删除失败，请重新输入delete，并指定正确的session_id,可通过list指令查询",
                "action":  None,
                "session_id":None,
                "status": "Failed"
            }

        # 执行删除
        delete_flag = await delete_session(checkpointer, session_id)

        if delete_flag:
            return {
                "reply": f"🗑️ 会话 {session_id} 已删除",
                "action": None,
                "session_id": None,
                "status": "ok"
            }
        else:
            return {
                "reply": f"❌ 会话 {session_id} 不存在，删除失败,请检查失败原因，重启服务",
                "action": None,
                "session_id": None,
                "status": "Failed"
            }

    if user_msg.lower() == "new":
        request.app.state.is_new_mode = True
        print("✅ 触发 new 指令！")
        return {
            "action": "request_new_session",
            "status": "ok"
        }
    elif user_msg.lower() == "list":
        print("✅ 触发 list 指令！")
        sessions = load_sessions()
        session_data = get_sessions_data(sessions)
        return {
            "action": "show_session_list",
            "sessions": session_data["sessions"],
            "total": session_data["total"],
            "status": "ok"
        }
    elif user_msg.lower() == "del":
        print("✅ 触发 delete 指令！")
        request.app.state.is_new_mode = False
        return {
            "action": "delete_session",
            "status": "ok"
        }
    elif user_msg.lower() == "quit":
        print("✅ 触发 quit 指令！")
        quit_message = "现有会话：{session_id}已退出👋"
        summary = getattr(request.app.state, "summary", "")
        session_summary = getattr(request.app.state, "session_summary", "")
        if summary:
            save_session(session_id, summary)
        elif session_summary:
            save_session(session_id, session_summary)
        else:
            if len(messages_list) != 0:
                generate_summary_to_save(message_list, session_id)
            else:
                quit_message = "❌ 未输入任何内容，会话不需要保存。"
        session_id = None
        request.app.state.is_new_mode = False
        return {
            "reply": quit_message ,
            "status": "ok"
        }
    else:
        if session_id is None and is_new_mode:
            return {
                "reply": f"❌ 会话 {session_id} 不存在，请先指定session_id",
                "status": "Failed"
            }
        elif is_new_mode == False:
            # 关键判断：检查这个 session 是否已经存在
            sessions = load_sessions()
            session_find_flag = False
            session_id= user_msg.lower()
            for each_session in sessions:
                if each_session['session_id'] == session_id:
                    session_summary = each_session["summary"]
                    request.app.state.session_summary= session_summary
                    session_find_flag = True
                    break

            if not session_find_flag:
                return {
                    "reply": f"❌ 会话 {session_id} 不存在，提示：输入 'new' 创建新会话，或输入 'list' 查看已有会话。",
                    "status": "Failed"
                }
        if not user_msg:
            return {"status": "error", "reply": "消息不能为空"}

        if graph is None:
            return {"status": "error", "reply": "Graph 未初始化"}

        config = {"configurable": {"thread_id": session_id}}

        # 根据关键词判断使用哪个 graph（你可以继续扩展关键词）
        msg_lower = user_msg.lower()
        if msg_lower.startswith("不满意")or msg_lower.startswith("dissatisfied"):
            if ":" in msg_lower:
                feedback = msg_lower.split(":", 1)[1].strip()
            question = getattr(request.app.state, "question", "")
            ai_response = getattr(request.app.state, "context", "")
            summary =getattr(request.app.state, "summary", "")
            needs_web_search =getattr(request.app.state, "needs_web_search", False)
            print(f"🔀 使用 Feedback Graph ")

        else:
            # 不是feedback就要重置初始状态
            question = user_msg
            ai_response = ""
            summary = "无"
            needs_web_search = False
            feedback = ""
            print(f"🔀 使用 ragGraph ")

        initial_state = {
            "messages": [HumanMessage(content=user_msg)],
            "question": question,
            "context": ai_response,
            "summary": summary,
            "feedback": feedback,
            "needs_web_search": needs_web_search,
        }
        print(f"🔀 使用  initial_state: { initial_state} ")
        request.app.question = question
        # ====================== 执行 Graph ======================
        try:

            # 使用异步调用（必须配合 AsyncSqliteSaver）
            result = await graph.ainvoke(initial_state, config=config)

            # 取出最后一条 AI 回复
            ai_response = result["messages"][-1].content if isinstance(result.get("messages"), list) else str(result)

            messages_list.append(result.get("messages",[]))
            request.app.state.messages_list = messages_list
            request.app.state.summary = result.get("summary","")
            request.app.state.needs_web_search = result.get("needs_web_search", False)
            request.app.state.context = ai_response


            return {
                "status": "ok",
                "reply": ai_response ,
                "session_id": session_id,
                "action": None
            }

        except Exception as e:
            print(f"❌ Graph 执行出错: {e}")
            raise HTTPException(status_code=500, detail="对话处理失败")



if __name__ == "__main__":
    uvicorn.run("web_chat_bot:app", host="127.0.0.1", port=8000, reload=True)