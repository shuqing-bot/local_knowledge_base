# local_knowledge_base

一个基于 **LangChain + LangGraph + Ollama + Chroma** 的本地知识库问答项目，支持从本地文档检索知识，并在必要时补充网页搜索结果，同时提供人工审查与会话持久化能力。

---

## 1. 项目简介

该项目通过 RAG（Retrieval-Augmented Generation）流程实现“**本地知识优先 + 网络信息兜底**”的问答体验：

- 优先从本地向量库检索相关内容；
- 若本地上下文不足，自动触发网页搜索；
- 生成回答后进入人工审查环节，可通过/修改/跳过；
- 多轮对话可按会话 ID 持久化，支持历史总结。
- 并支持删除对话的功能。

知识源主要来自 `my_knowledge/` 目录中的个人文档（docx/pdf/ppt/pptx/xlsx 等）。

---

## 2. 核心功能

- **本地知识库检索**（Chroma 持久化向量库）
- **Ollama 本地模型推理**（LLM + Embedding）
- **检索不足自动网页搜索**（DDGS）
- **Human-in-the-loop 审查**（满意/不满意反馈重写/跳过）
- **会话级记忆与总结**
  - SQLite Checkpoint 持久化
  - `sessions.json` 会话摘要索引
- **无意义输入过滤**（短字符、纯标点、常见无效词）

---

## 3. 项目结构（关键文件）

```text
local_knowledge_base/
├─ rag_graph.py                    # 主程序：图流程、节点、交互式对话入口
├─ index_knowledge.py              # 向量库生成：从本地知识库生成向量并持久化
├─ requirements.txt                # Python 依赖
├─ loaded_docs_cache.json          # 已向量化文档目录（文档路径 -> 文档内容哈希值）
├─ prompts/                        # Prompt 模板
│  ├─ answer_node.txt
│  ├─ feedback_node.txt
│  ├─ generate_summary_to_save.txt
│  ├─ summary_conversation_prompt_01.txt
│  └─ summary_conversation_prompt_02.txt
├─ my_knowledge/                   # 本地知识文档目录
├─ chroma_db/                      # 向量数据库持久化目录
└─ chat_history/                   # 会话 checkpoint（SQLite）目录


```

---

## 4. 环境要求

- Python 3.10+
- 已安装并运行 Ollama
- 可访问网络（仅在触发网页搜索时需要）

---

## 5. 安装与准备

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 准备 Ollama 模型

`rag_graph.py` 默认模型配置：

- LLM: `qwen3:14b`
- Embedding: `qwen3-embedding:latest`

请先拉取模型：

```bash
ollama pull qwen3:14b
ollama pull qwen3-embedding:latest
```

### 5.3 准备知识库数据

- 将你的资料放入 `my_knowledge/`。
    - 目前 支持的资料格式有：
      - "**/*.pdf"
      -  "**/*.docx"
      - "**/*.md"
-  运行`index_knowledge.py` 生成向量库并持久化到 `chroma_db/`。
  - ```bash
      python index_knowledge.py
    ```
- 确保 `chroma_db/` 中已有可用向量数据（当前主程序会直接读取持久化向量库）。

---

## 6. 运行方式

```bash
python rag_graph.py
```

程序启动后会进入交互流程：

1. 选择会话：
   - 输入已有 `session_id` 继续历史会话
   - 输入 `new` 或回车创建新会话
   - 输入 `list` 查看会话列表
2. 输入问题进行问答。
3. 进入人工审查：
   - `满意` / `approve` / `yes` / 直接回车：通过
   - `不满意: ...`/`dissatisfied`：给出修改意见并触发重写
   - `skip` / `跳过`：跳过审查
4. 输入 `exit` / `q` / `返回` 结束当前会话并保存总结。

---

## 7. 运行流程概览

```text
用户问题
  -> 本地检索(retrieve)
      -> (上下文不足) 网页搜索(web_search)
  -> 回答生成(answer)
  -> 人工审查(human_review)
      -> 满意: 结束或总结(summarize)
      -> 不满意: 反馈重写(feedback_node) -> 回到审查
```

---

## 8. 主要配置项（位于 rag_graph.py）

- `DATA_DIR`：本地知识目录（默认 `./my_knowledge`）
- `PERSIST_DIR`：向量库目录（默认 `./chroma_db`）
- `CHECKPOINT_FILE` / `CHECKPOINT_DB`：会话持久化目录与数据库文件
- `SESSIONS_FILE`：会话摘要索引文件（默认 `./sessions.json`）
- `LLM_MODEL`：默认 `qwen3:14b`
- `EMBEDDING_MODEL`：默认 `qwen3-embedding:latest`

---

## 9. 知识库内容概览

从当前 `my_knowledge/` 文件名可见，知识内容覆盖多个主题，例如：

- Python / Java / Linux / Git
- 机器学习、人工智能、Tensorflow/Keras
- 算法与 LeetCode
- 日语学习（语法、初中级笔记）
- 经济学/财报/读书笔记等

该结构非常适合作为个人学习型知识库进行长期积累。

---

## 10. 备注

- 若本地检索结果较少，系统会自动尝试联网搜索补充信息。

---

## License

当前仓库未显式提供许可证文件（`LICENSE`）。如需开源发布，建议补充许可证说明。
