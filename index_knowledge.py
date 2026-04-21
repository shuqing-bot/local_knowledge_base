import os
import json
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import hashlib

# ================== 配置（只需修改这里） ==================
DATA_DIR = "./my_knowledge"  # 你的 PDF 和 DOCX 文件夹
PERSIST_DIR = "./chroma_db"  # 向量库保存路径（本地）

LLM_MODEL = "qwen3.5:14b"  # 你下载的模型（仅作参考）
EMBEDDING_MODEL = "qwen3-embedding:latest"  # 嵌入模型，必须提前 pull
# 可以把这个缓存文件放在 DATA_DIR 外面或里面都行
CACHE_FILE = "loaded_docs_cache.json"


# ================== 1. 提取文档内容 ==================
def get_file_hash(file_path):
    """计算文件内容的 hash，用于判断是否变化"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_docs_incremental():
    print("正在加载 PDF、DOCX 和 MD 文件（增量模式）...")

    loaders = {
        "**/*.pdf": PyMuPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.md": UnstructuredMarkdownLoader,
    }

    # 加载上次的缓存（哪个文件上次加载过 + hash）
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    all_docs = []
    new_cache = {}

    for glob_pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(
            DATA_DIR,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=True
        )
        # 获取所有文件路径
        p = Path(loader.path)
        files = list(p.rglob(glob_pattern))

        for file_path in files:
            file_str = str(file_path)
            current_hash = get_file_hash(file_path)

            # 如果文件不在缓存 或 hash 变了 → 需要重新加载
            if file_str not in cache or cache[file_str] != current_hash:
                print(f"  → 新增/更新: {file_str}")
                single_loader = loader_cls(file_path)
                docs = single_loader.load()
                all_docs.extend(docs)
                new_cache[file_str] = current_hash
            else:
                print(f"  → 跳过（未变化）: {file_str}")
                new_cache[file_str] = current_hash  # 保留旧的

    # 保存新的缓存
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_cache, f, ensure_ascii=False, indent=2)

    print(f"✅ 加载完成！本次新增/更新 {len(all_docs)} 个文档片段")
    return all_docs
# ================== 2. 切块 ==================
def split_docs(documents):
    print("正在进行文本切块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 中文文档建议 600-1000
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"✅ 切块完成！共生成 {len(splits)} 个文本片段")
    return splits


# ================== 3. 向量化并保存到本地 ==================
def build_vectorstore(splits):
    print("正在生成向量并保存到本地 Chroma...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    os.makedirs(PERSIST_DIR,exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="my_local_knowledge"
    )
    print(f"🎉 知识库向量化完成！已保存到 {PERSIST_DIR}")
    return vectorstore


# ================== 主程序 ==================
if __name__ == "__main__":
    # 先确保嵌入模型已下载
    # 终端运行：ollama pull nomic-embed-text

    docs = load_docs_incremental()
    # splits = split_docs(docs)
    # vectorstore = build_vectorstore(splits)

    print("\n✅ 全部完成！你现在可以用这个向量库做检索了。")