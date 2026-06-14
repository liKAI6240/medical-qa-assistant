"""
知识库构建模块：文本分块 → Embedding向量化 → 存入ChromaDB
使用本地 text2vec-large-chinese 模型（离线方案）
"""

import os
import time
from typing import Optional
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_PATH,
    EMBEDDING_BACKEND, EMBEDDING_MODEL, EMBEDDING_DIMS,
    LOCAL_EMBEDDING_MODEL,
)


def get_local_embeddings():
    """获取本地 Embedding 实例（基于 ONNX 运行时，无需 PyTorch，~100MB）"""
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            max_length=512,
        )
    except ImportError:
        raise ImportError("请安装 fastembed: pip install fastembed")


def get_dashscope_embeddings():
    """获取 DashScope/百炼 Embedding API 实例（无需下载模型）"""
    from embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS,
    )


def get_embeddings():
    """根据配置返回对应的 Embedding 实例"""
    if EMBEDDING_BACKEND == "dashscope":
        print("使用 DashScope Embedding API（无需本地模型）")
        return get_dashscope_embeddings()
    else:
        print("使用本地 text2vec-large-chinese 模型")
        return get_local_embeddings()


def create_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """创建中文友好的文本分割器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )


def build_knowledge_base(
    documents: list[dict],
    collection_name: str = "medical_qa",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Chroma:
    """构建向量知识库"""
    print(f"开始构建知识库 (embedding 后端: {EMBEDDING_BACKEND})...")

    # 1. 文本分块
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    print("1. 文本分块...")
    from langchain_core.documents import Document
    langchain_docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in documents
    ]
    split_docs = text_splitter.split_documents(langchain_docs)
    print(f"   原始文档 {len(documents)} → 分割后 {len(split_docs)} 个块")

    # 2. 初始化 Embedding
    print("2. 初始化 Embedding 模型...")
    embeddings = get_embeddings()

    # 3. 向量化并存入 ChromaDB
    print("3. 向量化并存入 ChromaDB...")
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    batch_size = 50
    for i in tqdm(range(0, len(split_docs), batch_size), desc="向量化批次"):
        batch = split_docs[i:i + batch_size]
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embeddings,
                persist_directory=VECTOR_DB_PATH, collection_name=collection_name,
            )
        else:
            vectorstore.add_documents(batch)
        if i + batch_size < len(split_docs):
            time.sleep(0.3)

    print(f"✅ 知识库构建完成！共 {len(split_docs)} 个向量块")
    return vectorstore


def load_knowledge_base(
    collection_name: str = "medical_qa",
) -> Optional[Chroma]:
    """加载已持久化的知识库"""
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"⚠ 知识库路径不存在: {VECTOR_DB_PATH}")
        return None

    embeddings = get_embeddings()
    try:
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        print(f"✅ 已加载知识库: {vectorstore._collection.count()} 个向量")
        return vectorstore
    except Exception as e:
        print(f"⚠ 加载失败: {e}")
        return None
