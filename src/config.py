"""
医学咨询助手 — 全局配置
配置读取优先级: Streamlit Secrets → 环境变量 → 默认值
"""

import os

# 项目根目录（config.py 在 src/ 下，项目根是上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get(key: str, default=None):
    """三级配置读取：Streamlit Secrets → 环境变量 → 默认值"""
    # 1. Streamlit Secrets（云端部署 + 本地 .streamlit/secrets.toml）
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # 2. 环境变量
    env_val = os.environ.get(key)
    if env_val is not None:
        return env_val
    # 3. 默认值
    return default


# ==================== 百炼 (DashScope) API 配置 ====================
DASHSCOPE_API_KEY = _get("DASHSCOPE_API_KEY", "your_api_key_here")
DASHSCOPE_BASE_URL = _get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 模型配置 (通义千问 / 百炼)
LLM_MODEL = _get("LLM_MODEL", "qwen-turbo")
# 可选: qwen-plus, qwen-max

# ==================== Embedding 配置 ====================
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_DIMS = int(_get("EMBEDDING_DIMS", "512"))
EMBEDDING_BACKEND = _get("EMBEDDING_BACKEND", "dashscope")  # "local" | "dashscope"

# 本地 embedding 模型名（EMBEDDING_BACKEND="local" 时使用）
LOCAL_EMBEDDING_MODEL = _get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ==================== 知识库配置 ====================
CHUNK_SIZE = int(_get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
RETRIEVAL_TOP_K = int(_get("RETRIEVAL_TOP_K", "5"))
VECTOR_DB_PATH = _get("VECTOR_DB_PATH", os.path.join(_PROJECT_ROOT, "knowledge_base"))

# ==================== 数据配置 ====================
DATA_DIR = _get("DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
RAW_DATA_DIR = _get("RAW_DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "raw"))
PROCESSED_DATA_DIR = _get("PROCESSED_DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "processed"))
