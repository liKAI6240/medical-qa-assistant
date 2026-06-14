# ==================== 百炼 (DashScope) API 配置 ====================
# 获取地址: https://bailian.console.aliyun.com/
#
# 配置方式（三选一，优先级从高到低）：
#   1. Streamlit Secrets   — 云端部署 / 本地 .streamlit/secrets.toml
#   2. 环境变量             — export DASHSCOPE_API_KEY=xxx
#   3. 修改本文件默认值     — 不推荐（容易泄露到 git）
#
DASHSCOPE_API_KEY = "your_api_key_here"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 模型配置 (通义千问 / 百炼)
LLM_MODEL = "qwen-turbo"           # 免费额度: 200万 tokens/月
# LLM_MODEL = "qwen-plus"          # 更强的模型
# LLM_MODEL = "qwen-max"           # 最强模型

# ==================== Embedding 配置 ====================
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIMS = 512
EMBEDDING_BACKEND = "local"        # "local" (本地模型) | "dashscope" (API，推荐云端)

# 本地 embedding 模型名（EMBEDDING_BACKEND="local" 时使用）
LOCAL_EMBEDDING_MODEL = "GanymedeNil/text2vec-large-chinese"

# ==================== 知识库配置 ====================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 5
VECTOR_DB_PATH = "./knowledge_base"

# ==================== 数据配置 ====================
DATA_DIR = "./data"
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
