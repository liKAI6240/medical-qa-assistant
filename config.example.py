# ==================== 通义千问 (DashScope) API 配置 ====================
# 获取地址: https://dashscope.console.aliyun.com/apiKey
DASHSCOPE_API_KEY = "your_api_key_here"  # 获取地址: https://dashscope.console.aliyun.com/apiKey

# 模型配置 (通义千问免费模型)
LLM_MODEL = "qwen-turbo"           # 免费额度: 200万 tokens/月
# LLM_MODEL = "qwen-plus"          # 更强的模型，有免费额度
# LLM_MODEL = "qwen-max"           # 最强模型

# ==================== Embedding 配置 ====================
# 使用本地模型，无需 API
EMBEDDING_MODEL_NAME = "GanymedeNil/text2vec-large-chinese"

# ==================== 知识库配置 ====================
CHUNK_SIZE = 512                       # 文本分块大小
CHUNK_OVERLAP = 50                     # 分块重叠长度
RETRIEVAL_TOP_K = 5                    # 检索返回的文档数量
VECTOR_DB_PATH = "./knowledge_base"    # 向量数据库持久化路径

# ==================== 数据配置 ====================
DATA_DIR = "./data"
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
