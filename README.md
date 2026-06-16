# 🏥 医学咨询助手 — Medical QA Assistant

基于 **RAG（检索增强生成）** 的医学专业知识问答系统。

🌐 **在线使用**: [https://medical-qa-assistant.streamlit.app](https://medical-qa-assistant.streamlit.app)  
📦 **源码**: [https://github.com/liKAI6240/medical-qa-assistant](https://github.com/liKAI6240/medical-qa-assistant)

## 技术栈

- **LLM**: 阿里云百炼（通义千问）— 免费额度 200万 tokens/月
- **向量数据库**: ChromaDB
- **框架**: LangChain
- **前端**: Streamlit
- **Embedding**: 百炼 Embedding API（云端）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

创建 `.streamlit/secrets.toml`，填入：

```toml
DASHSCOPE_API_KEY = "your_api_key_here"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-turbo"
EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIMS = "512"
EMBEDDING_BACKEND = "dashscope"
```

获取 Key：https://bailian.console.aliyun.com/

### 3. 构建知识库

```bash
cd src
python build_kb_cloud.py
```

### 4. 启动 Web 界面

```bash
cd src
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 一键部署到 Streamlit Cloud

点击下方按钮一键部署（需 GitHub 账号）：

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=liKAI6240/medical-qa-assistant&branch=main&mainModule=src/app.py)

部署后在 Streamlit Cloud Dashboard → Settings → Secrets 中配置 API Key（同上方步骤 2）。

## 项目结构

```
medical-qa-assistant/
├── config.py              # 全局配置（自动从 secrets/env 读取）
├── config.example.py      # 配置模板（供参考）
├── requirements.txt       # Python 依赖
├── README.md
├── data/                  # 数据目录
│   ├── raw/               # 原始数据集
│   └── processed/         # 清洗后的数据
├── knowledge_base/        # ChromaDB 向量存储（需构建）
└── src/
    ├── app.py             # Streamlit Web 前端
    ├── data_loader.py     # 数据加载 (Huatuo-26M, efaqa, PsyQA)
    ├── preprocess.py      # 数据清洗与预处理
    ├── build_kb.py        # 知识库构建（支持本地/API 双后端）
    ├── build_kb_cloud.py  # 云端 KB 构建脚本（API embedding）
    ├── embeddings.py      # DashScope Embedding API 封装
    ├── retriever.py       # 混合检索（BM25 + 向量 + RRF 融合）
    ├── generator.py       # LLM 生成模块
    └── rag_pipeline.py    # RAG 主流程
```

## 数据来源

| 数据集 | 来源 | 说明 |
|--------|------|------|
| Huatuo-26M | [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/Huatuo-26M) | 大规模中文医学对话 |
| Chinese Medical Dialogue | [GitHub](https://github.com/Toyhom/Chinese-medical-dialogue-data) | 6科室医学问答 |
| efaqa-corpus-raw | [GitHub](https://github.com/chatopera/efaqa-corpus-raw) | 医疗问答语料 |
| PsyQA | [GitHub](https://github.com/thu-coai/PsyQA) | 心理咨询问答 |
