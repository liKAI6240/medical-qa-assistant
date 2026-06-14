# 🏥 医学咨询助手 — Medical QA Assistant

基于 **RAG（检索增强生成）** 的医学专业知识问答系统。

## 技术栈

- **LLM**: 百度千帆 ERNIE (免费 Tokens)
- **向量数据库**: ChromaDB
- **框架**: LangChain
- **前端**: Streamlit
- **Embedding**: 百度千帆 Embedding API / text2vec-large-chinese (本地)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Keys

编辑 `config.py`，填入百度千帆的 Access Key 和 Secret Key：

```python
QIANFAN_AK = "your_access_key"
QIANFAN_SK = "your_secret_key"
```

获取地址：https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application

### 3. 下载数据集

```bash
# 方式A: 通过 HuggingFace (Huatuo-26M, 在线加载)
# 无需手动下载，data_loader.py 会自动处理

# 方式B: 手动克隆 (efaqa, PsyQA)
git clone https://github.com/chatopera/efaqa-corpus-raw.git data/raw/efaqa-corpus-raw
git clone https://github.com/thu-coai/PsyQA.git data/raw/PsyQA
```

### 4. 构建知识库

```bash
cd src
python -c "
from data_loader import load_all_data
from preprocess import preprocess_pipeline
from build_kb import build_knowledge_base

# 加载数据
df = load_all_data(huatuo_sample=10000)

# 预处理
documents = preprocess_pipeline(df)

# 构建向量知识库
build_knowledge_base(documents, use_local_embedding=False)
"
```

### 5. 启动 Web 界面

```bash
cd src
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 项目结构

```
medical-qa-assistant/
├── config.py              # API Keys 与配置
├── requirements.txt       # Python 依赖
├── README.md
├── data/                  # 数据目录
│   ├── raw/               # 原始数据集
│   └── processed/         # 清洗后的数据
├── knowledge_base/        # ChromaDB 向量存储
└── src/
    ├── app.py             # Streamlit Web 前端
    ├── data_loader.py     # 数据加载 (Huatuo-26M, efaqa, PsyQA)
    ├── preprocess.py      # 数据清洗与预处理
    ├── build_kb.py        # 知识库构建 (分块 + 向量化)
    ├── retriever.py       # 向量检索模块
    ├── generator.py       # LLM 生成模块
    └── rag_pipeline.py    # RAG 主流程
```

## 数据来源

| 数据集 | 来源 | 说明 |
|--------|------|------|
| Huatuo-26M | [HuggingFace](https://huggingface.co/datasets/FreedomIntelligence/Huatuo-26M) | 大规模中文医学对话 |
| efaqa-corpus-raw | [GitHub](https://github.com/chatopera/efaqa-corpus-raw) | 医疗问答语料 |
| PsyQA | [GitHub](https://github.com/thu-coai/PsyQA) | 心理咨询问答 |
