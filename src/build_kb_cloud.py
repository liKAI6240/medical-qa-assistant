"""
云端知识库构建脚本
================
在本地运行一次，生成精简版 ChromaDB 知识库，适合提交到 GitHub
并在 Streamlit Cloud 上直接加载使用。

特性：
- 使用 DashScope Embedding API（无需下载 1.3GB 本地模型）
- text-embedding-v3, 512 维向量，紧凑且高效
- 不分块：每条 QA 作为一个完整文档
- 目标：~8000 条 QA，KB 大小 ~40MB

用法：
    cd src
    python build_kb_cloud.py
"""

import os
import sys
import time
import shutil
import random

# ===== 在导入 config 之前，先加载 .streamlit/secrets.toml 到环境变量 =====
def _load_secrets_toml():
    """加载 .streamlit/secrets.toml 到 os.environ，供 config._get() 读取"""
    import tomllib  # Python 3.11+ 内置

    # 查找项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    secrets_path = os.path.join(project_root, ".streamlit", "secrets.toml")

    if os.path.exists(secrets_path):
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for key, value in secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
        print(f"[OK] 已加载 {secrets_path} ({len(secrets)} 项配置)")
    else:
        print(f"[WARN] 未找到 {secrets_path}，请确保 API Key 已配置")


_load_secrets_toml()

# 现在可以安全导入 config
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VECTOR_DB_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    EMBEDDING_BACKEND,
)
from build_kb import get_embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


# ==================== 配置 ====================

TARGET_COUNT = 8000            # 目标 QA 数量
MAX_ANSWER_LEN = 1500          # 答案最大字符数（超长截断）
COLLECTION_NAME = "medical_qa"


# ==================== 数据加载 ====================

def load_cleaned_qa() -> list[dict]:
    """加载已清洗的 QA 数据（来自 Huatuo-26M）"""
    csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_qa.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] 未找到 {csv_path}，跳过")
        return []

    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        source = str(row.get("source", "cleaned"))
        if q and a and len(q) >= 3 and len(a) >= 10:
            records.append({"question": q, "answer": a, "source": source})
    print(f"  从 cleaned_qa.csv 加载 {len(records)} 条")
    return records


def load_med_dialogue(sample_size: int = 3000) -> list[dict]:
    """
    从 Chinese Medical Dialogue Data 加载补充数据。
    数据包含 6 个科室的 CSV 文件（GBK 编码）。
    """
    base = os.path.join(RAW_DATA_DIR, "Chinese-medical-dialogue-data", "Data_数据")
    if not os.path.exists(base):
        print(f"  [WARN] 目录不存在: {base}，跳过")
        return []

    dept_dirs = {
        "Andriatria_男科": "男科",
        "IM_内科": "内科",
        "OAGD_妇产科": "妇产科",
        "Oncology_肿瘤科": "肿瘤科",
        "Pediatric_儿科": "儿科",
        "Surgical_外科": "外科",
    }

    # 计算每个科室应采样的数量
    per_dept = max(1, sample_size // len(dept_dirs))
    all_records = []

    for dir_name, dept_label in dept_dirs.items():
        dept_path = os.path.join(base, dir_name)
        if not os.path.exists(dept_path):
            continue

        csv_files = [f for f in os.listdir(dept_path) if f.endswith(".csv")]
        dept_records = []

        for csv_file in csv_files:
            filepath = os.path.join(dept_path, csv_file)
            try:
                # 尝试多种编码
                for enc in ["gbk", "gb18030", "utf-8"]:
                    try:
                        df = pd.read_csv(filepath, encoding=enc)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    continue

                # 字段名: ask/question/title → answer/reply
                q_col = next(
                    (c for c in df.columns if c in ("ask", "question", "title")),
                    df.columns[2] if len(df.columns) > 2 else df.columns[0],
                )
                a_col = next(
                    (c for c in df.columns if c in ("answer", "reply")),
                    df.columns[3] if len(df.columns) > 3 else df.columns[0],
                )

                for _, row in df.iterrows():
                    q = str(row.get(q_col, "")).strip()
                    a = str(row.get(a_col, "")).strip()
                    if q and a and len(q) >= 3 and len(a) >= 10:
                        dept_records.append({
                            "question": q,
                            "answer": a,
                            "source": f"med-dialogue/{dept_label}",
                        })
            except Exception:
                continue

        # 均衡采样
        if len(dept_records) > per_dept:
            random.seed(42)
            dept_records = random.sample(dept_records, per_dept)

        all_records.extend(dept_records)
        print(f"  {dept_label}: {len(dept_records)} 条")

    print(f"  从 Chinese Medical Dialogue Data 加载 {len(all_records)} 条")
    return all_records


# 云端 KB 使用独立子目录
CLOUD_KB_PATH = VECTOR_DB_PATH  # 保持原路径，先尝试关闭旧连接


# ==================== 主流程 ====================

def build_cloud_kb():
    """构建云端精简知识库"""

    # ---- 清理旧知识库 ----
    if os.path.exists(VECTOR_DB_PATH):
        print(f"清理旧知识库: {VECTOR_DB_PATH}")
        try:
            shutil.rmtree(VECTOR_DB_PATH, ignore_errors=True)
        except Exception as e:
            print(f"  [WARN] rmtree 异常（将尝试逐个删除）: {e}")
            for root, dirs, files in os.walk(VECTOR_DB_PATH, topdown=False):
                for f in files:
                    try:
                        os.unlink(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    # ---- 加载数据 ----
    print("\n>>> 加载数据...")
    records = load_cleaned_qa()

    # 如果不够 8000 条，从 Chinese Medical Dialogue 补充
    if len(records) < TARGET_COUNT:
        needed = TARGET_COUNT - len(records)
        print(f"  当前 {len(records)} 条，需补充 {needed} 条")
        supplement = load_med_dialogue(sample_size=needed + 500)
        records.extend(supplement)

    # 控制总量
    if len(records) > TARGET_COUNT:
        random.seed(42)
        records = random.sample(records, TARGET_COUNT)

    # 去重
    seen = set()
    unique_records = []
    for r in records:
        key = r["question"][:60]
        if key not in seen:
            seen.add(key)
            unique_records.append(r)
    records = unique_records

    print(f"\n[OK] 最终使用 {len(records)} 条 QA 对构建知识库")

    # ---- 构建 LangChain Documents（不分块） ----
    print("\n--构建文档（不分块）...")
    documents = []
    for r in records:
        # 截断过长答案
        answer = r["answer"][:MAX_ANSWER_LEN]
        content = f"【问题】{r['question']}\n【回答】{answer}"
        documents.append(Document(
            page_content=content,
            metadata={
                "question": r["question"],
                "answer": answer,
                "source": r["source"],
            },
        ))

    # ---- 向量化 ----
    print(f"\n[EMBED] 初始化 Embedding (后端: {EMBEDDING_BACKEND})...")
    embeddings = get_embeddings()

    print(f"向量化并写入 ChromaDB ({len(documents)} 个文档)...")
    batch_size = 20
    vectorstore = None

    for i in tqdm(range(0, len(documents), batch_size), desc="向量化批次"):
        batch = documents[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=VECTOR_DB_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)
        # 温和限速
        if i + batch_size < len(documents):
            time.sleep(0.3)

    # ---- 报告 ----
    collection = vectorstore._collection
    count = collection.count()
    print(f"\n[OK] 知识库构建完成！")
    print(f"   向量数量: {count}")
    print(f"   存储路径: {VECTOR_DB_PATH}")

    # 报告文件大小
    total_size = 0
    for root, dirs, files in os.walk(VECTOR_DB_PATH):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"   磁盘大小: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build_cloud_kb()
