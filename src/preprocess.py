"""
数据预处理模块：清洗、去重、格式标准化
将原始问答数据处理为适用于RAG的统一格式
"""

import os
import re
import pandas as pd
from config import PROCESSED_DATA_DIR


def clean_text(text: str) -> str:
    """清洗文本：去HTML标签、多余空白、特殊字符标准化"""
    if not isinstance(text, str):
        return ""
    # 去除HTML标签
    text = re.sub(r"<[^>]+>", " ", text)
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 将连续空白符压缩为单个空格
    text = re.sub(r"\s+", " ", text)
    # 去除首尾空白
    text = text.strip()
    return text


def filter_low_quality(df: pd.DataFrame) -> pd.DataFrame:
    """过滤低质量数据"""
    initial_count = len(df)

    # 1. 去除问题和答案为空的行（放宽阈值，保留更多数据）
    df = df[df["question"].str.len() >= 2]
    df = df[df["answer"].str.len() >= 3]

    # 2. 去除全是标点/数字的行
    df = df[~df["question"].str.match(r"^[\d\W_]+$")]

    # 3. 限制长度
    df = df[df["question"].str.len() <= 500]
    df = df[df["answer"].str.len() <= 5000]

    removed = initial_count - len(df)
    print(f"  过滤低质量数据: 移除 {removed} 条, 剩余 {len(df)} 条")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """去重：按问题去重，保留最长的答案"""
    before = len(df)
    df = df.sort_values("answer", key=lambda x: x.str.len(), ascending=False)
    df = df.drop_duplicates(subset=["question"], keep="first")
    after = len(df)
    print(f"  去重: 移除 {before - after} 条, 剩余 {after} 条")
    return df


def build_rag_documents(df: pd.DataFrame) -> list[dict]:
    """
    构建 RAG 文档格式。
    每条记录构造为一个文档，内容为 "问题: xxx\n回答: xxx"，
    或仅保留回答作为检索内容（问题用于匹配相似度）。

    同时保存多种格式便于检索实验：
    - full: "问题 + 回答" 拼接
    - answer_only: 仅回答
    """
    documents = []
    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]
        source = row.get("source", "unknown")

        # 格式1: 完整问答对（推荐用于检索）
        doc_full = f"【问题】{q}\n【回答】{a}"
        documents.append({
            "page_content": doc_full,
            "metadata": {
                "question": q,
                "answer": a,
                "source": source,
            }
        })
    print(f"  构建了 {len(documents)} 个 RAG 文档")
    return documents


def preprocess_pipeline(df: pd.DataFrame) -> list[dict]:
    """完整的预处理流水线"""
    print("开始数据预处理...")

    # 1. 文本清洗
    print("1. 文本清洗...")
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    # 2. 过滤低质量数据
    print("2. 过滤低质量...")
    df = filter_low_quality(df)

    # 3. 去重
    print("3. 去重...")
    df = deduplicate(df)

    # 4. 保存处理后的数据
    csv_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_qa.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"4. 已保存清洗后数据到 {csv_path}")

    # 5. 构建 RAG 文档
    documents = build_rag_documents(df)

    print(f"✅ 预处理完成: {len(documents)} 条可用于知识库")
    return documents


if __name__ == "__main__":
    # 测试：创建示例数据进行预处理
    test_df = pd.DataFrame([
        {"question": "  <p>高血压需要注意什么？</p>  ", "answer": "高血压患者应该注意饮食...", "source": "test"},
        {"question": "高血压需要注意什么？", "answer": "短", "source": "test"},  # 会被过滤
    ])
    docs = preprocess_pipeline(test_df)
    print(f"\n示例文档:")
    print(docs[0]["page_content"][:200])
