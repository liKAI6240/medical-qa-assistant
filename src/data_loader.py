"""
数据加载模块：下载和加载医学问答数据集
数据源：
  - Huatuo-26M (HuggingFace): 大规模中文医学对话
  - efaqa-corpus-raw (GitHub): 医疗问答语料库
  - PsyQA (GitHub): 心理咨询问答数据集
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_huatuo_26m(sample_size: int = None) -> list[dict]:
    """
    从 HuggingFace 加载 Huatuo-26M 数据集（已拆分为多个子集）。

    子数据集：
      - huatuo_encyclopedia_qa: 医学百科问答
      - huatuo_knowledge_graph_qa: 知识图谱问答
      - huatuo_consultation_qa: 在线问诊问答

    Args:
        sample_size: 可选，每个子集的采样数量

    Returns:
        list[dict]: [{"question": "问题", "answer": "回答"}, ...]
    """
    from datasets import load_dataset

    sub_datasets = [
        "FreedomIntelligence/huatuo_encyclopedia_qa",
        "FreedomIntelligence/huatuo_knowledge_graph_qa",
        "FreedomIntelligence/huatuo_consultation_qa",
    ]

    all_records = []

    for ds_path in sub_datasets:
        ds_name = ds_path.split("/")[-1]
        print(f"  加载 {ds_name}...")
        try:
            dataset = load_dataset(ds_path, split="train")
            # 采样
            n_total = len(dataset)
            per_ds = min(sample_size or n_total, n_total)
            if per_ds < n_total:
                dataset = dataset.shuffle(seed=42).select(range(per_ds))

            for item in dataset:
                # 实际字段名: questions (list of lists!), answers (list or str)
                q_raw = (item.get("questions") or item.get("input") or
                         item.get("question") or item.get("query") or [])
                a_raw = (item.get("answers") or item.get("output") or
                         item.get("answer") or item.get("response") or "")

                # questions 可能是嵌套列表 [["问题"]]，需要解包
                if isinstance(q_raw, list):
                    q = q_raw[0] if q_raw else ""
                    # 处理嵌套列表 [["问题1", "问题2"]]
                    if isinstance(q, list):
                        q = q[0] if q else ""
                else:
                    q = str(q_raw) if q_raw else ""

                # answers 可能是列表
                if isinstance(a_raw, list):
                    a = a_raw[0] if a_raw else ""
                    if isinstance(a, list):
                        a = a[0] if a else ""
                else:
                    a = str(a_raw) if a_raw else ""

                q, a = str(q).strip(), str(a).strip()
                if q and a and len(q) >= 3 and len(a) >= 5:
                    all_records.append({"question": q, "answer": a})

            print(f"    → {len(all_records)} 条 (累计)")
        except Exception as e:
            print(f"    ⚠ {ds_name} 加载失败: {e}")
            continue

    if not all_records:
        print("  ⚠ 所有 Huatuo 子集加载失败，将使用内置示例数据")
        all_records = _generate_sample_qa()

    print(f"  → Huatuo 总计: {len(all_records)} 条有效记录")
    return all_records


def _generate_sample_qa() -> list[dict]:
    """从本地 JSON 文件加载示例医学问答数据"""
    json_path = os.path.join(PROCESSED_DATA_DIR, "sample_qa.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"  → 从 {json_path} 加载了 {len(samples)} 条示例数据")
        return samples
    print("  → 无示例数据文件，返回空列表")
    return []


def load_efaqa() -> list[dict]:
    """
    加载 efaqa-corpus-raw 医疗问答语料。
    数据格式：CSV/JSON，字段包括 title, question, answer 等。
    需要手动 git clone 到 data/raw/ 目录。

    下载命令：
      git clone https://github.com/chatopera/efaqa-corpus-raw.git data/raw/efaqa-corpus-raw
    """
    print("正在加载 efaqa-corpus-raw...")
    records = []

    local_dir = os.path.join(RAW_DATA_DIR, "efaqa-corpus-raw")
    if not os.path.exists(local_dir):
        print(f"  ⚠ 目录不存在: {local_dir}")
        print("  请运行: git clone https://github.com/chatopera/efaqa-corpus-raw.git data/raw/efaqa-corpus-raw")
        return records

    # 遍历所有 JSON 文件
    json_files = []
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".json"):
                json_files.append(os.path.join(root, f))

    for filepath in tqdm(json_files, desc="处理 efaqa 文件"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 尝试多种可能的字段名
            if isinstance(data, list):
                for item in data:
                    q = item.get("question") or item.get("title") or item.get("query") or ""
                    a = item.get("answer") or item.get("reply") or item.get("content") or ""
                    q, a = str(q).strip(), str(a).strip()
                    if q and a and len(q) > 2 and len(a) > 5:
                        records.append({"question": q, "answer": a})
            elif isinstance(data, dict):
                for key in data:
                    item = data[key] if isinstance(data[key], dict) else None
                    if item:
                        q = item.get("question") or item.get("title") or ""
                        a = item.get("answer") or item.get("reply") or ""
                        q, a = str(q).strip(), str(a).strip()
                        if q and a:
                            records.append({"question": q, "answer": a})
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    print(f"  → 加载了 {len(records)} 条 efaqa 记录")
    return records


def load_psyqa(sample_size: int = None) -> list[dict]:
    """
    加载 PsyQA 心理咨询问答数据集。

    下载命令：
      git clone https://github.com/thu-coai/PsyQA.git data/raw/PsyQA
    """
    print("正在加载 PsyQA 数据集...")
    records = []

    local_dir = os.path.join(RAW_DATA_DIR, "PsyQA")
    if not os.path.exists(local_dir):
        print(f"  ⚠ 目录不存在: {local_dir}")
        print("  请运行: git clone https://github.com/thu-coai/PsyQA.git data/raw/PsyQA")
        return records

    # PsyQA 的结构：train.json / valid.json / test.json
    for split_file in ["train.json", "valid.json", "test.json"]:
        filepath = os.path.join(local_dir, split_file)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    q = item.get("question") or item.get("description") or ""
                    # PsyQA 的 answer 通常在 answers 数组里
                    answers = item.get("answers", [])
                    a = answers[0].get("answer_text", "") if answers else item.get("answer", "")
                    q, a = str(q).strip(), str(a).strip()
                    if q and a:
                        records.append({"question": q, "answer": a})
            except (json.JSONDecodeError, FileNotFoundError):
                continue

    if sample_size and sample_size < len(records):
        records = records[:sample_size]

    print(f"  → 加载了 {len(records)} 条 PsyQA 记录")
    return records


def load_huatuo_lite(min_score: int = 4, sample_size: int = None) -> list[dict]:
    """
    加载 Huatuo26M-Lite（精炼版），带评分和科室标签。

    Args:
        min_score: 最低评分（1-5），只保留高质量回答
        sample_size: 采样数量
    """
    print("正在加载 Huatuo26M-Lite...")
    try:
        from datasets import load_dataset
        ds = load_dataset("FreedomIntelligence/Huatuo26M-Lite", split="train")
        records = []
        for item in ds:
            score = item.get("score", 0)
            if score >= min_score:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                label = item.get("label", "")
                if q and a and len(q) >= 3 and len(a) >= 10:
                    records.append({
                        "question": q,
                        "answer": a,
                        "department": label,
                        "score": score,
                    })
        if sample_size and sample_size < len(records):
            import random; random.shuffle(records)
            records = records[:sample_size]
        print(f"  → {len(records)} 条 (评分>={min_score})")
        return records
    except Exception as e:
        print(f"  ⚠ Huatuo26M-Lite 加载失败: {e}")
        return []


def load_medical_dialogue(sample_per_dept: int = 5000) -> list[dict]:
    """
    加载 Chinese Medical Dialogue Data（6个科室，约79万条）。

    数据路径: data/raw/Chinese-medical-dialogue-data/Data_数据/

    Args:
        sample_per_dept: 每个科室采样数量
    """
    print("正在加载 Chinese Medical Dialogue Data...")
    base = os.path.join(RAW_DATA_DIR, "Chinese-medical-dialogue-data", "Data_数据")
    if not os.path.exists(base):
        print(f"  ⚠ 目录不存在: {base}")
        print("  请运行: git clone https://github.com/Toyhom/Chinese-medical-dialogue-data.git data/raw/Chinese-medical-dialogue-data")
        return []

    dept_dirs = {
        "Andriatria_男科": "男科",
        "IM_内科": "内科",
        "OAGD_妇产科": "妇产科",
        "Oncology_肿瘤科": "肿瘤科",
        "Pediatric_儿科": "儿科",
        "Surgical_外科": "外科",
    }

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
                    except UnicodeDecodeError:
                        continue
                else:
                    continue
                # 字段名: department, title, ask, answer
                q_col = next((c for c in df.columns if c in ("ask", "question", "title")), df.columns[2] if len(df.columns) > 2 else df.columns[0])
                a_col = next((c for c in df.columns if c in ("answer", "reply")), df.columns[3] if len(df.columns) > 3 else df.columns[0])
                for _, row in df.iterrows():
                    q = str(row.get(q_col, "")).strip()
                    a = str(row.get(a_col, "")).strip()
                    if q and a and len(q) >= 3 and len(a) >= 10:
                        dept_records.append({
                            "question": q,
                            "answer": a,
                            "department": dept_label,
                        })
            except Exception:
                continue

        # 采样
        if sample_per_dept and len(dept_records) > sample_per_dept:
            import random; random.shuffle(dept_records)
            dept_records = dept_records[:sample_per_dept]

        all_records.extend(dept_records)
        print(f"  {dept_label}: {len(dept_records)} 条")

    print(f"  → 总计 {len(all_records)} 条")
    return all_records


def load_all_data(huatuo_sample: int = 30000) -> pd.DataFrame:
    """
    加载所有数据源，合并为统一的 DataFrame。

    Args:
        huatuo_sample: Huatuo-26M 采样数量（全量太大）

    Returns:
        pd.DataFrame: 包含 question, answer, source 三列
    """
    all_records = []

    # 1. Huatuo-26M（三个子集：encyclopedia + knowledge_graph + consultation）
    huatuo = load_huatuo_26m(sample_size=huatuo_sample)
    for r in huatuo:
        r["source"] = "huatuo-26m"
    all_records.extend(huatuo)

    # 2. Huatuo26M-Lite（精炼版，只取评分>=4的高质量数据）
    huatuo_lite = load_huatuo_lite(min_score=4, sample_size=30000)
    for r in huatuo_lite:
        r["source"] = "huatuo-lite"
    all_records.extend(huatuo_lite)

    # 3. Chinese Medical Dialogue Data（6个科室）
    med_dialogue = load_medical_dialogue(sample_per_dept=5000)
    for r in med_dialogue:
        r["source"] = "med-dialogue"
    all_records.extend(med_dialogue)

    # 4. efaqa
    efaqa = load_efaqa()
    for r in efaqa:
        r["source"] = "efaqa"
    all_records.extend(efaqa)

    # 5. PsyQA
    psyqa = load_psyqa()
    for r in psyqa:
        r["source"] = "psyqa"
    all_records.extend(psyqa)

    df = pd.DataFrame(all_records)
    df = df.drop_duplicates(subset=["question"])
    df = df.reset_index(drop=True)

    print(f"\n✅ 总计加载 {len(df)} 条唯一问答对")
    for src in df["source"].unique():
        print(f"   {src}: {len(df[df['source']==src])} 条")

    return df


if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df = load_all_data(huatuo_sample=1000)  # 小样本测试
    print(df.head())


if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df = load_all_data(huatuo_sample=1000)  # 小样本测试
    print(df.head())
