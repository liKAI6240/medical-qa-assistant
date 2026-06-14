"""
RAG 主流程：串联 检索 → 生成 完整链路
支持 RAG 增强回答和纯 LLM 回答的对比
"""

import time
from typing import Iterator

from build_kb import load_knowledge_base
from retriever import MedicalRetriever
from generator import generate_answer, generate_answer_stream, generate_without_rag


class MedicalRAGPipeline:
    """医学 RAG 问答系统主流程"""

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self._initialized = False

    def initialize(self) -> bool:
        """加载知识库并初始化检索器"""
        print("正在加载知识库...")
        self.vectorstore = load_knowledge_base()

        if self.vectorstore is None:
            print("⚠ 知识库未找到，请先运行 build_kb.py 构建知识库")
            return False

        self.retriever = MedicalRetriever(self.vectorstore)
        self._initialized = True
        print("✅ RAG Pipeline 初始化完成")
        return True

    def ask(self, question: str) -> dict:
        """
        RAG 增强问答（非流式）。

        Returns:
            {
                "question": 问题,
                "answer": 回答,
                "context": 使用的上下文,
                "references": 引用来源列表,
                "time_cost": 耗时秒数
            }
        """
        if not self._initialized:
            return {"error": "Pipeline 未初始化，请先调用 initialize()"}

        start = time.time()

        # 1. 检索
        context = self.retriever.retrieve_as_context(question)
        references = self.retriever.get_source_references(question)

        # 2. 生成
        answer = generate_answer(question, context)

        elapsed = time.time() - start

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "references": references,
            "time_cost": round(elapsed, 2),
        }

    def ask_stream(self, question: str) -> Iterator[dict]:
        """
        RAG 增强问答（流式）。

        Yields:
            {"type": "references", "data": [...]}   — 引用来源
            {"type": "answer", "data": "文本片段"}   — 回答内容片段
            {"type": "done", "data": {...}}          — 完成，附带元数据
        """
        if not self._initialized:
            yield {"type": "error", "data": "Pipeline 未初始化"}
            return

        start = time.time()

        # 1. 检索
        context = self.retriever.retrieve_as_context(question)
        references = self.retriever.get_source_references(question)
        yield {"type": "references", "data": references}

        # 2. 流式生成
        full_answer = ""
        for chunk in generate_answer_stream(question, context):
            full_answer += chunk
            yield {"type": "answer", "data": chunk}

        elapsed = time.time() - start
        yield {
            "type": "done",
            "data": {
                "question": question,
                "answer": full_answer,
                "context": context,
                "references": references,
                "time_cost": round(elapsed, 2),
            }
        }

    def ask_without_rag(self, question: str) -> str:
        """纯 LLM 回答（不经过 RAG），用于对比实验"""
        return generate_without_rag(question)


# ==================== 便捷函数 ====================

def create_pipeline() -> MedicalRAGPipeline:
    """一键创建并初始化 Pipeline"""
    pipeline = MedicalRAGPipeline()
    pipeline.initialize()
    return pipeline
