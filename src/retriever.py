"""
检索模块：混合检索 = BM25 关键词 + 向量语义，双路召回 + 融合排序
"""

from typing import Optional
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from config import RETRIEVAL_TOP_K


class MedicalRetriever:
    """医学知识混合检索器"""

    def __init__(self, vectorstore: Chroma, top_k: int = RETRIEVAL_TOP_K):
        self.vectorstore = vectorstore
        self.top_k = top_k

        # 构建 BM25 索引（从 ChromaDB 提取所有文档）
        self._build_bm25_index()

    def _build_bm25_index(self):
        """从向量库提取文档，构建 BM25 关键词索引"""
        print("构建 BM25 关键词索引...")
        # 获取 ChromaDB 中所有文档
        collection = self.vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])

        self.bm25_docs = result["documents"] or []
        self.bm25_metadatas = result["metadatas"] or []

        # 分词（中文按字符 + jieba 风格分词）
        tokenized = [self._tokenize(doc) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        print(f"  BM25 索引: {len(self.bm25_docs)} 篇文档")

    def _tokenize(self, text: str) -> list[str]:
        """中文分词：按字/词切分"""
        # 简单但有效：按字切分 + 保留连续字母数字
        tokens = []
        buf = ""
        for ch in text:
            if ch.isalnum() or ch in "+-":
                buf += ch
            else:
                if buf:
                    tokens.append(buf.lower())
                    buf = ""
                if ch.strip() and not ch.isspace():
                    tokens.append(ch)
        if buf:
            tokens.append(buf.lower())
        return tokens

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        """BM25 关键词检索"""
        if not self.bm25:
            return []
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        # 取 top-k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:k]:
            if score > 0:  # 只保留有匹配的
                results.append({
                    "content": self.bm25_docs[idx],
                    "metadata": self.bm25_metadatas[idx] or {},
                    "score": float(score),
                    "source": "bm25",
                })
        return results

    def _vector_search(self, query: str, k: int) -> list[dict]:
        """向量语义检索"""
        raw = self.vectorstore.similarity_search_with_score(query, k=k)
        results = []
        for doc, score in raw:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),  # ChromaDB 距离，越小越相关
                "source": "vector",
            })
        return results

    def _reciprocal_rank_fusion(
        self, bm25_results: list[dict], vector_results: list[dict], k: int = 60
    ) -> list[dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合两个排序列表。
        无需调参，简单有效。
        """
        fused = {}
        for rank, r in enumerate(bm25_results, 1):
            cid = r["content"][:80]  # 用前80字符做 key
            fused[cid] = {"doc": r, "rrf": 1.0 / (k + rank)}

        for rank, r in enumerate(vector_results, 1):
            cid = r["content"][:80]
            if cid in fused:
                fused[cid]["rrf"] += 1.0 / (k + rank)
                # 保留向量结果的分数用于后续参考
                fused[cid]["doc"]["vector_score"] = r["score"]
            else:
                fused[cid] = {"doc": r, "rrf": 1.0 / (k + rank)}

        # 按 RRF 分数降序
        sorted_items = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)
        return [item["doc"] for item in sorted_items]

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        混合检索：BM25 + 向量 → RRF 融合。
        """
        k = top_k or self.top_k
        fetch_k = k * 2  # 每个路多召回一些

        bm25_results = self._bm25_search(query, fetch_k)
        vector_results = self._vector_search(query, fetch_k)

        # RRF 融合
        fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
        return fused[:k]

    def retrieve_as_context(self, query: str, top_k: int = None) -> str:
        """检索并格式化上下文，用于 Prompt"""
        docs = self.retrieve(query, top_k)
        if not docs:
            return "（未找到相关医学知识）"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc["metadata"].get("source", "未知")
            method = doc.get("source", "unknown")
            parts.append(
                f"[参考资料 {i}] (来源:{source} | 检索:{method})\n{doc['content']}"
            )

        return "\n\n" + "\n\n".join(parts)

    def get_source_references(self, query: str, top_k: int = None) -> list[dict]:
        """获取检索来源引用"""
        docs = self.retrieve(query, top_k)
        refs = []
        for doc in docs:
            meta = doc["metadata"]
            refs.append({
                "question": meta.get("question", ""),
                "source": meta.get("source", "未知"),
                "method": doc.get("source", "unknown"),
                "relevance": f"{1/(1+doc['score']):.2%}" if doc["score"] > 0 else "N/A"
            })
        return refs
