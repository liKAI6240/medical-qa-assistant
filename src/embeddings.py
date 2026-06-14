"""
DashScope / 百炼 Embedding API 模块
通过 OpenAI 兼容接口调用，无需下载本地模型（~1.3GB），
适合 Streamlit Cloud 等资源受限环境。
"""

import time
from typing import Optional

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMS


class DashScopeEmbeddings(Embeddings):
    """
    百炼 DashScope Embedding API 封装，符合 LangChain Embeddings 接口。

    使用 text-embedding-v3 模型，支持指定维度（默认 512），
    在保持检索质量的同时大幅减小向量体积。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 10,
    ):
        self._model = model or EMBEDDING_MODEL
        self._dimensions = dimensions or EMBEDDING_DIMS
        self._batch_size = batch_size

        self._client = OpenAI(
            api_key=api_key or DASHSCOPE_API_KEY,
            base_url=base_url or DASHSCOPE_BASE_URL,
            timeout=60.0,
            max_retries=3,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量向量化文档。

        Args:
            texts: 文本列表

        Returns:
            等长的向量列表，每个向量为 float 列表
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            try:
                # 注意：百炼 API 不支持 dimensions 参数，直接调用即可
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
                # 按 index 排序确保顺序
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                batch_embeddings = [d.embedding for d in sorted_data]
                all_embeddings.extend(batch_embeddings)

                # 温和限速，避免触发 API 频率限制
                if i + self._batch_size < len(texts):
                    time.sleep(0.2)

            except Exception as e:
                # 重试一次（单次调用）
                print(f"  ⚠ Embedding 批次 {i // self._batch_size} 失败: {e}，重试...")
                try:
                    time.sleep(1.0)
                    resp = self._client.embeddings.create(
                        model=self._model,
                        input=batch,
                    )
                    sorted_data = sorted(resp.data, key=lambda x: x.index)
                    batch_embeddings = [d.embedding for d in sorted_data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e2:
                    raise RuntimeError(
                        f"Embedding API 调用失败（已重试）: {e2}\n"
                        f"请检查 API Key 和网络连接。"
                    )

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询文本"""
        embeddings = self.embed_documents([text])
        return embeddings[0]
