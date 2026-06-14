"""
生成模块：调用通义千问 (DashScope) LLM 进行医学问答生成
通过 OpenAI 兼容接口调用，API 格式标准、简单
"""

import time
from typing import Iterator
from openai import OpenAI
from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL


# ==================== DashScope 客户端 ====================


def _get_client() -> OpenAI:
    """获取 DashScope OpenAI 兼容客户端"""
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        timeout=30.0,
        max_retries=2,
    )


# ==================== Prompt 模板 ====================

SYSTEM_PROMPT = """你是一个专业的医学咨询助手，基于提供的医学知识库来回答用户的健康问题。

重要规则：
1. 请严格依据【参考资料】中的内容回答问题。如果参考资料中没有相关信息，请如实说明"参考资料中未找到相关信息"。
2. 请用专业但通俗易懂的语言回答，适当使用分点说明。
3. 回答时请注明引用的参考资料编号（如 [参考资料 1]）。
4. 如果用户的问题涉及紧急情况，请建议立即就医。
5. 在回答末尾添加免责声明：【免责声明】本回答仅基于医学知识库提供参考，不构成医疗诊断或治疗建议。如有健康问题，请及时咨询专业医生。"""


def build_prompt(question: str, context: str) -> str:
    """构建完整的 Prompt"""
    return f"""【用户问题】
{question}

{context}

请基于以上参考资料回答用户的问题。"""


def check_config() -> bool:
    """检查 API 配置是否有效"""
    return DASHSCOPE_API_KEY not in ("your_api_key_here", "")


# ==================== 生成函数 ====================

def generate_answer(question: str, context: str) -> str:
    """
    非流式生成回答。
    """
    if not check_config():
        raise ValueError(
            "请先在 config.py 中配置通义千问 API Key。\n"
            "获取地址: https://dashscope.console.aliyun.com/apiKey"
        )

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, context)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def generate_answer_stream(question: str, context: str) -> Iterator[str]:
    """
    流式生成回答，适合在前端逐字展示。
    """
    if not check_config():
        yield "⚠ 请先在 config.py 中配置通义千问 API Key。\n获取地址: https://dashscope.console.aliyun.com/apiKey"
        return

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, context)},
            ],
            temperature=0.3,
            max_tokens=1024,
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n\n⚠ 生成过程出错: {str(e)}"


def generate_without_rag(question: str) -> str:
    """
    不使用 RAG 的纯 LLM 回答（用于对比实验）。
    """
    if not check_config():
        raise ValueError("请先配置 API Key")

    client = _get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个专业的医学咨询助手，请根据你的医学知识回答问题。"},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
