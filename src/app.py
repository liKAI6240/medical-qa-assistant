"""
医学咨询助手 — Streamlit 前端界面
基于 RAG 的医学专业知识问答系统
"""

import sys
import os

# 确保项目根目录在 path 中（支持直接运行和模块导入）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_pipeline import create_pipeline, MedicalRAGPipeline
from generator import check_config


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="医学咨询助手",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 样式 ====================
st.markdown("""
<style>
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 16px 0;
        font-size: 0.9em;
        color: #856404;
    }
    .reference-card {
        background-color: #f0f8ff;
        border-left: 4px solid #4a90d9;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化 Session State ====================
def init_session():
    """初始化会话状态"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_loaded" not in st.session_state:
        st.session_state.kb_loaded = False


# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏配置面板"""
    with st.sidebar:
        st.title("🏥 医学咨询助手")
        st.markdown("基于 RAG 的医学专业知识问答系统")
        st.divider()

        # API 配置检查
        st.subheader("⚙️ 系统配置")
        if check_config():
            st.success("通义千问 API ✅ 已配置")
        else:
            st.error("通义千问 API ❌ 未配置")
            with st.expander("📝 如何配置？"):
                st.markdown("""
                1. 访问 [阿里云 DashScope](https://dashscope.console.aliyun.com/apiKey)
                2. 注册/登录后创建 API Key
                3. 填入 `config.py` 文件中的 `DASHSCOPE_API_KEY`
                """)

        # 知识库状态
        st.subheader("📚 知识库")
        if st.session_state.kb_loaded:
            st.success("知识库已加载")
        else:
            st.warning("知识库未加载")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 加载知识库", use_container_width=True):
                with st.spinner("加载中..."):
                    try:
                        st.session_state.pipeline = create_pipeline()
                        st.session_state.kb_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败: {e}")

        with col2:
            if st.button("🗑 清空对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # 模型参数
        st.divider()
        st.subheader("🎛️ 检索设置")
        top_k = st.slider("检索文档数", min_value=1, max_value=15, value=8, step=1)

        st.divider()
        st.subheader("💬 对话记忆")
        memory_turns = st.slider("记忆轮数", min_value=0, max_value=10, value=3, step=1,
                                 help="0=无记忆，每次独立问答")
        st.divider()
        st.subheader("📊 对比模式")
        show_comparison = st.checkbox("显示 RAG vs 纯LLM 对比", value=False)

        st.divider()
        st.markdown("""
        ### ⚠️ 免责声明
        本系统仅供学习研究使用，不构成医疗诊断或治疗建议。
        如有健康问题，请及时咨询专业医生。
        """)

        return top_k, show_comparison, memory_turns


# ==================== 主页面对话区 ====================
def render_chat(top_k: int, show_comparison: bool, memory_turns: int):
    """渲染对话界面"""
    st.title("🏥 医学咨询助手")

    # 未加载知识库的提示
    if not st.session_state.kb_loaded:
        st.info("👈 请先在左侧边栏点击 **「🔄 加载知识库」** 按钮", icon="ℹ️")
        st.markdown("""
        ### 快速开始
        1. 确保 `config.py` 中已配置通义千问 API Key
        2. 点击左侧「加载知识库」按钮
        3. 开始提问！
        """)
        return

    # 显示对话历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("references"):
                with st.expander("📚 参考来源"):
                    for ref in msg["references"]:
                        method_tag = f"🔤{ref.get('method','')}" if ref.get('method') else ""
                        st.markdown(
                            f"<div class='reference-card'>"
                            f"<strong>{ref['source']}</strong> {method_tag} | "
                            f"相关度 {ref['relevance']}<br>"
                            f"{ref['question'][:100]}..."
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # 输入框
    if question := st.chat_input("请输入您的健康问题...", key="chat_input"):
        # 构建对话记忆上下文
        context_question = question
        if memory_turns > 0:
            recent = st.session_state.messages[-(memory_turns * 2):]
            if recent:
                history_parts = ["[对话历史]"]
                for m in recent:
                    role = "用户" if m["role"] == "user" else "助手"
                    history_parts.append(f"{role}: {m['content'][:200]}")
                history_parts.append(f"用户: {question}")
                context_question = "\n".join(history_parts)

        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 生成回答
        pipeline: MedicalRAGPipeline = st.session_state.pipeline
        pipeline.retriever.top_k = top_k

        if show_comparison:
            # 对比模式：左右分栏
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔍 RAG 增强回答")
                with st.spinner("RAG 生成中..."):
                    result = pipeline.ask(context_question)
                if "error" in result:
                    st.error(result["error"])
                else:
                    rag_placeholder = st.empty()
                    rag_placeholder.markdown(result["answer"])
                    st.caption(f"⏱ 耗时 {result['time_cost']}s")
                    with st.expander("📚 参考来源"):
                        for ref in result["references"]:
                            st.markdown(
                                f"<div class='reference-card'>"
                                f"<strong>{ref['source']}</strong> | 相关度 {ref['relevance']}"
                                f"</div>",
                                unsafe_allow_html=True
                            )

            with col2:
                st.markdown("### 🤖 纯 LLM 回答")
                with st.spinner("纯 LLM 生成中..."):
                    pure_answer = pipeline.ask_without_rag(question)
                st.markdown(pure_answer)

            # 保存 RAG 回答到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**RAG 增强回答：**\n\n{result.get('answer', '')}",
                "references": result.get("references", []),
            })
        else:
            # 流式输出模式
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                refs_placeholder = st.empty()

                full_answer = ""
                references = []

                for chunk_data in pipeline.ask_stream(context_question):
                    if chunk_data["type"] == "references":
                        references = chunk_data["data"]
                        # 显示引用来源
                        ref_html = ""
                        for ref in references:
                            ref_html += (
                                f"<span class='reference-card'>"
                                f"📖 {ref['source']} | 相关度 {ref['relevance']}"
                                f"</span> "
                            )
                        refs_placeholder.markdown(ref_html, unsafe_allow_html=True)

                    elif chunk_data["type"] == "answer":
                        full_answer += chunk_data["data"]
                        response_placeholder.markdown(full_answer + "▌")

                    elif chunk_data["type"] == "done":
                        response_placeholder.markdown(full_answer)
                        elapsed = chunk_data["data"]["time_cost"]
                        st.caption(f"⏱ 耗时 {elapsed}s")

                        # 展开查看参考来源
                        with st.expander("📚 查看参考来源详情"):
                            for ref in chunk_data["data"]["references"]:
                                st.markdown(
                                    f"<div class='reference-card'>"
                                    f"<strong>来源:</strong> {ref['source']} | "
                                    f"<strong>相关度:</strong> {ref['relevance']}<br>"
                                    f"<strong>匹配问题:</strong> {ref['question'][:100]}..."
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

            # 保存到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "references": references,
            })

    # 底部免责声明
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <strong>免责声明：</strong>本回答仅基于医学知识库提供参考，不构成医疗诊断或治疗建议。"
        "如有健康问题，请及时咨询专业医生。"
        "</div>",
        unsafe_allow_html=True
    )


# ==================== 主函数 ====================
def main():
    init_session()
    top_k, show_comparison, memory_turns = render_sidebar()
    render_chat(top_k, show_comparison, memory_turns)


if __name__ == "__main__":
    main()
