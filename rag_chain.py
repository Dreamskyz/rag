"""
RAG 问答链模块

将检索到的上下文与用户问题组合，调用 Gemini 生成回答。
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME

# 复用 LLM 实例，避免每次问答重复初始化
_llm_instance: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    """获取（或创建）全局 Gemini 实例。"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = create_llm()
    return _llm_instance


# 中文 RAG 提示词：要求模型仅依据检索上下文作答，避免胡编
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是一个严谨的知识库助手。请仅根据下面提供的「参考资料」回答用户问题。\n"
                "若参考资料中没有足够信息，请明确说明「根据现有资料无法回答」，不要编造事实。\n"
                "回答请使用简洁、清晰的中文。\n\n"
                "参考资料：\n{context}"
            ),
        ),
        ("human", "{question}"),
    ]
)


def create_llm() -> ChatGoogleGenerativeAI:
    """
    创建 Google Gemini 聊天模型（通过 Google Generative AI）。

    API Key 来自环境变量 GEMINI_API_KEY。
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "未检测到环境变量 GEMINI_API_KEY。\n"
            "请在项目根目录创建 .env 文件并写入：GEMINI_API_KEY=你的密钥\n"
            "或在系统环境中设置该变量后再运行。"
        )

    # 显式传入 os.getenv("GEMINI_API_KEY") 读到的密钥（见 config.py）
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )
    print(f"[LLM] 已初始化 Gemini 模型：{GEMINI_MODEL_NAME}")
    return llm


def _format_docs(docs: list[Document]) -> str:
    """将检索到的多个文档块拼接成一段上下文字符串。"""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[片段 {i} | 来源: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(retriever: VectorStoreRetriever) -> Runnable:
    """
    组装完整的 RAG 链：检索 → 填充提示词 → Gemini 生成 → 解析为字符串。

    Args:
        retriever: 向量检索器

    Returns:
        可直接 invoke({"question": "..."}) 的 Runnable
    """
    llm = get_llm()

    # LCEL 流水线：
    # 1. 用 question 检索上下文，并原样传递 question
    # 2. 填入提示词模板
    # 3. 调用 Gemini
    # 4. 取出纯文本回答
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def ask_with_context(question: str, docs: list[Document]) -> str:
    """
    基于已检索到的文档块生成回答（避免重复检索）。

    Args:
        question: 用户问题
        docs: 检索到的文档块

    Returns:
        模型回答文本
    """
    llm = get_llm()
    context = _format_docs(docs)
    messages = RAG_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return StrOutputParser().invoke(response)


def ask(chain: Runnable, question: str) -> str:
    """
    向 RAG 链提问并返回答案。

    Args:
        chain: build_rag_chain 返回的链
        question: 用户问题

    Returns:
        模型生成的回答文本
    """
    # 检索器需要字符串作为查询；这里把 question 同时作为链的输入
    return chain.invoke(question)
