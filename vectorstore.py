"""
向量库模块

使用 all-MiniLM-L6-v2 生成嵌入，并将文档写入 / 读出本地 ChromaDB。
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_CANDIDATE_K,
    RETRIEVER_TOP_K,
)


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    创建 HuggingFace 嵌入模型实例。

    首次运行会自动从 HuggingFace Hub 下载 all-MiniLM-L6-v2 并缓存到本地。
    """
    print(f"[嵌入模型] 正在加载：{EMBEDDING_MODEL_NAME} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # 归一化向量，便于用余弦相似度做检索
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[嵌入模型] 加载完成。")
    return embeddings


def build_vectorstore(
    chunks: list[Document],
    persist_directory: Path | None = None,
) -> Chroma:
    """
    根据文本块构建（或重建）本地 Chroma 向量库。

    Args:
        chunks: 切分后的文档块
        persist_directory: 持久化目录，默认 config.CHROMA_DIR

    Returns:
        Chroma 向量库实例
    """
    chroma_path = persist_directory or CHROMA_DIR
    chroma_path.mkdir(parents=True, exist_ok=True)

    embeddings = create_embeddings()

    print(f"[向量库] 正在写入 Chroma：{chroma_path}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print("[向量库] 构建完成。")
    return vectorstore


def load_vectorstore(persist_directory: Path | None = None) -> Chroma:
    """
    从磁盘加载已有的 Chroma 向量库（不重新嵌入文档）。

    Args:
        persist_directory: 持久化目录

    Returns:
        Chroma 向量库实例
    """
    chroma_path = persist_directory or CHROMA_DIR
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"向量库目录不存在：{chroma_path}\n请先运行一次完整索引构建。"
        )

    embeddings = create_embeddings()
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print(f"[向量库] 已从 {chroma_path} 加载。")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int | None = None) -> VectorStoreRetriever:
    """
    从向量库创建检索器。

    Args:
        vectorstore: Chroma 实例
        k: 返回的相似文档数量

    Returns:
        VectorStoreRetriever
    """
    top_k = k if k is not None else RETRIEVER_TOP_K
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def _distance_to_relevance(distance: float) -> float:
    """
    将 Chroma 返回的距离/分数转换为 0~1 的相关度。

    不同版本可能返回距离（越小越相似）或已归一化的相似度。
    """
    if 0.0 <= distance <= 1.0:
        # 已是相似度
        return float(distance)
    if distance < 0.0:
        # 部分后端直接返回负的相似度残差
        return max(0.0, 1.0 + distance)
    # 距离值：用平滑函数映射到 (0, 1]
    return 1.0 / (1.0 + distance)


def retrieve_with_scores(
    vectorstore: Chroma,
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """
    带相似度分数的向量检索（分数越高越相关，范围 0~1）。

    Args:
        vectorstore: Chroma 实例
        query: 查询文本
        k: 候选片段数量

    Returns:
        [(Document, relevance_score), ...]，按相关度降序
    """
    candidate_k = k if k is not None else RETRIEVER_CANDIDATE_K
    raw_results = vectorstore.similarity_search_with_score(query, k=candidate_k)
    results = [(doc, _distance_to_relevance(score)) for doc, score in raw_results]
    results.sort(key=lambda item: item[1], reverse=True)
    return results
