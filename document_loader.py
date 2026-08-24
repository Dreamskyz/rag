"""
文档加载与切分模块

从 docs 目录加载 txt / pdf / md 文档，并切分为适合向量化的文本块。
"""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR


def _load_by_glob(docs_dir: Path, glob_pattern: str, loader_cls, **loader_kwargs) -> list[Document]:
    """
    使用 DirectoryLoader 按通配符批量加载某一类文件。

    Args:
        docs_dir: 文档根目录
        glob_pattern: 如 "**/*.txt"
        loader_cls: LangChain 文档加载器类
        **loader_kwargs: 传给加载器的额外参数

    Returns:
        Document 列表
    """
    loader = DirectoryLoader(
        path=str(docs_dir),
        glob=glob_pattern,
        loader_cls=loader_cls,
        loader_kwargs=loader_kwargs,
        # 某个文件解析失败时跳过，不中断整个流程
        silent_errors=True,
        show_progress=False,
        use_multithreading=False,
    )
    return loader.load()


def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """
    加载 docs 目录下所有支持的文档（txt、pdf、md）。

    Args:
        docs_dir: 文档目录，默认使用 config.DOCS_DIR

    Returns:
        原始 Document 列表（尚未切分）
    """
    target_dir = docs_dir or DOCS_DIR
    if not target_dir.exists():
        raise FileNotFoundError(
            f"文档目录不存在：{target_dir}\n请创建该目录并放入 .txt / .pdf / .md 文件。"
        )

    documents: list[Document] = []

    # .txt：按 UTF-8 读取纯文本
    documents.extend(
        _load_by_glob(
            target_dir,
            "**/*.txt",
            TextLoader,
            encoding="utf-8",
        )
    )

    # .md：Markdown 也可直接用 TextLoader 读取全文
    documents.extend(
        _load_by_glob(
            target_dir,
            "**/*.md",
            TextLoader,
            encoding="utf-8",
        )
    )

    # .pdf：按页解析为文本
    documents.extend(
        _load_by_glob(
            target_dir,
            "**/*.pdf",
            PyPDFLoader,
        )
    )

    if not documents:
        raise ValueError(
            f"在 {target_dir} 中未找到可用的 .txt / .pdf / .md 文档，请先放入示例文件。"
        )

    print(f"[文档加载] 共加载 {len(documents)} 个原始文档片段（按文件/页计）。")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    将长文档切分为固定大小的文本块，便于嵌入与检索。

    Args:
        documents: 原始文档列表

    Returns:
        切分后的 Document 列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 优先按段落、换行、空格等自然边界切分
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[文档切分] 共得到 {len(chunks)} 个文本块。")
    return chunks
