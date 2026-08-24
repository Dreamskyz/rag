"""
RAG 服务层

封装向量库初始化、索引重建与问答逻辑，供命令行与图形界面共用。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from config import (
    CHROMA_DIR,
    DOCS_DIR,
    GEMINI_API_KEY,
    MAX_DISPLAY_SOURCES,
    RETRIEVER_TOP_K,
)
from document_loader import load_documents, split_documents
from rag_chain import ask_with_context, build_rag_chain
from source_selector import format_relevant_sources, select_relevant_sources
from vectorstore import build_vectorstore, get_retriever, load_vectorstore, retrieve_with_scores

# 支持的文档扩展名
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md"}


class RAGService:
    """本地 RAG 应用的核心服务类。"""

    def __init__(self) -> None:
        self.vectorstore = None
        self.retriever: VectorStoreRetriever | None = None
        self.chain: Runnable | None = None
        self.status_message = "尚未加载向量库"

    @staticmethod
    def check_api_key() -> str | None:
        """检查 API Key 是否已配置，未配置时返回错误信息。"""
        if not GEMINI_API_KEY:
            return (
                "未设置 GEMINI_API_KEY。\n"
                "请在项目根目录 .env 中配置：GEMINI_API_KEY=你的密钥"
            )
        return None

    def is_ready(self) -> bool:
        """向量库与问答链是否已就绪。"""
        return self.chain is not None and self.retriever is not None

    def _bind(self, vectorstore) -> str:
        """将向量库绑定到检索器与问答链。"""
        self.vectorstore = vectorstore
        self.retriever = get_retriever(vectorstore)
        self.chain = build_rag_chain(self.retriever)
        self.status_message = "向量库已就绪，可以开始提问"
        return self.status_message

    def rebuild_index(self) -> str:
        """扫描 docs/ 并重建向量库。"""
        if error := self.check_api_key():
            self.status_message = error
            return error

        try:
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            documents = load_documents()
            chunks = split_documents(documents)
            vectorstore = build_vectorstore(chunks)
            message = self._bind(vectorstore)
            return f"索引重建完成：共 {len(chunks)} 个文本块。\n{message}"
        except Exception as exc:  # noqa: BLE001
            self.status_message = f"索引重建失败：{exc}"
            return self.status_message

    def load_index(self) -> str:
        """加载已有 chroma_db 向量库。"""
        if error := self.check_api_key():
            self.status_message = error
            return error

        if not CHROMA_DIR.exists():
            self.status_message = "向量库不存在，请先点击「重建索引」。"
            return self.status_message

        try:
            vectorstore = load_vectorstore()
            message = self._bind(vectorstore)
            return f"已加载本地向量库。\n{message}"
        except Exception as exc:  # noqa: BLE001
            self.status_message = f"加载向量库失败：{exc}"
            return self.status_message

    def auto_init(self) -> str:
        """启动时自动初始化：有向量库则加载，否则提示用户重建。"""
        if error := self.check_api_key():
            self.status_message = error
            return error

        if CHROMA_DIR.exists():
            return self.load_index()
        self.status_message = "首次使用：请将文档放入 docs/ 后点击「重建索引」。"
        return self.status_message

    @staticmethod
    def list_doc_files() -> list[Path]:
        """列出 docs/ 目录下所有支持的文档文件。"""
        if not DOCS_DIR.exists():
            return []
        files: list[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(DOCS_DIR.rglob(f"*{ext}"))
        return sorted(files, key=lambda p: p.name.lower())

    def get_docs_summary(self) -> str:
        """返回 docs/ 目录文档概览文本。"""
        files = self.list_doc_files()
        if not files:
            return "docs/ 目录暂无文档（支持 .txt / .pdf / .md）"

        lines = [f"共 {len(files)} 个文件："]
        for path in files:
            rel = path.relative_to(DOCS_DIR)
            size_kb = path.stat().st_size / 1024
            lines.append(f"- {rel} ({size_kb:.1f} KB)")
        return "\n".join(lines)

    def save_uploaded_files(self, file_paths: list[str | Path]) -> str:
        """
        将用户上传的文件保存到 docs/ 目录。

        Args:
            file_paths: Gradio 传入的临时文件路径列表

        Returns:
            操作结果说明
        """
        if not file_paths:
            return "未选择任何文件。"

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        skipped: list[str] = []

        for raw_path in file_paths:
            src = Path(raw_path)
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped.append(f"{src.name}（不支持的格式）")
                continue

            dest = DOCS_DIR / src.name
            # 同名文件直接覆盖，便于更新资料
            dest.write_bytes(src.read_bytes())
            saved.append(dest.name)

        parts: list[str] = []
        if saved:
            parts.append("已保存：" + "、".join(saved))
        if skipped:
            parts.append("已跳过：" + "、".join(skipped))
        parts.append("请记得点击「重建索引」使新文档生效。")
        return "\n".join(parts)

    def chat(self, question: str) -> tuple[str, str]:
        """
        执行一次问答，并返回回答与最相关的引用来源。

        流程：
        1. 向量检索较多候选片段
        2. 取最相关的若干条作为 LLM 上下文生成回答
        3. 结合答案内容再次筛选，只展示真正支撑回答的片段

        Returns:
            (回答文本, 引用来源 Markdown)
        """
        question = question.strip()
        if not question:
            return "请输入问题。", ""

        if not self.is_ready() or self.vectorstore is None:
            return "向量库尚未就绪，请先重建或加载索引。", ""

        try:
            # 1. 检索候选片段（带相似度分数）
            candidates = retrieve_with_scores(self.vectorstore, question)

            if not candidates:
                return "未检索到相关文档，请确认 docs/ 中是否有相关资料。", ""

            # 2. 取向量最相似的前 K 条作为 LLM 上下文
            context_docs = [doc for doc, _ in candidates[:RETRIEVER_TOP_K]]
            answer = ask_with_context(question, context_docs)

            # 3. 根据答案内容筛选真正相关的引用片段
            relevant = select_relevant_sources(
                candidates,
                question,
                answer,
                max_sources=MAX_DISPLAY_SOURCES,
            )
            sources = format_relevant_sources(relevant)
            return answer, sources
        except Exception as exc:  # noqa: BLE001
            return f"生成回答时出错：{exc}", ""
