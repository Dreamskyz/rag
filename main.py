"""
本地 RAG 应用入口

功能概览：
1. 从 docs/ 加载 .txt / .pdf / .md 文档
2. 使用 all-MiniLM-L6-v2 向量化并写入本地 ChromaDB
3. 基于 Gemini（gemini-2.5-flash）进行检索增强问答

用法：
    # 首次或文档更新后：重建索引并进入问答
    python main.py

    # 仅加载已有向量库（跳过重新嵌入，启动更快）
    python main.py --load-only

依赖环境变量：
    GEMINI_API_KEY  —— 通过 os.getenv 读取（可用 .env 提供）
"""

from __future__ import annotations

import argparse
import sys

from config import CHROMA_DIR, DOCS_DIR, GEMINI_API_KEY
from document_loader import load_documents, split_documents
from rag_chain import ask, build_rag_chain
from vectorstore import build_vectorstore, get_retriever, load_vectorstore


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="本地 RAG 问答（LangChain + Chroma + Gemini）")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="不重新扫描 docs，直接加载已有 chroma_db 向量库",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="强制根据 docs 重建向量库（默认行为；与 --load-only 互斥）",
    )
    return parser.parse_args()


def ensure_api_key() -> None:
    """启动前检查 API Key 是否已配置。"""
    # 明确使用 os.getenv 语义：config 中已通过 os.getenv("GEMINI_API_KEY") 读取
    if not GEMINI_API_KEY:
        print(
            "错误：未设置 GEMINI_API_KEY。\n"
            "请在 .env 中写入 GEMINI_API_KEY=你的密钥，或先在终端导出该环境变量。",
            file=sys.stderr,
        )
        sys.exit(1)


def prepare_vectorstore(load_only: bool):
    """
    准备向量库：要么从磁盘加载，要么从 docs 重建。

    Returns:
        Chroma 向量库实例
    """
    if load_only:
        if not CHROMA_DIR.exists():
            print(
                f"错误：指定了 --load-only，但向量库目录不存在：{CHROMA_DIR}\n"
                "请先不带该参数运行一次，以完成索引构建。",
                file=sys.stderr,
            )
            sys.exit(1)
        return load_vectorstore()

    print(f"[准备] 扫描文档目录：{DOCS_DIR}")
    documents = load_documents()
    chunks = split_documents(documents)
    return build_vectorstore(chunks)


def interactive_qa(chain) -> None:
    """进入命令行交互式问答循环。"""
    print("\n" + "=" * 60)
    print("RAG 问答已就绪。输入问题后回车；输入 exit / quit / q 退出。")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        print("\n助手：正在检索并生成回答，请稍候…\n")
        try:
            answer = ask(chain, question)
            print(f"助手：{answer}\n")
        except Exception as exc:  # noqa: BLE001 — 交互层统一捕获，避免崩溃退出
            print(f"助手：生成回答时出错 —— {exc}\n", file=sys.stderr)


def main() -> None:
    """程序主入口：检查配置 → 构建/加载向量库 → 启动问答。"""
    args = parse_args()
    ensure_api_key()

    if args.load_only and args.rebuild:
        print("错误：--load-only 与 --rebuild 不能同时使用。", file=sys.stderr)
        sys.exit(1)

    # 默认：有 chroma_db 且用户没要求 rebuild 时，仍按「重建」更直观；
    # 若用户明确 --load-only 则跳过嵌入。
    load_only = bool(args.load_only)
    vectorstore = prepare_vectorstore(load_only=load_only)
    retriever = get_retriever(vectorstore)
    chain = build_rag_chain(retriever)

    interactive_qa(chain)


if __name__ == "__main__":
    main()
