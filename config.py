"""
项目配置模块

集中管理路径、模型名称、分块参数等常量，便于后续调整。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件（若存在）
load_dotenv()

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

# 项目根目录（本文件所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# 存放待检索文档的目录（支持 .txt / .pdf / .md）
DOCS_DIR = PROJECT_ROOT / "docs"

# Chroma 向量库持久化目录（首次运行后会自动创建）
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Chroma 集合名称
CHROMA_COLLECTION_NAME = "rag_documents"

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

# HuggingFace 文本嵌入模型（本地下载并缓存）
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Google Gemini 对话模型
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# 从环境变量读取 API Key（不要硬编码在代码里）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# 文档切分参数
# ---------------------------------------------------------------------------

# 每个文本块的最大字符数
CHUNK_SIZE = 800

# 相邻块之间的重叠字符数（有助于保留上下文连贯性）
CHUNK_OVERLAP = 150

# 检索时返回的最相似文档块数量（送入 LLM 的上下文）
RETRIEVER_TOP_K = 4

# 初筛候选片段数量（用于后续按答案相关度再筛选）
RETRIEVER_CANDIDATE_K = 12

# 界面版本标识（用于确认是否加载了最新代码）
APP_VERSION = "2.1"

# 界面最多展示的引用片段数
MAX_DISPLAY_SOURCES = 3
