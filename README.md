# 本地 RAG 知识库问答

基于 **LangChain + ChromaDB + Google Gemini** 的轻量级本地检索增强生成（RAG）项目。将 `docs/` 目录中的文档向量化后存入本地向量库，再通过 Gemini 模型结合检索结果回答用户问题。

## 功能特性

- 支持 **txt / pdf / md** 三种文档格式
- 使用 **all-MiniLM-L6-v2** 进行本地文本向量化
- 使用 **ChromaDB** 持久化向量库，无需重复嵌入
- 使用 **gemini-2.5-flash** 生成回答（Google Generative AI）
- **Gradio 图形界面**：对话问答、文档上传、索引管理
- 命令行交互式问答，开箱即用

## 技术栈

| 组件 | 用途 |
|------|------|
| LangChain | 文档加载、切分、检索与问答链编排 |
| ChromaDB | 本地向量数据库 |
| sentence-transformers/all-MiniLM-L6-v2 | 文本嵌入模型 |
| Google Gemini (gemini-2.5-flash) | 大语言模型生成 |
| pypdf | PDF 文档解析 |
| python-dotenv | 环境变量管理 |
| Gradio | 图形化 Web 界面 |

## 项目结构

```
RAG/
├── docs/                 # 待检索文档目录（txt / pdf / md）
├── chroma_db/            # 向量库持久化目录（运行后自动生成）
├── config.py             # 路径、模型与分块参数配置
├── document_loader.py    # 文档加载与文本切分
├── vectorstore.py        # 嵌入模型与 Chroma 向量库
├── rag_chain.py          # RAG 问答链（检索 + Gemini）
├── rag_service.py        # RAG 服务层（CLI / GUI 共用）
├── main.py               # 命令行入口
├── app.py                # 图形化界面入口（Gradio）
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量示例
└── .env                  # 本地 API Key（勿提交 Git）
```

## 环境要求

- Python 3.10+
- 可访问 Google Generative AI API
- 首次运行需联网下载 HuggingFace 嵌入模型（约数百 MB）

## 快速开始

### 1. 克隆或进入项目目录

```powershell
cd G:\RAG
```

### 2. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. 配置 API Key

复制环境变量示例文件，并填入你的 Gemini API Key：

```powershell
copy .env.example .env
```

编辑 `.env`：

```env
GEMINI_API_KEY=你的_Gemini_API_Key
```

程序通过 `os.getenv("GEMINI_API_KEY")` 读取密钥，不会将 Key 硬编码在源码中。

> 获取 API Key：[Google AI Studio](https://aistudio.google.com/apikey)

### 4. 放入文档

将你的资料放入 `docs/` 目录，支持以下格式：

- `.txt` — 纯文本
- `.md` — Markdown
- `.pdf` — PDF 文档

项目已自带 `docs/sample.txt` 与 `docs/sample.md` 作为示例。

### 5. 运行

**方式一：图形界面（推荐）**

```powershell
python app.py
```

浏览器访问 [http://127.0.0.1:7860](http://127.0.0.1:7860)，即可进行对话问答、上传文档、重建索引。

**方式二：命令行**

```powershell
python main.py
```

首次运行会扫描文档、下载嵌入模型、构建向量库，然后进入交互式问答。输入问题后回车即可；输入 `exit`、`quit` 或 `q` 退出。

## 使用说明

### 图形界面功能

| 功能 | 说明 |
|------|------|
| 智能问答 | 右侧输入问题，基于 docs/ 资料检索并生成回答 |
| 引用来源 | 每次回答下方展示检索到的文档片段 |
| 重建索引 | 扫描 docs/ 并重新向量化（文档更新后使用） |
| 加载索引 | 直接加载已有 chroma_db/，跳过重新嵌入 |
| 上传文档 | 将 txt / pdf / md 保存到 docs/ 目录 |

### 命令行参数

| 参数 | 说明 |
|------|------|
| （无参数） | 扫描 `docs/` 并重建向量库，然后进入问答 |
| `--load-only` | 跳过文档扫描与嵌入，直接加载已有 `chroma_db/` |
| `--rebuild` | 强制重建向量库（与默认行为相同） |

示例：

```powershell
# 文档更新后，重新建库并问答
python main.py

# 向量库已存在，快速启动
python main.py --load-only
```

### 更新知识库

1. 向 `docs/` 添加、修改或删除文档
2. 重新运行 `python main.py`（不要加 `--load-only`）
3. 程序会重新切分文档并写入 `chroma_db/`

## 工作流程

```
docs/ 文档
    ↓ 加载（txt / pdf / md）
文本切分（chunk_size=800, overlap=150）
    ↓ all-MiniLM-L6-v2 向量化
ChromaDB 持久化（chroma_db/）
    ↓ 用户提问
相似度检索（top_k=4）
    ↓ 检索结果 + 问题
Gemini (gemini-2.5-flash) 生成回答
```

## 配置说明

可在 `config.py` 中调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | 嵌入模型 |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Gemini 对话模型 |
| `CHUNK_SIZE` | `800` | 每个文本块最大字符数 |
| `CHUNK_OVERLAP` | `150` | 相邻块重叠字符数 |
| `RETRIEVER_TOP_K` | `4` | 检索返回的文档块数量 |

## 常见问题

**Q：提示未设置 `GEMINI_API_KEY`？**

确认项目根目录存在 `.env` 文件，且其中包含 `GEMINI_API_KEY=...`，或在系统环境变量中设置该变量。

**Q：首次运行很慢？**

首次需下载 `all-MiniLM-L6-v2` 嵌入模型并对文档做向量化，属正常现象。之后可使用 `--load-only` 跳过嵌入步骤。

**Q：Windows 上出现 HuggingFace 符号链接警告？**

不影响使用。如需消除警告，可开启 Windows 开发者模式，或设置环境变量 `HF_HUB_DISABLE_SYMLINKS_WARNING=1`。

**Q：回答与文档内容不符？**

- 确认相关文档已放入 `docs/` 并重新运行 `python main.py`
- 尝试调大 `RETRIEVER_TOP_K` 或减小 `CHUNK_SIZE`
- 模型会尽量依据检索到的上下文作答，资料不足时会明确说明

## 安全提示

- **切勿**将 `.env` 或 API Key 提交到 Git（已加入 `.gitignore`）
- 若 Key 曾泄露，请在 Google AI Studio 中轮换密钥

## 许可证

本项目仅供学习与个人使用。
