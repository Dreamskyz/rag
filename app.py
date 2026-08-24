"""
RAG 图形化界面（Gradio）

启动方式：
    python app.py

功能：
- 对话式问答
- 展示检索到的引用片段
- 上传文档到 docs/
- 一键重建 / 加载向量库
"""

from __future__ import annotations

import gradio as gr

from config import GEMINI_MODEL_NAME, APP_VERSION
from rag_service import RAGService

# 全局服务实例（Gradio 单进程内复用）
service = RAGService()


def on_startup() -> tuple[str, str]:
    """页面加载时自动尝试初始化向量库。"""
    status = service.auto_init()
    docs = service.get_docs_summary()
    return status, docs


def on_rebuild() -> tuple[str, str]:
    """重建向量库并刷新状态。"""
    status = service.rebuild_index()
    docs = service.get_docs_summary()
    return status, docs


def on_load() -> str:
    """加载已有向量库。"""
    return service.load_index()


def on_upload(files) -> tuple[str, str]:
    """处理文件上传。"""
    paths: list[str] = []
    if files:
        if isinstance(files, str):
            paths = [files]
        else:
            for item in files:
                if isinstance(item, str):
                    paths.append(item)
                else:
                    paths.append(item.name)
    upload_msg = service.save_uploaded_files(paths)
    docs = service.get_docs_summary()
    status = f"{upload_msg}\n\n当前状态：{service.status_message}"
    return status, docs


def on_send(question: str, history: list[dict]) -> tuple[list[dict], str, str]:
    """
    发送用户问题，更新聊天记录，并返回引用来源。

    Gradio Chatbot 使用 messages 格式：
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not question.strip():
        return history, "", ""

    answer, sources = service.chat(question)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return history, "", sources


def on_clear() -> tuple[list, str, str, str]:
    """清空对话与引用面板。"""
    return [], "", "", service.status_message


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面。"""
    with gr.Blocks(title="本地 RAG 知识库") as demo:
        gr.Markdown(
            f"""
            # 📚 本地 RAG 知识库问答
            基于 **LangChain + ChromaDB + Gemini ({GEMINI_MODEL_NAME})** · 版本 {APP_VERSION}
            """
        )

        with gr.Row():
            # ------------------------- 左侧：控制面板 -------------------------
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### ⚙️ 控制面板")

                status_box = gr.Textbox(
                    label="系统状态",
                    value="正在初始化…",
                    lines=5,
                    interactive=False,
                    elem_classes=["status-box"],
                )

                with gr.Row():
                    rebuild_btn = gr.Button("🔨 重建索引", variant="primary")
                    load_btn = gr.Button("📂 加载索引")

                gr.Markdown("### 📁 文档管理")
                docs_box = gr.Textbox(
                    label=f"docs/ 目录",
                    lines=8,
                    interactive=False,
                )

                upload = gr.File(
                    label="上传文档（txt / pdf / md）",
                    file_count="multiple",
                    file_types=[".txt", ".pdf", ".md"],
                )
                upload_btn = gr.Button("⬆️ 保存到 docs/")

                gr.Markdown(
                    """
                    **使用提示**
                    1. 将资料放入 `docs/` 或在此上传
                    2. 点击 **重建索引**
                    3. 在右侧输入问题开始对话
                    """
                )

            # ------------------------- 右侧：对话区 -------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 💬 智能问答")

                chatbot = gr.Chatbot(
                    label="对话记录",
                    height=420,
                    buttons=["copy"],
                )

                with gr.Row():
                    question_box = gr.Textbox(
                        label="输入问题",
                        placeholder="例如：什么是 RAG？",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

                sources_box = gr.Markdown(
                    value="*与本次回答最相关的参考片段将显示在这里*",
                    elem_classes=["sources-box"],
                )

        # ------------------------- 事件绑定 -------------------------
        demo.load(fn=on_startup, outputs=[status_box, docs_box])

        rebuild_btn.click(fn=on_rebuild, outputs=[status_box, docs_box])
        load_btn.click(fn=on_load, outputs=[status_box])
        upload_btn.click(fn=on_upload, inputs=[upload], outputs=[status_box, docs_box])

        send_btn.click(
            fn=on_send,
            inputs=[question_box, chatbot],
            outputs=[chatbot, question_box, sources_box],
        )
        question_box.submit(
            fn=on_send,
            inputs=[question_box, chatbot],
            outputs=[chatbot, question_box, sources_box],
        )
        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, question_box, sources_box, status_box],
        )

    return demo


def main() -> None:
    """启动 Gradio Web 服务。"""
    if error := RAGService.check_api_key():
        print(f"启动失败：{error}")
        raise SystemExit(1)

    demo = build_ui()
    print(f"\n✅ RAG 图形界面 v{APP_VERSION} 已启动")
    print("   引用片段将按「相关度」排序展示（非片段1/2/3）")
    print("   访问地址: http://127.0.0.1:7860")
    print("   若界面未更新，请先 Ctrl+C 停止旧进程后重新运行\n")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
            .status-box textarea { font-size: 13px !important; }
            .sources-box { font-size: 13px; }
        """,
    )


if __name__ == "__main__":
    main()
