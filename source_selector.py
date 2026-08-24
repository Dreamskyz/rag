"""
引用片段筛选模块

根据向量相似度、问题匹配度、答案支撑度，筛选与问答最相关的参考片段。
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

# 中文连续字符（用于提取关键词）
_CHINESE_TERM_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
# 英文/数字词
_LATIN_TERM_PATTERN = re.compile(r"[A-Za-z0-9]{2,}")

# 答案中常见泛化词，不参与匹配
_STOP_WORDS = {
    "根据", "参考", "资料", "回答", "显示", "表明", "已经", "正式",
    "可以", "进行", "一个", "这个", "那个", "以及", "其中",
}


def extract_terms(text: str, min_len: int = 2) -> set[str]:
    """从文本中提取中文词组与英文单词，作为匹配关键词。"""
    terms: set[str] = set()
    for token in _CHINESE_TERM_PATTERN.findall(text):
        if len(token) >= min_len:
            terms.add(token)
    for token in _LATIN_TERM_PATTERN.findall(text):
        if len(token) >= min_len:
            terms.add(token.lower())
    return terms


def extract_key_phrases(text: str) -> set[str]:
    """
    提取用于答案匹配的关键短语。

    除完整中文词组外，还会从长短语中切出 3~6 字子串，
    便于匹配「大庙镇中学」「丽秀中学」等实体。
    """
    phrases = extract_terms(text)
    for seq in _CHINESE_TERM_PATTERN.findall(text):
        if len(seq) <= 4:
            continue
        for size in (3, 4, 5, 6):
            if size > len(seq):
                continue
            for i in range(len(seq) - size + 1):
                phrases.add(seq[i : i + size])
    return phrases


def extract_answer_anchors(answer: str) -> set[str]:
    """
    从模型回答中提取用于回匹配原文的锚点词。

    除常规短语外，还会补充常见别名（如「人大」↔「中国人民大学」）。
    """
    anchors = {
        p for p in extract_key_phrases(answer)
        if len(p) >= 2 and not any(sw in p for sw in _STOP_WORDS)
    }

    # 常见实体别名扩展
    if any(k in answer for k in ("中国人民大学", "人民大学")):
        anchors.update({"人大", "人民大学", "中国人民大学", "录取", "录取线"})
    if "心理学" in answer:
        anchors.update({"心理学", "心理学专业"})
    if "大学" in answer:
        anchors.update({"大学", "高考", "成绩", "录取"})

    return anchors


def _term_overlap_score(content: str, terms: set[str]) -> float:
    """计算文档内容与关键词集合的重叠比例（0~1）。"""
    if not terms:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for term in terms if term in content or term.lower() in content_lower)
    return hits / len(terms)


def _matched_terms(content: str, terms: set[str]) -> list[str]:
    """返回在文档中命中的关键词列表。"""
    content_lower = content.lower()
    matched = [t for t in terms if t in content or t.lower() in content_lower]
    # 优先展示较短、更有信息量的词
    matched.sort(key=len)
    return matched[:6]


def score_source_relevance(
    doc: Document,
    question: str,
    answer: str,
    vector_score: float,
) -> float:
    """
    综合评分：向量相似度 + 问题匹配 + 答案支撑。

    答案支撑权重最高，确保展示的片段确实支撑了模型回答。
    """
    content = doc.page_content
    question_terms = extract_terms(question)
    answer_anchors = extract_answer_anchors(answer)

    question_overlap = _term_overlap_score(content, question_terms)
    answer_support = _term_overlap_score(content, answer_anchors)

    # 同时命中多个答案锚点时加分（如「人大」+「心理学」+「录取」）
    anchor_hits = sum(1 for a in answer_anchors if a in content)
    multi_hit_bonus = min(0.15, anchor_hits * 0.04)

    return (
        0.25 * vector_score
        + 0.20 * question_overlap
        + 0.55 * answer_support
        + multi_hit_bonus
    )


def select_relevant_sources(
    docs_with_scores: list[tuple[Document, float]],
    question: str,
    answer: str,
    *,
    max_sources: int = 3,
    min_combined_score: float = 0.15,
    min_answer_support: float = 0.05,
) -> list[tuple[Document, float, float, list[str]]]:
    """
    从候选片段中筛选与问答最相关的引用。

    Returns:
        [(Document, 综合分, 向量分, 命中关键词), ...]，按综合分降序
    """
    answer_anchors = extract_answer_anchors(answer)
    all_ranked: list[tuple[Document, float, float, list[str]]] = []

    for doc, vector_score in docs_with_scores:
        combined = score_source_relevance(doc, question, answer, vector_score)
        answer_support = _term_overlap_score(doc.page_content, answer_anchors)
        matched = _matched_terms(doc.page_content, answer_anchors | extract_terms(question))

        all_ranked.append((doc, combined, vector_score, matched))

    # 严格筛选：答案中有实质支撑
    strict = [
        item for item in all_ranked
        if _term_overlap_score(item[0].page_content, answer_anchors) >= min_answer_support
        and item[1] >= min_combined_score
    ]
    strict.sort(key=lambda item: item[1], reverse=True)
    if strict:
        return strict[:max_sources]

    # 宽松回退：避免界面空白，仍按综合分展示最相关片段
    all_ranked.sort(key=lambda item: item[1], reverse=True)
    return all_ranked[:max_sources]


def format_relevant_sources(
    ranked: list[tuple[Document, float, float, list[str]]],
    preview_len: int = 260,
) -> str:
    """将筛选后的引用片段格式化为 Markdown。"""
    if not ranked:
        return "未找到与本次回答相关的参考片段。"

    parts: list[str] = ["### 📎 相关参考片段（按相关度排序）\n"]
    for doc, combined, vector_score, matched in ranked:
        source = doc.metadata.get("source", "未知来源")
        source_name = source.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        content = doc.page_content.strip().replace("\n", " ")
        if len(content) > preview_len:
            content = content[:preview_len] + "…"

        match_hint = ""
        if matched:
            match_hint = f"\n> 🔑 命中关键词：`{'`、`'.join(matched)}`"

        parts.append(
            f"**相关度 {combined * 100:.0f}%** · `{source_name}` "
            f"（向量 {vector_score * 100:.0f}%）\n> {content}{match_hint}"
        )
    return "\n\n".join(parts)
