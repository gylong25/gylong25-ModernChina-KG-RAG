"""GraphRAG service with hybrid retrieval: Neo4j + ChromaDB."""

from __future__ import annotations

from itertools import combinations

from openai import OpenAI

from src.config import Settings
from src.graph_store import Neo4jGraphStore
from src.vector_store import ChromaVectorStore


class GraphRAGService:
    """Retrieve graph facts and vector matches before asking Qwen."""

    def __init__(self, graph: Neo4jGraphStore, settings: Settings) -> None:
        self.graph = graph
        self.settings = settings
        self.model = settings.dashscope_model
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self._vector_store: ChromaVectorStore | None = None
        self._vector_store_error: Exception | None = None

    def _get_vector_store(self) -> ChromaVectorStore | None:
        """Initialize Chroma lazily so the UI can load even if vector setup fails."""
        if self._vector_store is not None:
            return self._vector_store
        if self._vector_store_error is not None:
            return None

        try:
            store = ChromaVectorStore(self.settings)
            store.ensure_ready(self.settings.data_path)
            self._vector_store = store
            return store
        except Exception as exc:  # noqa: BLE001
            self._vector_store_error = exc
            return None

    def _extract_entities(self, question: str) -> list[str]:
        """Pull candidate entities from the question using graph-backed lookup."""
        return self.graph.find_person_entities_in_question(question, limit=6)

    @staticmethod
    def _format_directed_row(row: dict) -> str:
        """Render one relation as an explicit directed statement."""
        source = row.get("source_name") or row.get("path_source_name") or "Unknown"
        target = row.get("target_name") or row.get("path_target_name") or "Unknown"
        relation = row.get("relation_name") or row.get("relation_type") or "RELATED_TO"

        if row.get("path_source_name") and row.get("path_target_name"):
            if source == row.get("source_name") and target == row.get("target_name"):
                return f"{source} --({relation})--> {target}"
            if source == row.get("target_name") and target == row.get("source_name"):
                return f"{source} <--({relation})-- {target}"
        return f"{source} --({relation})--> {target}"

    @staticmethod
    def _format_vector_item(item: dict, idx: int) -> str:
        """Format one vector retrieval result."""
        meta = item.get("metadata") or {}
        entity_name = meta.get("entity_name", "Unknown")
        score = item.get("score", 0.0)
        document = item.get("document", "")
        return f"[向量召回 {idx}] {entity_name} | score={score} | {document}"

    def _format_context(self, title: str, rows: list[dict], entities: list[str] | None = None) -> str:
        """Turn graph rows into a compact text block for the LLM."""
        lines: list[str] = [f"[{title}]"]
        if entities:
            lines.append("命中实体: " + "、".join(entities))
        for row in rows:
            lines.append(self._format_directed_row(row))
        return "\n".join(lines)

    def _path_context(self, entities: list[str]) -> str:
        """Try pairwise multi-hop path discovery and return the first valid path."""
        for entity_a, entity_b in combinations(entities, 2):
            path = self.graph.find_path_between_entities(
                entity_a,
                entity_b,
                max_depth=self.settings.graph_path_max_depth,
                prefer_known=True,
            )
            if not path:
                continue

            rows = path["relationships"]
            return self._format_context("图谱多跳路径", rows, [entity_a, entity_b])
        return ""

    def _single_entity_context(self, entity: str) -> str:
        """Build one-hop context around a single entity with hard unknown cap."""
        rows = self.graph.get_prioritized_neighborhood(entity, known_limit=40, unknown_limit=10)
        if not rows:
            return ""
        return self._format_context("一跳关系", rows, [entity])

    def _graph_context(self, question: str, entities: list[str]) -> str:
        """Build the structured-graph part of the prompt."""
        if len(entities) >= 2:
            path_context = self._path_context(entities)
            if path_context:
                return path_context

            for entity in entities:
                context = self._single_entity_context(entity)
                if context:
                    return context
            return ""

        if len(entities) == 1:
            return self._single_entity_context(entities[0])

        return self.graph.get_graph_context(question)

    def _vector_context(self, question: str, entities: list[str]) -> str:
        """Build the semantic-retrieval part of the prompt."""
        vector_store = self._get_vector_store()
        if vector_store is None:
            return ""

        queries = [question] + entities
        seen_names: set[str] = set()
        lines: list[str] = ["[向量检索]"]

        merged: list[dict] = []
        for query in queries:
            if not query.strip():
                continue
            for item in vector_store.search(query, top_k=self.settings.vector_top_k):
                entity_name = (item.get("metadata") or {}).get("entity_name", "")
                if entity_name in seen_names:
                    continue
                seen_names.add(entity_name)
                merged.append(item)

        if not merged:
            return ""

        for idx, item in enumerate(merged[: self.settings.vector_top_k], start=1):
            lines.append(self._format_vector_item(item, idx))
        return "\n".join(lines)

    def build_context(self, question: str) -> str:
        """Create a hybrid context from graph retrieval and Chroma retrieval."""
        entities = self._extract_entities(question)
        graph_context = self._graph_context(question, entities)
        vector_context = self._vector_context(question, entities)

        parts = []
        if graph_context:
            parts.append(graph_context)
        if vector_context:
            parts.append(vector_context)
        return "\n\n".join(parts)

    def answer(self, question: str) -> tuple[str, str]:
        """Build hybrid context first, then send it to Qwen for a grounded answer."""
        context = self.build_context(question)
        if not context:
            return "图谱中没有检索到足够的相关事实，建议换一个更具体的人名或关系继续提问。", ""

        prompt = f"""你是一位严谨的中国近代史知识图谱问答助手。
请只依据【图谱上下文】回答用户问题；如果上下文不足，请明确说明“不足以判断”，不要编造事实。
如果上下文中存在“图谱多跳路径”，请优先沿完整路径回答，不要只回答起点和终点的直接关系。

【图谱上下文】
{context}

【用户问题】
{question}
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你擅长基于结构化知识图谱事实进行简洁、准确的中文回答。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            return f"OpenAI 兼容接口调用失败：{exc}", context

        answer_text = completion.choices[0].message.content or ""
        return answer_text, context
