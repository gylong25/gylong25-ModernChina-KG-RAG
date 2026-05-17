"""ChromaDB vector retrieval for GraphRAG.

This version uses a lightweight, pure-Python hash embedding so the Streamlit
app does not depend on torch or sentence-transformers. That keeps the front end
stable on Windows while still giving us semantic-ish recall for mixed retrieval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import chromadb

from src.config import Settings


class HashEmbeddingFunction:
    """A small embedding function that does not require any ML runtime.

    It hashes character uni/bi/tri-grams into a fixed-size vector and L2
    normalizes the result. The vector size matches common Chinese embedding
    models so persisted Chroma collections stay compatible.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def __call__(self, input: list[str]) -> list[list[float]]:  # Chroma calls this.
        return [self._embed_one(text) for text in input]

    def _embed_one(self, text: str) -> list[float]:
        cleaned = re.sub(r"\s+", "", str(text or ""))
        if not cleaned:
            return [0.0] * self.dimension

        chars = list(cleaned)
        features: list[str] = []
        features.extend(chars)
        features.extend(chars[i : i + 2] for i in range(len(chars) - 1))
        features.extend(chars[i : i + 3] for i in range(len(chars) - 2))

        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class ChromaVectorStore:
    """Persistent Chroma collection backed by document summaries."""

    EMBEDDING_BACKEND = "hash-v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.embedding_fn = HashEmbeddingFunction(dimension=384)
        self.collection = self._open_collection()

    @staticmethod
    def _list_to_text(values: Any) -> str:
        if not values:
            return ""
        if isinstance(values, list):
            return "、".join(str(v).strip() for v in values if str(v).strip())
        return str(values).strip()

    def _open_collection(self):
        """Open the collection, recreating it if the embedding backend changed."""
        collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "embedding_backend": self.EMBEDDING_BACKEND,
            },
        )

        metadata = collection.metadata or {}
        if metadata.get("embedding_backend") != self.EMBEDDING_BACKEND:
            try:
                self.client.delete_collection(self.settings.chroma_collection_name)
            except Exception:
                pass
            collection = self.client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                embedding_function=self.embedding_fn,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_backend": self.EMBEDDING_BACKEND,
                },
            )
        return collection

    def _record_to_document(self, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Convert one JSON person record into a searchable text chunk."""
        name = str(record.get("中文名", "")).strip()
        aliases = self._list_to_text(record.get("附加名称"))
        schools = self._list_to_text(record.get("毕业于"))
        works = self._list_to_text(record.get("作品"))
        related = record.get("相关人物") or {}
        related_parts: list[str] = []
        if isinstance(related, dict):
            for rel_name, rel_values in related.items():
                rel_text = self._list_to_text(rel_values)
                if rel_text:
                    related_parts.append(f"{rel_name}:{rel_text}")

        text = (
            f"人物：{name}。"
            f"别名：{aliases or '无'}。"
            f"国籍：{record.get('国籍', '未知')}。"
            f"民族：{record.get('民族', '未知')}。"
            f"出生地：{record.get('出生地', '未知')}。"
            f"出生日期：{record.get('出生日期', '未知')}。"
            f"死亡日期：{record.get('死亡日期', '未知')}。"
            f"工作职责：{record.get('工作职责', '未知')}。"
            f"毕业于：{schools or '无'}。"
            f"作品：{works or '无'}。"
            f"相关人物：{'; '.join(related_parts) if related_parts else '无'}。"
        )
        metadata = {
            "entity_name": name,
            "aliases": aliases,
            "source": "data-json.json",
            "doc_type": "person_profile",
        }
        return text, metadata

    def rebuild_from_json(self, data_path: Path) -> None:
        """Rebuild the vector index from the source JSON file."""
        records = json.loads(data_path.read_text(encoding="utf-8"))
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, record in enumerate(records):
            name = str(record.get("中文名", "")).strip()
            if not name:
                continue
            text, metadata = self._record_to_document(record)
            ids.append(f"person-{idx:05d}-{name}")
            documents.append(text)
            metadatas.append(metadata)

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def ensure_ready(self, data_path: Path) -> None:
        """Build the collection on first use if it is empty."""
        if self.collection.count() == 0:
            self.rebuild_from_json(data_path)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Run vector retrieval and return scored documents."""
        if not query.strip():
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items: list[dict[str, Any]] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            items.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": distance,
                    "score": round(1 - float(distance), 4),
                }
            )
        return items
