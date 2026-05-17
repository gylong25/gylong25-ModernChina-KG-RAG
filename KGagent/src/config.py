"""Application configuration loaded from environment variables."""

from __future__ import annotations #延时类型注解

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings for Neo4j, DashScope and local data paths."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_model: str
    chroma_path: Path
    chroma_collection_name: str
    vector_top_k: int
    graph_path_max_depth: int
    data_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "your_neo4j_password"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            dashscope_base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            dashscope_model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
            chroma_path=Path(os.getenv("CHROMA_PATH", r"D:\ALLM\KGagent\chroma_db")),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "kg_people_docs_hash_v1"),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", "5")),
            graph_path_max_depth=int(os.getenv("GRAPH_PATH_MAX_DEPTH", "4")),
            data_path=Path(os.getenv("KG_DATA_PATH", r"D:\ALLM\KGagent\data-json.json")),
        )
