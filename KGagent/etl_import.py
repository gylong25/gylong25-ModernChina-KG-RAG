"""Command line entry for importing JSON data into Neo4j.

Usage:
    python etl_import.py

Before running, copy .env.example to .env and fill in Neo4j credentials.
"""

from src.config import Settings
from src.etl import import_json_to_neo4j
from src.graph_store import Neo4jGraphStore


def main() -> None:
    settings = Settings.from_env()
    graph = Neo4jGraphStore(settings)
    try:
        graph.ensure_constraints() # 确保创建唯一约束
        result = import_json_to_neo4j(settings.data_path, graph)
        print(
            "Import completed: "
            f"{result['people']} people, "
            f"{result['organizations']} organizations, "
            f"{result['works']} works, "
            f"{result['relations']} relations."
        )
    finally:
        graph.close()


if __name__ == "__main__":
    main()
