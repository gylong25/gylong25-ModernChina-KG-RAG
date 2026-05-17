"""Neo4j data access layer with retrieval helpers for GraphRAG."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from neo4j import GraphDatabase

from src.config import Settings


class Neo4jGraphStore:
    """Thin wrapper around the official Neo4j driver.

    The class exposes small, focused retrieval methods so the GraphRAG layer
    can choose the best strategy without duplicating Cypher everywhere.
    """

    def __init__(self, settings: Settings) -> None:
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        """Execute a Cypher query and return a list of plain dictionaries."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **parameters)
            return [dict(record) for record in result]

    def ensure_constraints(self) -> None:
        """Create the unique constraints used by MERGE operations."""
        self.run("CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
        self.run("CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
        self.run("CREATE CONSTRAINT work_name IF NOT EXISTS FOR (w:Work) REQUIRE w.name IS UNIQUE")

    def merge_person(self, props: dict[str, Any]) -> None:
        """Upsert a Person node and keep aliases deduplicated."""
        self.run(
            """
            MERGE (p:Person {name: $name})
            SET p += $props
            WITH p, coalesce(p.aliases, []) + $aliases AS raw_aliases
            SET p.aliases = reduce(unique = [], alias IN raw_aliases |
                CASE WHEN alias IN unique THEN unique ELSE unique + alias END
            )
            """,
            name=props["name"],
            props={k: v for k, v in props.items() if k != "aliases"},
            aliases=props.get("aliases", []),
        )

    def merge_organization(self, name: str) -> None:
        self.run("MERGE (:Organization {name: $name})", name=name)

    def merge_work(self, name: str) -> None:
        self.run("MERGE (:Work {name: $name})", name=name)

    def merge_relation(
        self,
        source_name: str,
        target_name: str,
        rel_type: str,
        target_label: str,
        display_name: str,
        weight: int | float | None = None,
    ) -> None:
        """Upsert one relationship.

        The relationship type is interpolated because Neo4j does not parameterize
        rel types. The ETL layer already sanitizes the type string.
        """
        effective_weight = weight if weight is not None else (1 if rel_type == "UNTYPED_RELATION" else 10)
        query = f"""
        MATCH (source:Person {{name: $source_name}})
        MATCH (target:{target_label} {{name: $target_name}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.name = $display_name
        SET r.weight = $weight
        """
        self.run(
            query,
            source_name=source_name,
            target_name=target_name,
            display_name=display_name,
            weight=effective_weight,
        )

    def find_person_entities_in_question(self, question: str, limit: int = 8) -> list[str]:
        """Extract canonical person names mentioned in a question.

        This is a lightweight entity recognizer: we look for exact name or alias
        overlaps against stored Person nodes, then rank candidates by match
        position and length so longer, more specific names win first.
        """
        rows = self.run(
            """
            MATCH (p:Person)
            WHERE $question CONTAINS p.name
               OR any(alias IN coalesce(p.aliases, []) WHERE $question CONTAINS alias)
            RETURN DISTINCT p.name AS name, coalesce(p.aliases, []) AS aliases
            """,
            question=question,
        )

        scored_rows: list[tuple[int, int, str]] = []
        for row in rows:
            candidates = [row["name"], *row.get("aliases", [])]
            positions = [question.find(text) for text in candidates if text and question.find(text) >= 0]
            if not positions:
                continue
            scored_rows.append((min(positions), -len(row["name"]), row["name"]))

        scored_rows.sort()
        names: list[str] = []
        for _, _, name in scored_rows:
            if name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    def get_prioritized_neighborhood(
        self,
        entity_name: str,
        known_limit: int = 40,
        unknown_limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return one-hop relations around an entity.

        The query is undirected and splits known relations from
        UNTYPED_RELATION so that untyped edges cannot dominate the top-k budget.
        """
        known_rows = self.run(
            """
            MATCH (p:Person)
            WHERE p.name = $entity_name OR $entity_name IN coalesce(p.aliases, [])
            MATCH (p)-[r]-(n)
            WHERE type(r) <> 'UNTYPED_RELATION'
            RETURN
                startNode(r).name AS source_name,
                endNode(r).name AS target_name,
                coalesce(r.name, type(r)) AS relation_name,
                type(r) AS relation_type,
                coalesce(r.weight, 0) AS weight,
                labels(startNode(r)) AS source_labels,
                labels(endNode(r)) AS target_labels,
                properties(startNode(r)) AS source_properties,
                properties(endNode(r)) AS target_properties
            ORDER BY weight DESC, relation_name ASC, source_name ASC, target_name ASC
            LIMIT $limit
            """,
            entity_name=entity_name,
            limit=known_limit,
        )

        unknown_rows = self.run(
            """
            MATCH (p:Person)
            WHERE p.name = $entity_name OR $entity_name IN coalesce(p.aliases, [])
            MATCH (p)-[r]-(n)
            WHERE type(r) = 'UNTYPED_RELATION'
            RETURN
                startNode(r).name AS source_name,
                endNode(r).name AS target_name,
                coalesce(r.name, type(r)) AS relation_name,
                type(r) AS relation_type,
                coalesce(r.weight, 0) AS weight,
                labels(startNode(r)) AS source_labels,
                labels(endNode(r)) AS target_labels,
                properties(startNode(r)) AS source_properties,
                properties(endNode(r)) AS target_properties
            ORDER BY weight DESC, relation_name ASC, source_name ASC, target_name ASC
            LIMIT $limit
            """,
            entity_name=entity_name,
            limit=unknown_limit,
        )

        return known_rows + unknown_rows[:unknown_limit]

    def find_path_between_entities(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 2,
        prefer_known: bool = True,
    ) -> dict[str, Any] | None:
        """Find a path between two entities within max_depth hops.

        We keep the pattern undirected so the lookup can follow either stored
        direction. The returned relationships preserve the actual Neo4j edge
        orientation via startNode/endNode.
        """
        safe_depth = max(1, min(int(max_depth), 6))
        if prefer_known:
            # Prefer a path made only of typed relations. If none exists, fall
            # back to any relationship so recall does not collapse completely.
            known_path = self._find_path_between_entities(
                entity_a,
                entity_b,
                max_depth=safe_depth,
                known_only=True,
            )
            if known_path:
                return known_path

        return self._find_path_between_entities(
            entity_a,
            entity_b,
            max_depth=safe_depth,
            known_only=False,
        )

    def _find_path_between_entities(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int,
        known_only: bool,
    ) -> dict[str, Any] | None:
        rel_filter = "WHERE all(rel IN relationships(p) WHERE type(rel) <> 'UNTYPED_RELATION')" if known_only else ""
        query = """
            MATCH (a:Person)
            WHERE a.name = $entity_a OR $entity_a IN coalesce(a.aliases, [])
            WITH a LIMIT 1
            MATCH (b:Person)
            WHERE b.name = $entity_b OR $entity_b IN coalesce(b.aliases, [])
            WITH a, b LIMIT 1
            MATCH p = shortestPath((a)-[*1..__DEPTH__]-(b))
            __REL_FILTER__
            WITH p, range(0, size(relationships(p)) - 1) AS rel_indexes
            UNWIND rel_indexes AS idx
            WITH
                p,
                idx,
                nodes(p)[idx] AS path_source,
                nodes(p)[idx + 1] AS path_target,
                relationships(p)[idx] AS rel
            RETURN
                collect({
                    step: idx,
                    path_source_name: path_source.name,
                    path_target_name: path_target.name,
                    source_name: startNode(rel).name,
                    target_name: endNode(rel).name,
                    relation_name: coalesce(rel.name, type(rel)),
                    relation_type: type(rel),
                    weight: coalesce(rel.weight, 0),
                    source_labels: labels(startNode(rel)),
                    target_labels: labels(endNode(rel)),
                    source_properties: properties(startNode(rel)),
                    target_properties: properties(endNode(rel))
                }) AS relationships,
                [node IN nodes(p) | {
                    id: elementId(node),
                    name: node.name,
                    labels: labels(node),
                    properties: properties(node)
                }] AS nodes
            LIMIT 1
        """
        query = query.replace("__DEPTH__", str(max_depth)).replace("__REL_FILTER__", rel_filter)
        rows = self.run(
            query,
            entity_a=entity_a,
            entity_b=entity_b,
        )
        if not rows:
            return None
        return rows[0]

    def get_person_neighborhood(
        self,
        keyword: str,
        depth: int = 2,
        limit: int = 150,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return nodes and relationships within N hops of a matched person."""
        safe_depth = max(1, min(int(depth), 4))
        query = """
        MATCH (p:Person)
        WHERE p.name CONTAINS $keyword OR $keyword IN coalesce(p.aliases, [])
        MATCH path = (p)-[*1..__DEPTH__]-(n)
        WITH path
        LIMIT $limit
        WITH collect(path) AS paths
        UNWIND paths AS path
        UNWIND nodes(path) AS node
        WITH paths, collect(DISTINCT node) AS nodes
        UNWIND paths AS path
        UNWIND relationships(path) AS rel
        RETURN
            [node IN nodes | {
                id: elementId(node),
                name: node.name,
                labels: labels(node),
                properties: properties(node)
            }] AS nodes,
            collect(DISTINCT {
                id: elementId(rel),
                source: elementId(startNode(rel)),
                target: elementId(endNode(rel)),
                type: type(rel),
                name: coalesce(rel.name, type(rel)),
                weight: coalesce(rel.weight, 0)
            }) AS relationships
        """
        query = query.replace("__DEPTH__", str(safe_depth))
        rows = self.run(query, keyword=keyword, limit=limit)
        if not rows:
            return {"nodes": [], "relationships": []}
        return {"nodes": rows[0]["nodes"], "relationships": rows[0]["relationships"]}

    def get_basic_stats(self) -> dict[str, int]:
        people = self.run("MATCH (p:Person) RETURN count(p) AS count")[0]["count"]
        relations = self.run("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        return {"people_count": people, "relation_count": relations}

    def get_top_degree_people(self, limit: int = 5) -> list[dict[str, Any]]:
        return self.run(
            """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[r]-()
            RETURN p.name AS name, count(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            """,
            limit=limit,
        )

    def get_graph_context(self, question: str, limit: int = 50) -> str:
        """Compatibility wrapper that returns a readable graph context string.

        GraphRAG now does its own strategy selection, but this method remains for
        callers that still expect a preformatted text context.
        """
        entities = self.find_person_entities_in_question(question, limit=4)
        rows: list[dict[str, Any]] = []

        if len(entities) >= 2:
            for entity_a, entity_b in combinations(entities, 2):
                path = self.find_path_between_entities(entity_a, entity_b)
                if path:
                    rows = path["relationships"]
                    break

        if not rows and len(entities) >= 1:
            rows = self.get_prioritized_neighborhood(entities[0], known_limit=max(limit - 10, 0), unknown_limit=min(10, limit))

        if not rows:
            rows = self.run(
                """
                MATCH (p:Person)-[r]-(n)
                WHERE p.name CONTAINS $question OR n.name CONTAINS $question
                RETURN
                    startNode(r).name AS source_name,
                    endNode(r).name AS target_name,
                    coalesce(r.name, type(r)) AS relation_name,
                    type(r) AS relation_type,
                    coalesce(r.weight, 0) AS weight,
                    labels(startNode(r)) AS source_labels,
                    labels(endNode(r)) AS target_labels,
                    properties(startNode(r)) AS source_properties,
                    properties(endNode(r)) AS target_properties
                ORDER BY CASE WHEN type(r) = 'UNTYPED_RELATION' THEN 1 ELSE 0 END,
                         coalesce(r.weight, 0) DESC
                LIMIT $limit
                """,
                question=question,
                limit=limit,
            )

        return "\n".join(self._format_context_row(row) for row in rows)

    @staticmethod
    def _format_context_row(row: dict[str, Any]) -> str:
        """Format one relation row into a human-readable directed sentence."""
        source = row.get("path_source_name") or row.get("source_name") or "Unknown"
        target = row.get("path_target_name") or row.get("target_name") or "Unknown"
        relation = row.get("relation_name") or row.get("relation_type") or "RELATED_TO"
        stored_source = row.get("source_name") or source
        stored_target = row.get("target_name") or target

        if source == stored_source and target == stored_target:
            return f"{source} --({relation})--> {target}"
        elif source == stored_target and target == stored_source:
            return f"{source} <--({relation})-- {target}"
        return f"{source} --({relation})-- {target}"
