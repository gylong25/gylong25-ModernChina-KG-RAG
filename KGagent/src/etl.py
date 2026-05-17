"""ETL utilities for converting the JSON file into Neo4j nodes and relations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.graph_store import Neo4jGraphStore


PERSON_PROPERTY_KEYS = {
    "国籍": "nationality",
    "民族": "ethnicity",
    "出生地": "birth_place",
    "出生日期": "birth_date",
    "死亡日期": "death_date",
    "工作职责": "occupation",
}


def load_people_json(path: Path) -> list[dict[str, Any]]:
    """Load the source JSON with tolerant encoding handling.

    The provided file is UTF-8. The fallback list makes the loader friendlier to
    future exports that may come from Windows tools.
    """
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def as_list(value: Any) -> list[str]:
    """Normalize strings/lists into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def relation_type(raw_name: str) -> str:
    """Convert a Chinese relation label into a safe Cypher relationship type."""
    if raw_name == "未知":
        return "UNTYPED_RELATION"
    # Neo4j relationship types must be ASCII-safe when interpolated plainly.
    # Chinese relation labels are encoded as Unicode code points, while the
    # original readable label is preserved on r.name.
    slug = "".join(
        char if re.match(r"[0-9A-Za-z_]", char) else f"U{ord(char):X}"
        for char in raw_name.strip()
    )
    return f"RELATED_{slug}".upper()


def person_properties(record: dict[str, Any]) -> dict[str, Any]:
    """Extract person node properties from one JSON record."""
    aliases = as_list(record.get("附加名称"))
    props: dict[str, Any] = {
        "name": str(record["中文名"]).strip(),
        "aliases": aliases,
    }
    for source_key, target_key in PERSON_PROPERTY_KEYS.items():
        if record.get(source_key):
            props[target_key] = record[source_key]
    return props


def import_json_to_neo4j(path: Path, graph: "Neo4jGraphStore") -> dict[str, int]:
    """Parse JSON data and upsert all graph elements into Neo4j."""
    people = load_people_json(path)
    counters = {"people": 0, "organizations": 0, "works": 0, "relations": 0}

    for record in people:
        if not record.get("中文名"):
            continue

        person = person_properties(record)
        graph.merge_person(person)
        counters["people"] += 1

        # Schools are modeled as Organization nodes.
        for school in as_list(record.get("毕业于")):
            graph.merge_organization(school)
            graph.merge_relation(
                person["name"],
                school,
                "GRADUATED_FROM",
                target_label="Organization",
                display_name="毕业于",
            )
            counters["organizations"] += 1
            counters["relations"] += 1

        # Works are modeled as Work nodes.
        for work in as_list(record.get("作品")):
            graph.merge_work(work)
            graph.merge_relation(
                person["name"],
                work,
                "CREATED_WORK",
                target_label="Work",
                display_name="作品",
            )
            counters["works"] += 1
            counters["relations"] += 1

        # Related people are merged lazily to avoid dangling relationship ends.
        related_people = record.get("相关人物") or {}
        if isinstance(related_people, dict):
            for raw_relation, names in related_people.items():
                rel_type = relation_type(raw_relation)
                for target_name in as_list(names):
                    graph.merge_person({"name": target_name, "aliases": []})
                    graph.merge_relation(
                        person["name"],
                        target_name,
                        rel_type,
                        target_label="Person",
                        display_name=raw_relation,
                    )
                    counters["relations"] += 1

    return counters
