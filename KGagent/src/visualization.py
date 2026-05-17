"""Pyvis graph rendering helpers."""

from __future__ import annotations

from pyvis.network import Network


NODE_STYLE = {
    "Person": {"color": "#d94a4a", "shape": "dot"},
    "Organization": {"color": "#3b82f6", "shape": "box"},
    "Work": {"color": "#22c55e", "shape": "ellipse"},
}


def node_style(labels: list[str]) -> dict[str, str]:
    """Pick color/shape based on the most specific known label."""
    for label in ("Person", "Organization", "Work"):
        if label in labels:
            return NODE_STYLE[label]
    return {"color": "#9ca3af", "shape": "dot"}


def build_pyvis_html(subgraph: dict[str, list[dict]]) -> str:
    """Build a self-contained Pyvis HTML document for Streamlit embedding."""
    network = Network(
        height="700px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
    )
    network.barnes_hut(gravity=-4500, central_gravity=0.25, spring_length=140)

    for node in subgraph["nodes"]:
        labels = node.get("labels", [])
        props = node.get("properties", {})
        style = node_style(labels)
        title_lines = [f"{key}: {value}" for key, value in props.items() if value not in (None, "", [])]
        network.add_node(
            node["id"],
            label=node.get("name", ""),
            title="<br>".join(title_lines),
            color=style["color"],
            shape=style["shape"],
        )

    for rel in subgraph["relationships"]:
        network.add_edge(
            rel["source"],
            rel["target"],
            label=rel.get("name", rel.get("type", "")),
            title=rel.get("type", ""),
            arrows="to",
        )

    network.set_options(
        """
        {
          "nodes": {"font": {"size": 18, "face": "Microsoft YaHei"}},
          "edges": {
            "font": {"size": 12, "align": "middle", "face": "Microsoft YaHei"},
            "smooth": {"type": "dynamic"}
          },
          "physics": {"stabilization": true}
        }
        """
    )
    return network.generate_html(notebook=False)
