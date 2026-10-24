"""Paper knowledge graph store (node-link JSON on disk, NetworkX in memory).

`GraphStore` is the seam where a Neo4j implementation can replace this one without touching agents.
The graph itself is built by `graph_builder.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import networkx as nx

MAX_NODES = 20


class GraphStore(Protocol):
    def resolve(self, text: str) -> str | None: ...

    def neighbors(self, node_id: str, relation: str | None = None, limit: int = MAX_NODES) -> list[dict]: ...


class JsonGraphStore:
    def __init__(self, path: Path):
        self.graph = nx.MultiDiGraph()
        if path.exists():
            self.graph = nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), directed=True,
                                            multigraph=True, edges="edges")

    def resolve(self, text: str) -> str | None:
        needle = text.lower().strip()
        if not needle:
            return None
        for node, data in self.graph.nodes(data=True):
            names = [node, data.get("title", ""), *data.get("aliases", [])]
            if any(needle == n.lower() or (len(needle) > 4 and needle in n.lower()) for n in names):
                return node
        return None

    def neighbors(self, node_id: str, relation: str | None = None, limit: int = MAX_NODES) -> list[dict]:
        out = []
        edges = [(u, v, d, "out") for u, v, d in self.graph.out_edges(node_id, data=True)]
        edges += [(u, v, d, "in") for u, v, d in self.graph.in_edges(node_id, data=True)]
        for u, v, d, direction in edges:
            if relation and d.get("type") != relation:
                continue
            other = v if direction == "out" else u
            node = self.graph.nodes[other]
            out.append({"relation": d.get("type"), "direction": direction, "node": other,
                        "title": node.get("title", other), "venue": node.get("venue"), "year": node.get("year"),
                        "note": d.get("note", "")})
            if len(out) >= min(limit, MAX_NODES):
                break
        return out
