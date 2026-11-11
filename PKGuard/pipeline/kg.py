from __future__ import annotations

from collections import defaultdict


class KG:

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

        self._out: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._in: dict[tuple[str, str], list[str]] = defaultdict(list)


    def add_node(self, nid: str, ntype: str, **attrs):
        cur = self.nodes.setdefault(nid, {"type": ntype})
        cur.setdefault("type", ntype)
        cur.update(attrs)
        return nid

    def add_edge(self, h: str, r: str, t: str):
        self.edges.append((h, r, t))
        self._out[(h, r)].append(t)
        self._in[(r, t)].append(h)


    def query(self, h=None, r=None, t=None) -> list[tuple[str, str, str]]:
        if h is not None and r is not None:
            return [(h, r, x) for x in self._out.get((h, r), []) if t is None or x == t]
        if r is not None and t is not None:
            return [(x, r, t) for x in self._in.get((r, t), []) if h is None or x == h]
        if h is not None and t is not None:
            return [(x, y, z) for (x, y, z) in self.edges if x == h and z == t]
        return [(x, y, z) for (x, y, z) in self.edges
                if (h is None or x == h) and (r is None or y == r) and (t is None or z == t)]

    def one(self, h=None, r=None, t=None):
        res = self.query(h=h, r=r, t=t)
        if not res:
            return None
        if len(res) > 1:
            raise ValueError(f"Expected exactly one match, found {len(res)}: {res[:3]}")
        return res[0]

    def targets(self, h: str, r: str) -> list[str]:
        return list(self._out.get((h, r), []))

    def target(self, h: str, r: str):
        ts = self._out.get((h, r), [])
        return ts[0] if ts else None

    def sources(self, r: str, t: str) -> list[str]:
        return list(self._in.get((r, t), []))

    def nodes_of_type(self, ntype: str) -> list[str]:
        return [n for n, a in self.nodes.items() if a.get("type") == ntype]

    def node(self, nid: str) -> dict | None:
        return self.nodes.get(nid)


    def traverse(self, start: str, *rels: str) -> list[str]:
        frontier = [start]
        for r in rels:
            nxt = []
            for n in frontier:
                nxt.extend(self._out.get((n, r), []))
            frontier = list(dict.fromkeys(nxt))
            if not frontier:
                return []
        return frontier


    def stats(self) -> dict:
        from collections import Counter
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": dict(Counter(a.get("type") for a in self.nodes.values())),
            "relations": dict(Counter(r for _, r, _ in self.edges)),
        }


REL_ALLOWS = "allows"
REL_DENIES = "denies"


REL_OPAQUE = "opaque"

REL_BOUND_BY = "bound_by"

REL_RDF_TYPE = "rdf:type"
REL_COMPARATOR = "comparator"

REL_FEEDS = "feeds"
REL_BLOCKS = "blocks"
REL_YIELDS = "yields"

REL_ELSE = "else"

REL_CAUSED_BY = "caused_by"

NODE_ROLE, NODE_FIELD, NODE_VALUE = "role", "field", "value"
NODE_STATE, NODE_AGG, NODE_DECISION = "state", "agg", "decision"


STATE_IDS = {s: f"state:{s}" for s in ("met", "partial", "violated", "unaddressed")}


def build_rule_subgraph(kg: KG, decision_ids: dict, unaddressed_within: bool = True):
    for s, nid in STATE_IDS.items():
        kg.add_node(nid, NODE_STATE)
    kg.add_node("agg:OR", NODE_AGG)
    kg.add_node("agg:AND", NODE_AGG)
    for d, nid in decision_ids.items():
        kg.add_node(nid, NODE_DECISION)


    kg.add_edge(STATE_IDS["violated"], REL_FEEDS, "agg:OR")
    kg.add_edge("agg:OR", REL_YIELDS, decision_ids["rejected"])


    kg.add_edge(STATE_IDS["met"], REL_FEEDS, "agg:AND")
    kg.add_edge(STATE_IDS["unaddressed"], REL_FEEDS if unaddressed_within else REL_BLOCKS,
                "agg:AND")
    kg.add_edge(STATE_IDS["partial"], REL_BLOCKS, "agg:AND")
    kg.add_edge("agg:AND", REL_YIELDS, decision_ids["full"])
    kg.add_edge("agg:AND", REL_ELSE, decision_ids["partial"])
    return kg
