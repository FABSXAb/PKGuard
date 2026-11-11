from __future__ import annotations

import json

from .kg import (KG, NODE_ROLE, NODE_FIELD, NODE_VALUE, REL_ALLOWS, REL_DENIES,
                 REL_OPAQUE, REL_BOUND_BY, REL_RDF_TYPE, REL_COMPARATOR,
                 build_rule_subgraph)

ORG_DECISIONS = {"rejected": "decision:rejected", "full": "decision:full",
                 "partial": "decision:partial"}
BIRD_DECISIONS = {"rejected": "decision:DENY", "full": "decision:PERMIT",
                  "partial": "decision:PERMIT"}

_LEQ = ("timeout", "limit", "max", "cap", "threshold", "rate", "budget", "ceiling", "quota")
_GEQ = ("required", "quorum", "min", "minimum", "atleast", "at_least", "floor")


def infer_comparator(field: str):
    s = field.lower()
    if any(t in s for t in _GEQ):
        return "cmp:geq"
    if any(t in s for t in _LEQ):
        return "cmp:leq"
    return None


def infer_type(v) -> str:
    if isinstance(v, bool):
        return "type:bool"
    if isinstance(v, int):
        return "type:numeric"
    if isinstance(v, list):
        return "type:list"
    return "type:str"


def _add_value(g: KG, fid: str, nid: str, label: str, rel: str, generated: bool = False):
    g.add_node(nid, NODE_VALUE, label=label, generated=generated)
    g.add_edge(fid, rel, nid)


def _bind_field(g: KG, rid: str, fid: str, label: str, t: str, cmp_=None, **attrs):
    g.add_node(fid, NODE_FIELD, label=label, **attrs)
    g.add_edge(rid, REL_BOUND_BY, fid)
    g.add_edge(fid, REL_RDF_TYPE, t)
    if cmp_:
        g.add_edge(fid, REL_COMPARATOR, cmp_)


def build_orgaccess_graph(permissions: dict, role: str = "role",
                          unaddressed_within: bool = True) -> KG:
    g = KG()
    rid = f"role:{role}"
    g.add_node(rid, NODE_ROLE)
    for f, v in permissions.items():
        fid = f"field:{f}"
        t = infer_type(v)
        cmp_ = infer_comparator(f)
        _bind_field(g, rid, fid, f, t, cmp_, value=v)

        if t in ("type:list", "type:str"):

            g.add_edge(fid, REL_OPAQUE, "true")
        if t == "type:list":
            for u in v:
                _add_value(g, fid, f"{fid}/v:{u}", str(u), REL_ALLOWS)
            _add_value(g, fid, f"{fid}/outside", "anything not in this list", REL_DENIES)
        elif t == "type:bool":


            _add_value(g, fid, f"{fid}/true", f"that `{f}` be true",
                       REL_ALLOWS if v else REL_DENIES, generated=True)
            _add_value(g, fid, f"{fid}/false", f"that `{f}` be false",
                       REL_DENIES if v else REL_ALLOWS, generated=True)
        elif t == "type:numeric":

            _add_value(g, fid, f"{fid}/equal", f"exactly {v}", REL_ALLOWS, generated=True)
            if cmp_ == "cmp:leq":
                _add_value(g, fid, f"{fid}/below", f"a value below {v}", REL_ALLOWS, generated=True)
                _add_value(g, fid, f"{fid}/above", f"a value above {v}", REL_DENIES, generated=True)
            elif cmp_ == "cmp:geq":
                _add_value(g, fid, f"{fid}/above", f"a value above {v}", REL_ALLOWS, generated=True)
                _add_value(g, fid, f"{fid}/below", f"a value below {v}", REL_DENIES, generated=True)
            else:
                _add_value(g, fid, f"{fid}/outside", "a different value", REL_DENIES, generated=True)
        else:
            _add_value(g, fid, f"{fid}/equal", f"`{v}`", REL_ALLOWS)
            _add_value(g, fid, f"{fid}/outside", "a different value", REL_DENIES, generated=True)

    build_rule_subgraph(g, ORG_DECISIONS, unaddressed_within)
    return g


def build_bird_graph(req: dict, unaddressed_within: bool = True) -> KG:
    from .acl_explicit import candidate_resources, granted_set

    g = KG()
    rid = f"role:{req.get('role')}"
    g.add_node(rid, NODE_ROLE)
    A = granted_set(req)
    for r in candidate_resources(req):
        table = r.split(".")[0]
        fid = f"field:{table}"
        if g.node(fid) is None:
            _bind_field(g, rid, fid, table, "type:subset", "cmp:subset")
        _add_value(g, fid, f"value:{r}", r, REL_ALLOWS if r in A else REL_DENIES)
    build_rule_subgraph(g, BIRD_DECISIONS, unaddressed_within)
    return g
