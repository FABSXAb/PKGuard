import argparse
import json
import os
from pathlib import Path

from . import acl_explicit_u as _au, orgaccess_explicit as _oe
from .dm_client import DMClient
from .graphs import build_orgaccess_graph, build_bird_graph


def build_graph(dataset, req, unaddressed_within):
    if dataset == "OrgAccess":
        return build_orgaccess_graph(json.loads(req["permissions"]),
                                     role=req.get("role", "role"),
                                     unaddressed_within=unaddressed_within)
    return build_bird_graph(req, unaddressed_within=unaddressed_within)


def state_of(dataset, req):
    if dataset == "OrgAccess":
        return f"{_oe.SYS}\n\nPERMISSIONS: {req['permissions']}\nUSER QUERY: {req['query']}"
    return f"{_au.SYS}\n\nROLE: {req.get('role')}\n\nREQUEST: {req.get('question')}"


def evaluate_dist(g, dists, verdicts, unaddressed_within, probabilistic):
    from pipeline.evaluator import aggregate_graph, aggregate_graph_prob
    if probabilistic:
        post = aggregate_graph_prob(g, dists, unaddressed_within)
        return max(post, key=post.get), verdicts, post
    return aggregate_graph(g, verdicts, unaddressed_within), verdicts, None


def _is_opaque(g, fid):
    from pipeline.kg import REL_OPAQUE, REL_ALLOWS, REL_DENIES
    if g.target(fid, REL_OPAQUE) is not None:
        return True
    for nid in g.targets(fid, REL_ALLOWS) + g.targets(fid, REL_DENIES):
        if (g.node(nid) or {}).get("generated"):
            return True
    return False


def run_one_R(dataset, req, client, cache_extra, unaddressed_within, probabilistic):
    from pipeline.kg import REL_ALLOWS, REL_DENIES, NODE_STATE
    g = build_graph(dataset, req, unaddressed_within)
    fields = g.nodes_of_type("field")
    recs = []


    q2, opaque_q, probs = {}, {}, {}
    for i, fid in enumerate(fields):
        lab = (g.node(fid) or {}).get("label", fid)
        if _is_opaque(g, fid):
            q2[f"d{i}"] = {"type": "noul",
                           "instructions": (f"Does the user's request demand anything that the "
                                            f"permission field `{lab}` governs? Answer true if "
                                            f"the request touches this dimension at all, false if "
                                            f"the request is entirely silent about it.")}
            opaque_q[fid] = f"d{i}"
        else:
            for nid in g.targets(fid, REL_DENIES):
                nlab = (g.node(nid) or {}).get("label", nid)

                q2[nid] = {"type": "noul",
                           "instructions": f"Does the request need to read `{nlab}`?"}
    r2 = None
    if q2:
        r2 = client.decide("kg_touch", state_of(dataset, req), q2,
                           cache_key_extra={**cache_extra, "step": "touch"})
        recs.append(r2)
    a2 = (r2 or {}).get("answers") or {}


    verdicts = {}
    for i, fid in enumerate(fields):
        if _is_opaque(g, fid):
            continue
        hit = any(((a2.get(n) or {}).get("noul") or 0.0) >= 0.5
                  for n in g.targets(fid, REL_DENIES))
        verdicts[fid] = "violated" if hit else "met"

    touched = [fid for fid in fields
               if _is_opaque(g, fid) and ((a2.get(opaque_q[fid]) or {}).get("noul") or 0.0) >= 0.5]
    for fid in fields:
        if _is_opaque(g, fid) and fid not in touched:
            verdicts[fid] = "unaddressed"

    q3 = {}
    for i, fid in enumerate(touched):
        lab = (g.node(fid) or {}).get("label", fid)
        vals = [(g.node(n) or {}).get("label", n) for n in g.targets(fid, REL_ALLOWS)]
        q3[f"e{i}"] = {"type": "choice",
                       "instructions": (f"The permission `{lab}` allows: "
                                        f"{', '.join('`%s`' % c for c in vals)}. "
                                        f"Given ONLY this permission, is the request's demand:"),
                       "criteria": {"met": "fully satisfied by this permission",
                                    "partial": "only partly satisfied by this permission",
                                    "violated": "contradicts or exceeds this permission"}}
    if q3:
        r3 = client.decide("kg_state", state_of(dataset, req), q3,
                           cache_key_extra={**cache_extra, "step": "state"})
        recs.append(r3)
        a3 = r3.get("answers") or {}
        for i, fid in enumerate(touched):
            a = a3.get(f"e{i}") or {}
            verdicts[fid] = a.get("choice") or "violated"

            pr = a.get("probabilities")
            if pr:
                probs[fid] = {k: float(v) for k, v in pr.items()}


    dists = {}
    for fid in fields:
        ch = verdicts.get(fid, "unaddressed")
        if fid in probs:
            dists[fid] = dict(probs[fid])
            dists[fid].setdefault("unaddressed", 0.0)
        else:
            dists[fid] = {"met": 0.0, "partial": 0.0, "violated": 0.0, "unaddressed": 0.0}
            dists[fid][ch] = 1.0
    dec, verdicts, post = evaluate_dist(g, dists, verdicts, unaddressed_within, probabilistic)
    return dec, verdicts, post, g, q3, recs

def main():
    parser = argparse.ArgumentParser(description="Run PKGuard on normalized JSONL requests.")
    parser.add_argument("--dataset", choices=("OrgAccess", "BIRD-AC", "eICU-AC"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dm")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    cache = args.cache_dir or args.output.parent / ".pkguard-cache"
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    client = DMClient(args.endpoint, args.model, cache_dir=cache,
                      api_key=os.environ.get("DECISION_API_KEY"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as dest:
        for i, line in enumerate(source):
            if not line.strip():
                continue
            req = json.loads(line)
            sid = req.get("sample_id", str(i))
            decision, states, scores, *_ = run_one_R(
                args.dataset, req, client, {"d": args.dataset, "s": sid},
                unaddressed_within=True, probabilistic=args.dataset == "OrgAccess",
            )
            print(json.dumps({"sample_id": sid, "decision": decision,
                              "local_states": states, "scores": scores}, ensure_ascii=False), file=dest)


if __name__ == "__main__":
    main()
