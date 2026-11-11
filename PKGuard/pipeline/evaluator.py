import math
from .kg import KG, REL_YIELDS, REL_ELSE

def aggregate_graph(g: KG, verdicts: dict, unaddressed_within: bool = True) -> str:
    vals = list(verdicts.values())
    if "violated" in vals:
        return g.target("agg:OR", REL_YIELDS)
    within = all(v == "met" or (v == "unaddressed" and unaddressed_within) for v in vals)
    return g.target("agg:AND", REL_YIELDS if within else REL_ELSE)


def aggregate_graph_prob(g: KG, dists: dict, unaddressed_within: bool = True) -> dict:
    log_no_breach = log_within = 0.0
    for d in dists.values():
        pb = float(d.get("violated", 0.0))
        pw = float(d.get("met", 0.0)) + (1.0 if unaddressed_within else 0.0) * float(d.get("unaddressed", 0.0))
        log_no_breach += math.log(max(1e-12, 1.0 - pb))
        log_within += math.log(max(1e-12, pw))
    rej = g.target("agg:OR", REL_YIELDS)
    full = g.target("agg:AND", REL_YIELDS)
    part = g.target("agg:AND", REL_ELSE)
    p_rej = 1.0 - math.exp(log_no_breach)
    p_full = math.exp(log_within)
    return {rej: p_rej, full: p_full, part: max(0.0, 1.0 - p_rej - p_full)}
