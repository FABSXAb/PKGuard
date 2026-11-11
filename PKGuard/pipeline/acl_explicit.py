import re
from .reads import parse_grants

def candidate_resources(req):
    v = req.get("resource_vocab") or []
    if v:
        return list(dict.fromkeys(v))
    out = []
    for m in re.finditer(r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*;", req.get("schema_text") or "",
                         re.I | re.S):
        t, body = m.group(1), m.group(2)
        out.append(t)
        for cm in re.finditer(r"(\w+)\s+\w", body):
            out.append(f"{t}.{cm.group(1)}")
    return list(dict.fromkeys(out))


def granted_set(req):
    pm = req.get("policy_map")
    if pm:
        A = set()
        for t, cols in pm.items():
            A.add(t)
            for c in cols:
                A.add(f"{t}.{c}")
        return A
    g = parse_grants(req.get("policy_text") or "")
    role = req.get("role")
    if role not in g:
        cand = [u for u in g if u.endswith(role or "") or (role or "").endswith(u)]
        role = cand[0] if cand else role
    A = set()
    for t, cols in g.get(role, {}).items():
        A.add(t)
        if cols is None:


            for c in candidate_resources(req):
                if c.startswith(f"{t}."):
                    A.add(c)
        else:
            for c in cols:
                A.add(f"{t}.{c}")
    return A
