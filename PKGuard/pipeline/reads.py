import re

_QUOTED = r'(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[^()])*'


_OBJECT = r'(?:"[^"]*"|`[^`]*`|\[[^\]]*\]|[^\s;]+)'


_GRANT_RE2 = re.compile(
    r"GRANT\s+SELECT\s*(?:\((" + _QUOTED + r")\))?\s+ON\s+(" + _OBJECT + r")\s+TO\s+(" + _OBJECT + r")",
    re.I)


def split_ident_list(s):
    out, cur, q = [], "", None
    for ch in s:
        if q:
            if ch == q:
                q = None
            cur += ch
        elif ch in '"`[':
            q = "]" if ch == "[" else ch
            cur += ch
        elif ch == ",":
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [x.strip().strip('"`[]').lower() for x in out if x.strip()]


def parse_grants(text):
    g = {}
    for m in _GRANT_RE2.finditer(text or ""):
        cols, tbl, user = m.group(1), m.group(2), m.group(3)
        tbl = tbl.split(".")[-1].strip('"`[]').lower()
        user = user.strip('"`[]')
        d = g.setdefault(user, {})
        if cols is None:
            d[tbl] = None
        else:
            cur = d.get(tbl)
            if cur is None and tbl in d:
                continue
            d[tbl] = (cur or set()) | set(split_ident_list(cols))
    return g
