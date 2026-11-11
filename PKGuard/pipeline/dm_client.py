import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class DMFailure(RuntimeError):
    pass


class LedgerCache:

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self._lock = threading.Lock()
        n = 0
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    k = r.get("cache_key")
                    if k and "answers" in r:
                        self.index[k] = r
                        n += 1
        self.loaded = n

    def get(self, key):
        return self.index.get(key)

    def put(self, key, rec):
        self.index[key] = rec
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class DMClient:
    def __init__(self, endpoint, model, cache_dir=None, ledger_path=None, api_key=None,
                 timeout=300):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._lock = threading.Lock()
        if cache_dir is not None:

            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.ledger = Path(ledger_path) if ledger_path else None
            self.store = None
        else:

            self.cache_dir = None
            self.ledger = Path(ledger_path)
            self.store = LedgerCache(self.ledger)

    def decide(self, stage, state, questions, cache_key_extra=None):
        body = {"model": self.model, "state": state, "questions": questions}
        key = hashlib.sha256(json.dumps(
            {**body, "_extra": cache_key_extra}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

        if self.store is not None:
            rec = self.store.get(key)
        else:
            f = self.cache_dir / f"{key}.json"
            rec = json.loads(f.read_text()) if f.exists() else None
        if rec is not None:
            rec = dict(rec)
            rec["_cache_hit"] = True
            return rec

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.endpoint + "/v1/systemone",
                                     data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers=headers)
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            self._log({"stage": stage, "cache_key": key, "HTTP_status": e.code, "error": err[:500]})
            raise DMFailure(f"HTTP {e.code}")
        except Exception as e:
            self._log({"stage": stage, "cache_key": key, "error": type(e).__name__})
            raise DMFailure(type(e).__name__)
        elapsed = time.monotonic() - started

        usage = resp.get("usage", {}) or {}
        rec = {"stage": stage, "cache_key": key, "seconds": elapsed,
               "input_tokens": usage.get("input_tokens", 0),
               "output_tokens": usage.get("output_tokens", 0),
               "answers": resp.get("answers", {}),
               "_cache_hit": False}
        if self.store is not None:
            self.store.put(key, rec)
        else:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(rec, ensure_ascii=False))
            self._log(rec)
        return rec

    def _log(self, rec):
        if self.ledger is None:
            return
        with self._lock:
            with self.ledger.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
