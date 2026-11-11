import json, os, random, sys, time
import numpy as np, torch
from pipeline.dm_runtime import model_class, runtime

sdpa_backends = runtime.sdpa_backends
from pipeline import unified_format as uf
from pipeline.acl_explicit import candidate_resources, granted_set

ROOT = os.environ.get("PKGUARD_DATA_DIR", "data")
CACHE_DIR = os.environ.get("PKGUARD_CACHE_DIR", "calib")
os.makedirs(CACHE_DIR, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
out = f"{CACHE_DIR}/ufb_{N}.npz"
if os.path.exists(out): print("Cache already exists:", out); sys.exit(0)
BATCH = int(os.environ.get("UF_BATCH", "16"))

rows = [json.loads(l) for l in open(f"{ROOT}/BIRD-AC/requests.jsonl")]
rng = random.Random(0); idx = list(range(len(rows))); rng.shuffle(idx)
samples = []
for i in sorted(idx[:N]):
    x = rows[i]; sid = x["sample_id"]
    items = uf.bird_items(x)
    A = granted_set(x); cand = candidate_resources(x)
    st = uf.state(x.get("role", "role"), items, x["question"])
    for t, _ in items:
        S = {c for c in cand if c.split(".")[0] == t}
        denied = sorted(S - A)
        if not denied: continue
        q = {"type": "noul",
             "instructions": (f"The role's permission `{t}` does NOT allow: "
                              + ", ".join(f"`{d}`" for d in denied)
                              + ". Does the request need to read any of them? "
                                "Answer true if the request needs at least one of these, "
                                "false if it needs none of them.")}
        samples.append((sid, t, st, q, len(denied)))

t0 = time.time()
m = model_class().from_pretrained(os.environ["PKGUARD_DM_MODEL"], device=os.environ.get("PKGUARD_DEVICE", "cuda:0"))
print(f"Loaded model in {time.time()-t0:.0f}s. Samples: {len(samples)}. "
      f"Mean number of prohibited columns: {np.mean([s[4] for s in samples]):.1f}", flush=True)
SID, FIELD, H, LD = [], [], [], []
buf = []
def flush(buf):
    if not buf: return
    buf.sort(key=lambda r: len(r[3]))
    for c0 in range(0, len(buf), BATCH):
        ch = buf[c0:c0+BATCH]; w = max(len(r[3]) for r in ch)
        ids = torch.full((len(ch), w), m.pad_id, dtype=torch.long)
        msk = torch.zeros((len(ch), w), dtype=torch.long)
        for j, r in enumerate(ch):
            ids[j, w-len(r[3]):] = torch.as_tensor(r[3], dtype=torch.long); msk[j, w-len(r[3]):] = 1
        ids, msk = ids.to(m.device), msk.to(m.device)
        with torch.inference_mode(), sdpa_backends(m.device):
            emb = m.backbone.get_input_embeddings()(ids); hh = m.encoder(emb, msk)[:, -1]
            lg = (hh.float() @ m.readout.T).float().cpu().numpy()
        hid = hh.float().cpu().numpy().astype(np.float16); del ids, msk, emb, hh
        for j, r in enumerate(ch):
            sid, t, _, _, nd = r
            SID.append(sid); FIELD.append(t); H.append(hid[j])
            LD.append(float(lg[j][1] - lg[j][0]))
    torch.cuda.empty_cache()
n_fail = 0
for n, (sid, t, st, q, nd) in enumerate(samples):
    p = m.prepare(st, {"q": q})
    if "q" in p.runnable:
        buf.append((sid, t, None, p.sequences["q"], nd))
    else:
        n_fail += 1
        if n_fail <= 3: print(f"  FAIL {sid} {t}: {p.errors.get('q')}", flush=True)
    if len(buf) >= BATCH*8: flush(buf); buf = []
    if (n+1) % 500 == 0: print(f"  {n+1}/{len(samples)} samples, {len(SID)} rows, {time.time()-t0:.0f}s elapsed", flush=True)
flush(buf)
np.savez(out, SID=np.array(SID), FIELD=np.array(FIELD), H=np.array(H, dtype=np.float16),
         LD=np.array(LD, dtype=np.float32))
print(f"Saved {len(SID)} rows. Failures: {n_fail}. Paired samples: {len(samples)}. Elapsed: {time.time()-t0:.0f}s. Output: {out}")
