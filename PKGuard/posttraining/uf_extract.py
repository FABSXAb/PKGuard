import json, os, random, sys, time
import numpy as np, torch
from pipeline.dm_runtime import model_class, runtime

sdpa_backends = runtime.sdpa_backends
from pipeline import unified_format as uf

ROOT = os.environ.get("PKGUARD_DATA_DIR", "data")
CACHE_DIR = os.environ.get("PKGUARD_CACHE_DIR", "calib")
os.makedirs(CACHE_DIR, exist_ok=True)
RAW_ORG = os.environ.get("PKGUARD_ORG_DIR", os.path.join(ROOT, "OrgAccess", "jsonl"))
BATCH = int(os.environ.get("UF_BATCH", "8"))

DS = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else (316 if DS == "eicu" else 2000)
out = f"{CACHE_DIR}/uf_{DS}_{N}.npz"
if os.path.exists(out):
    print("Cache already exists:", out); sys.exit(0)


samples = []
if DS == "bird":
    rows = [json.loads(l) for l in open(f"{ROOT}/BIRD-AC/requests.jsonl")]
    rng = random.Random(0); idx = list(range(len(rows))); rng.shuffle(idx)
    for i in sorted(idx[:N]):
        x = rows[i]; sid = x["sample_id"]
        items = uf.bird_items(x)
        samples.append((sid, uf.state(x.get("role", "role"), items, x["question"]), items))
elif DS == "eicu":
    rows = [json.loads(l) for l in open(f"{ROOT}/eICU-AC/requests.jsonl")]
    for x in rows:
        for kk in ("policy_map", "resource_vocab", "catalog"):
            if isinstance(x.get(kk), str):
                x[kk] = json.loads(x[kk])
    for x in rows:
        sid = x["sample_id"]
        items = uf.bird_items(x)
        samples.append((sid, uf.state(x.get("role", "role"), items, x["question"]), items))
else:
    for sp in ["easy", "medium", "hard"]:
        rr = [json.loads(l) for l in open(f"{RAW_ORG}/{sp}.jsonl")]
        if N >= 55800:
            take = list(range(len(rr)))
        else:
            rng = random.Random(0); idx = list(range(len(rr))); rng.shuffle(idx)
            take = sorted(idx[:N // 3])
        for i in take:
            x = rr[i]
            items = uf.orgaccess_items(json.loads(x["permissions"]))
            samples.append((f"org-{sp}-{i:06d}",
                            uf.state(x.get("role", "role"), items, x["query"]),
                            items))

t0 = time.time()
m = model_class().from_pretrained(os.environ["PKGUARD_DM_MODEL"], device=os.environ.get("PKGUARD_DEVICE", "cuda:0"))
print(f"Loaded model in {time.time()-t0:.0f}s. Samples: {len(samples)}", flush=True)

SID, FIELD, KIND, H, LD = [], [], [], [], []


def flush(buf):
    if not buf:
        return
    buf.sort(key=lambda r: len(r[3]))
    c0 = 0
    while c0 < len(buf):
        w = len(buf[c0][3])
        bs = 64 if w < 512 else 32 if w < 1024 else 8 if w < 2048 else 4
        ch = buf[c0:c0 + bs]; c0 += bs
        w = max(len(r[3]) for r in ch)
        ids = torch.full((len(ch), w), m.pad_id, dtype=torch.long)
        msk = torch.zeros((len(ch), w), dtype=torch.long)
        for j, r in enumerate(ch):
            ids[j, w - len(r[3]):] = torch.as_tensor(r[3], dtype=torch.long)
            msk[j, w - len(r[3]):] = 1
        ids, msk = ids.to(m.device), msk.to(m.device)
        with torch.inference_mode(), sdpa_backends(m.device):
            emb = m.backbone.get_input_embeddings()(ids)
            hh = m.encoder(emb, msk)[:, -1]
            lg = (hh.float() @ m.readout.T).float().cpu().numpy()
        hid = hh.float().cpu().numpy().astype(np.float16)
        del ids, msk, emb, hh
        for j, r in enumerate(ch):
            sid, f, kind, _, cnt = r
            SID.append(sid); FIELD.append(f); KIND.append(kind)
            H.append(hid[j])
            d = lg[j][:cnt] - lg[j][0]
            LD.append([float(d[0]), float(d[1]), float(d[2]) if cnt > 2 else 0.0])
    torch.cuda.empty_cache()


MAXLEN = int(os.environ.get("UF_MAXLEN", "4096"))
buf = []
n_skip = n_skip_s = 0
for n, (sid, st, items) in enumerate(samples):
    for kind, q in ((2, uf.demand_questions(items)), (3, uf.eval_questions(items))):
        p = m.prepare(st, q)
        for k in p.runnable:
            i = int(k[1:])
            if len(p.sequences[k]) > MAXLEN:
                n_skip += 1; continue
            buf.append((sid, items[i][0], kind, p.sequences[k],
                        len(p.questions[k].keys)))
    if len(buf) >= BATCH * 4:
        flush(buf); buf = []
    if (n + 1) % 200 == 0:
        print(f"  {n+1}/{len(samples)} samples, {len(SID)} rows, {time.time()-t0:.0f}s elapsed", flush=True)
flush(buf)
np.savez(out, SID=np.array(SID), FIELD=np.array(FIELD), KIND=np.array(KIND, dtype=np.int8),
         H=np.array(H, dtype=np.float16), LD=np.array(LD, dtype=np.float32))
print(f"Saved {len(SID)} rows from {len(set(SID))} samples in {time.time()-t0:.0f}s: {out}")
