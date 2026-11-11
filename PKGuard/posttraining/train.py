import argparse
import json
import random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pipeline.labelmap import normalize

CACHE_DIR = Path("calib")
DATA_DIR = Path("data")
DEV = "cuda:0"; EPS = 1e-7
Y3 = {"rejected": 0, "partial": 2, "full": 1}
SLOT = (3, 4)


class Head(nn.Module):
    def __init__(s, dim=256):
        super().__init__(); s.W1 = nn.Linear(5120, dim); s.W2 = nn.Linear(dim, 255)
        nn.init.zeros_(s.W2.weight); nn.init.zeros_(s.W2.bias)
    def corr(s, h): return s.W2(F.relu(s.W1(h)))


def _split_universe(sids, frac_tr=.6, frac_te=.8):
    o = sorted(range(len(sids))); random.Random(0).shuffle(o)
    n = len(sids)
    return set(sids[i] for i in o[:int(n * frac_tr)]), set(sids[i] for i in o[int(n * frac_te):])


def load_bird_shared(cache2, cache3):
    z2 = np.load(f"{CACHE_DIR}/{cache2}", allow_pickle=True)
    z3 = np.load(f"{CACHE_DIR}/{cache3}", allow_pickle=True)
    lab = json.load(open(f"{DATA_DIR}/BIRD-AC/evaluation_private/labels.json"))
    S2 = [str(s) for s in z2["SID"]]; F2 = [str(f) for f in z2["FIELD"]]; K2 = z2["KIND"].astype(int)
    S3 = [str(s) for s in z3["SID"]]; F3 = [str(f) for f in z3["FIELD"]]

    i2u = {(S2[i], F2[i]) for i in range(len(S2)) if K2[i] == 2 and S2[i] in lab}
    i3u = {(S2[i], F2[i]) for i in range(len(S2)) if K2[i] == 3 and S2[i] in lab}
    i3c = {(S3[i], F3[i]) for i in range(len(S3)) if S3[i] in lab}

    universe = sorted({k[0] for k in (i2u & i3u)})
    TR, TE = _split_universe(universe)
    H2 = torch.tensor(z2["H"].astype(np.float32), device=DEV); LD2 = torch.tensor(z2["LD"], device=DEV)
    H3 = torch.tensor(z3["H"].astype(np.float32), device=DEV); state_logits = torch.tensor(z3["LD"], device=DEV)

    def build(pairs2, pairs3, kind):
        ks = sorted(pairs2 & pairs3)
        sids = sorted({k[0] for k in ks}); si = {s: n for n, s in enumerate(sids)}


        idx2 = {(S2[i], F2[i]): i for i in range(len(S2)) if K2[i] == 2}
        if kind == "birdC":
            idx3 = {(S3[i], F3[i]): i for i in range(len(S3))}
        else:
            idx3 = {(S2[i], F2[i]): i for i in range(len(S2)) if K2[i] == 3}
        r2 = [idx2[k] for k in ks]; r3 = [idx3[k] for k in ks]
        if kind == "birdC":
            H3v, L3v = H3[r3], state_logits[r3].unsqueeze(1)
        else:
            H3v, L3v = H2[r3], LD2[r3][:, :3]
        y = torch.tensor([float(lab[s]["label"]) for s in sids], device=DEV)
        tr = torch.tensor(sorted(si[s] for s in sids if s in TR), device=DEV)
        te = torch.tensor(sorted(si[s] for s in sids if s in TE), device=DEV)
        return dict(H2=H2[r2], L2=LD2[r2][:, :2], H3=H3v, L3=L3v,
                    SEG=torch.tensor([si[k[0]] for k in ks], device=DEV),
                    N=len(sids), y=y, tr=tr, te=te, kind=kind, sids=sids)

    B_C = build(i2u, i3c, "birdC")
    B_U = build(i2u, i3u, "bird")
    return B_C, B_U, TR, TE


def load_org(cache):
    z = np.load(f"{CACHE_DIR}/{cache}", allow_pickle=True)
    SID = [str(s) for s in z["SID"]]; FL = [str(f) for f in z["FIELD"]]; K = z["KIND"].astype(int)
    lab = json.load(open(f"{DATA_DIR}/OrgAccess/evaluation_private/labels.json"))
    keep = {s for s, v in lab.items() if str(v.get("raw", "")).lower() in Y3
            and normalize("OrgAccess", v["raw"]) is not None}
    sids = sorted(keep & set(SID)); si = {s: n for n, s in enumerate(sids)}
    i2 = {(SID[i], FL[i]): i for i in range(len(SID)) if K[i] == 2 and SID[i] in keep}
    i3 = {(SID[i], FL[i]): i for i in range(len(SID)) if K[i] == 3 and SID[i] in keep}
    ks = sorted(set(i2) & set(i3))
    H = torch.tensor(z["H"].astype(np.float32), device=DEV); LD = torch.tensor(z["LD"], device=DEV)
    N = len(sids)
    y = torch.tensor([Y3[str(lab[s]["raw"]).lower()] for s in sids], device=DEV)
    o = sorted(range(N)); random.Random(0).shuffle(o)
    tr = torch.tensor(sorted(o[:int(N * .6)]), device=DEV)
    te = torch.tensor(sorted(o[int(N * .8):]), device=DEV)
    return dict(H2=H[[i2[k] for k in ks]], L2=LD[[i2[k] for k in ks]][:, :2],
                H3=H[[i3[k] for k in ks]], L3=LD[[i3[k] for k in ks]][:, :3],
                SEG=torch.tensor([si[k[0]] for k in ks], device=DEV), N=N, y=y,
                tr=tr, te=te, kind="org", sids=sids)


def pol(D, h, ref=False):
    c2 = 0.0 if ref else h.corr(D["H2"])[:, :2]
    p_dem = F.softmax(D["L2"] + c2, -1)[:, 1]
    if D["kind"] == "birdC":
        c3 = 0.0 if ref else h.corr(D["H3"])[:, SLOT[0]:SLOT[1]]
        p_viol = torch.sigmoid(D["L3"] + c3).squeeze(1)


        state_probs = torch.stack([(1 - p_viol) * p_dem, torch.zeros_like(p_viol), p_viol], 1)
        return p_dem, state_probs, p_viol
    c3 = 0.0 if ref else h.corr(D["H3"])[:, :3]
    return p_dem, F.softmax(D["L3"] + c3, -1), None


def agg(D, p_dem, state_probs, seg, n):
    if D["kind"] == "birdC":
        vio = state_probs[:, 2]
        pj = 1 - torch.exp(torch.zeros(n, device=seg.device).index_add_(0, seg, torch.log(1 - vio + EPS)))
        return torch.stack([pj, 1 - pj, torch.zeros_like(pj)], 1)
    met, vio, un = p_dem * state_probs[:, 0], p_dem * state_probs[:, 2], 1 - p_dem
    lnb = torch.zeros(n, device=seg.device).index_add_(0, seg, torch.log(1 - vio + EPS))
    lin = torch.zeros(n, device=seg.device).index_add_(0, seg, torch.log(met + un + EPS))
    pj = 1 - torch.exp(lnb); pf = torch.exp(lin)
    return torch.stack([pj, pf, (1 - pj - pf).clamp(0, 1)], 1)


def reward(D, P, y):
    if D["kind"] == "birdC":
        return (P[:, 0] >= .5).float() == y.float()
    if D["kind"] in ("bird", "eicu"):
        return ((1 - P[:, 1]).clamp(0, 1) >= .5).float() == y.float()
    return (P.argmax(1) == y).float()


def _logp(D, p_dem, state_probs, p_viol, DEM, ST):
    lp_dem = torch.where(DEM.bool(), torch.log(p_dem.clamp(min=EPS)).unsqueeze(0),
                         torch.log((1 - p_dem).clamp(min=EPS)).unsqueeze(0))
    if D["kind"] == "birdC":
        lp3 = torch.where(ST.bool(), torch.log(p_viol.clamp(min=EPS)).unsqueeze(0),
                          torch.log((1 - p_viol).clamp(min=EPS)).unsqueeze(0))
        return lp_dem + lp3
    G, n = ST.shape
    return lp_dem + torch.log(state_probs.clamp(min=EPS))[None].expand(G, n, 3).gather(
        2, ST.unsqueeze(-1)).squeeze(-1)


def grpo(D, h, opt, G=24, inner=1, eps_clip=0.2, beta=0.05):
    trm = torch.isin(D["SEG"], D["tr"])
    seg = D["SEG"][trm]; _, seg = torch.unique(seg, return_inverse=True)
    n = int(seg.max()) + 1
    y = D["y"][D["tr"]]
    with torch.no_grad():
        p, state_probs, pv = pol(D, h); p, state_probs = p[trm], state_probs[trm]; pv = pv[trm] if pv is not None else None
        DEM = torch.stack([torch.bernoulli(p) for _ in range(G)])
        if D["kind"] == "birdC":
            ST = torch.stack([torch.bernoulli(pv) for _ in range(G)]).long()
            lp_old = _logp(D, p, state_probs, pv, DEM, ST)
            R = []
            for g in range(G):
                vio = ST[g].float()
                pj = 1 - torch.exp(torch.zeros(n, device=seg.device).index_add_(
                    0, seg, torch.log(1 - vio + EPS)))
                R.append((pj >= .5).float() == y.float())
        else:
            ST = torch.stack([torch.multinomial(state_probs, 1).squeeze(1) for _ in range(G)])
            lp_old = _logp(D, p, state_probs, None, DEM, ST)
            R = []
            for g in range(G):
                one = F.one_hot(ST[g], 3).float()
                R.append(reward(D, agg(D, DEM[g], one, seg, n), y))
        R = torch.stack(R).float()
        A = (R - R.mean(0, keepdim=True)) / (R.std(0, keepdim=True) + 1e-6)
    for _e in range(inner):
        p, state_probs, pv = pol(D, h); p, state_probs = p[trm], state_probs[trm]; pv = pv[trm] if pv is not None else None
        ratio = torch.exp(_logp(D, p, state_probs, pv, DEM, ST) - lp_old)
        Ar = A[:, seg]
        s = torch.min(ratio * Ar, ratio.clamp(1 - eps_clip, 1 + eps_clip) * Ar)
        pr, ref_state_probs, pvr = pol(D, h, ref=True); pr, ref_state_probs = pr[trm], ref_state_probs[trm]
        pvr = pvr[trm] if pvr is not None else None
        if D["kind"] == "birdC":
            kl = (F.kl_div(torch.log(torch.stack([1 - p, p], 1).clamp(min=EPS)),
                           torch.stack([1 - pr, pr], 1), reduction="none").sum(1).mean()
                  + F.kl_div(torch.log(torch.stack([1 - pv, pv], 1).clamp(min=EPS)),
                             torch.stack([1 - pvr, pvr], 1), reduction="none").sum(1).mean())
        else:
            kl = (F.kl_div(torch.log(torch.stack([1 - p, p], 1).clamp(min=EPS)),
                           torch.stack([1 - pr, pr], 1), reduction="none").sum(1).mean()
                  + F.kl_div(torch.log(state_probs.clamp(min=EPS)), ref_state_probs, reduction="none").sum(1).mean())
        loss = -s.mean() + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()


def main():
    global CACHE_DIR, DATA_DIR, DEV
    parser = argparse.ArgumentParser(description="Train the PKGuard residual head.")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--bird2", default="uf_bird_10000.npz")
    parser.add_argument("--bird3", default="ufb_10000.npz")
    parser.add_argument("--org", default="uf_org_55800.npz")
    parser.add_argument("--device", default=DEV)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    CACHE_DIR, DATA_DIR, DEV = args.cache_dir, args.data_dir, args.device
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    bird_c, bird_u, _, _ = load_bird_shared(args.bird2, args.bird3)
    arms = (bird_c, load_org(args.org), bird_u)

    for seed in map(int, args.seeds.split(",")):
        torch.manual_seed(seed)
        head = Head().to(DEV)
        opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
        for _ in range(args.iters):
            for inputs in arms:
                grpo(inputs, head, opt, G=args.samples, beta=args.beta)
        path = args.checkpoint_dir / f"head_third_on_seed{seed}.pt"
        torch.save({"W1_w": head.W1.weight.detach().cpu(), "W1_b": head.W1.bias.detach().cpu(),
                    "W2_w": head.W2.weight.detach().cpu(), "W2_b": head.W2.bias.detach().cpu()}, path)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
