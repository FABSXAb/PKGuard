import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import Head, agg, pol


def load_head(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    head = Head().to(device)
    head.load_state_dict({f"{layer}.{param}": checkpoint[f"{layer}_{short}"].float()
                          for layer in ("W1", "W2")
                          for param, short in (("weight", "w"), ("bias", "b"))})
    return head.eval()


def load_inputs(cache, prohibited_cache, view, device, request_ids=None):
    def index(data, kind=None):
        return {(str(sid), str(field)): i
                for i, (sid, field) in enumerate(zip(data["SID"], data["FIELD"]))
                if kind is None or int(data["KIND"][i]) == kind}

    with np.load(cache, allow_pickle=True) as primary:
        prohibited = view == "birdC"
        with np.load(prohibited_cache, allow_pickle=True) if prohibited else np.load(cache, allow_pickle=True) as secondary:
            relevance = index(primary, 2)
            state = index(secondary, None if prohibited else 3)
            keys = relevance.keys() & state.keys()
            if prohibited:
                local_keys = relevance.keys() & index(primary, 3).keys()
                pool = {sid for sid, _ in local_keys}
                keys = {(sid, field) for sid, field in keys if sid in pool}
            if request_ids is not None:
                keys = {(sid, field) for sid, field in keys if sid in request_ids}
            keys = sorted(keys)
            sids = sorted({sid for sid, _ in keys})
            if not sids:
                raise ValueError("No paired cache rows for the selected requests")
            positions = {sid: i for i, sid in enumerate(sids)}
            i2 = [relevance[key] for key in keys]
            i3 = [state[key] for key in keys]
            as_float = lambda array: torch.tensor(array.astype(np.float32), device=device)
            state_logits = secondary["LD"][i3]
            if prohibited:
                state_logits = state_logits.reshape(-1, 1)
            else:
                state_logits = state_logits[:, :3]
            return dict(H2=as_float(primary["H"][i2]), L2=as_float(primary["LD"][i2, :2]),
                        H3=as_float(secondary["H"][i3]), L3=as_float(state_logits),
                        SEG=torch.tensor([positions[sid] for sid, _ in keys], device=device),
                        N=len(sids), kind=view, sids=sids)


@torch.inference_mode()
def predict(inputs, heads):
    scores = []
    for head in heads or [None]:
        relevance, local, _ = pol(inputs, head, ref=head is None)
        scores.append(agg(inputs, relevance, local, inputs["SEG"], inputs["N"]))
    votes = torch.stack([s.argmax(1) for s in scores])
    selected = votes.mode(dim=0).values if len(scores) > 1 else votes[0]
    if inputs["kind"] == "org":
        labels = ("rejected", "full", "partial")
    else:
        selected = (selected != 1).long()
        labels = ("PERMIT", "DENY")
    return [labels[i] for i in selected.tolist()]


def main():
    parser = argparse.ArgumentParser(description="Predict authorization labels from cached decision model inputs.")
    parser.add_argument("--view", choices=("org", "bird", "birdC", "eicu"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prohibited-cache", type=Path)
    parser.add_argument("--head", type=Path, action="append", default=[])
    parser.add_argument("--request-ids", type=Path, help="Optional text file of request IDs, one per line")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.view == "birdC" and not args.prohibited_cache:
        parser.error("--prohibited-cache is required for birdC")
    if len(args.head) > 1 and (args.view != "org" or len(args.head) != 3):
        parser.error("Pass one head, or three heads for OrgAccess voting")
    ids = set(args.request_ids.read_text().split()) if args.request_ids else None
    inputs = load_inputs(args.cache, args.prohibited_cache, args.view, args.device, ids)
    labels = predict(inputs, [load_head(p, args.device) for p in args.head])
    for sid, label in zip(inputs["sids"], labels):
        print(json.dumps({"sample_id": sid, "decision": label}))


if __name__ == "__main__":
    main()
