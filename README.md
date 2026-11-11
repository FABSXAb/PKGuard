# PKGuard

PKGuard builds a permission knowledge graph (PKG) for a role, uses a decision model (DM) to assess the request against permission items, and combines the assessments into an authorization decision. The residual head can be trained separately on cached DM features using request-level labels.

| Path | Purpose |
| --- | --- |
| `pipeline/kg.py`, `pipeline/graphs.py` | PKG representation, policy adapters, and rule encoding |
| `pipeline/run_kg.py`, `pipeline/evaluator.py` | Base PKGuard inference and rule composition |
| `posttraining/train.py` | Request-level rewards and residual-head training |
| `posttraining/predict_cached.py` | Checkpoint loading and authorization predictions from cached inputs |
| `posttraining/uf_extract*.py` | Cache extraction from the DM |

## Base authorization

PKGuard uses a locally deployed 27B decision model. See the [model discussion on Hugging Face](https://huggingface.co/spaces/multimodalart/jev-decision-index/discussions/68) for the model reference.

Run the commands from the repository root with Python 3.10 or later. The inference client connects to the local `/v1/systemone` endpoint. Add your model runtime directory to `PYTHONPATH` and pass the local 27B model path to `dm_server.py`. The runtime module and class default to `dm_runtime` and `DM`. Set `PKGUARD_DM_RUNTIME_MODULE` and `PKGUARD_DM_RUNTIME_CLASS` to match your installation if it uses different names. The server and client use `dm` as the default model identifier:

```bash
PYTHONPATH=/path/to/model-runtime python dm_server.py --model /path/to/dm-27b --port 8000
python -m pipeline.run_kg --dataset OrgAccess --input examples/orgaccess.jsonl --output predictions.jsonl --endpoint http://127.0.0.1:8000
```

If the server requires authentication, set `DECISION_API_KEY`. Use `--model` on the inference CLI if the server exposes a model identifier other than `dm`. The runner accepts normalized JSONL for OrgAccess, BIRD-AC, and eICU-AC. For BIRD-AC or eICU-AC, use the ACL input format in `examples/acl.jsonl`: `resource_vocab` is an array and `policy_map` is an object. The parser also supports `schema_text` and `policy_text` as fallbacks.

The runner retains the R configuration. It uses probability composition for OrgAccess and the hard prohibited-demand check for BIRD-AC and eICU-AC. The base runner prefixes the original dataset labels with `decision:`: OrgAccess `full`, `partial`, and `rejected` correspond to **Granted**, **Partial Granted**, and **Deny**. ACL labels `PERMIT` and `DENY` correspond to **Granted** and **Deny**. The base runner also returns the local assessments and, when applicable, their composed scores.

## Post-training and cached prediction

Install the training dependencies with `python -m pip install -r requirements-posttraining.txt`. The default training device is CUDA. Cached prediction supports CPU.

```bash
python -m posttraining.train --data-dir data --cache-dir calib --checkpoint-dir checkpoints
python -m posttraining.predict_cached --view eicu --cache calib/uf_eicu_316.npz --head checkpoints/head_third_on_seed2.pt
```

Training reads `data/BIRD-AC/evaluation_private/labels.json` and `data/OrgAccess/evaluation_private/labels.json`. The BIRD records contain numeric `label` values (1 for deny, 0 for permit); OrgAccess records contain the original label in `raw`. The trainer expects `uf_bird_10000.npz`, `ufb_10000.npz`, and `uf_org_55800.npz` in the cache directory. It uses the shared BIRD request ID split, keeps the DM frozen, and fits the residual head using BIRD-C, OrgAccess, and BIRD-u. Defaults are seeds 0, 1, and 2, 2,000 outer rounds, 24 samples per request, and KL weight 0.02. Each outer round updates the head once per view. The trainer saves a separate checkpoint for each seed.

For BIRD-C cached prediction, use `--view birdC` and pass the second cache with `--prohibited-cache`. The prohibited-demand score enters composition directly. For the other views, the relevance answer and local state distribution determine the scores. Binary predictions from the local state view select the three-way outcome first and then map it to a binary label. For OrgAccess, pass three `--head` arguments to combine their predicted labels by vote. A three-way tie selects `rejected`. To score only a selected subset, pass its request IDs with `--request-ids`. The prediction command returns one authorization label per request.

## Generating caches

Cache extraction uses the encoder hidden state and readout of the same 27B decision model. Set `PKGUARD_DM_MODEL` to the local model path. Set `PKGUARD_DATA_DIR`, `PKGUARD_ORG_DIR`, and `PKGUARD_CACHE_DIR` to override the default input and output directories. The OrgAccess directory contains `easy.jsonl`, `medium.jsonl`, and `hard.jsonl`. The BIRD-AC and eICU-AC directories under the data directory each contain `requests.jsonl`.

```bash
python -m posttraining.uf_extract bird 10000
python -m posttraining.uf_extract_bird_denies 10000
python -m posttraining.uf_extract org 55800
python -m posttraining.uf_extract eicu 316
```

The local state caches contain `SID`, `FIELD`, `KIND`, `H`, and `LD`. Records are paired by request and permission item. `KIND=2` denotes relevance and `KIND=3` denotes the local state question. The BIRD-C cache uses `SID`, `FIELD`, `H`, and the scalar true-minus-false logit in `LD`.
