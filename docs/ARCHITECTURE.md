# Architecture

fraudnet is one Python package (`src/fraudnet`) with five layers: a simulator that produces the event log, a
feature layer that turns the log into point-in-time features, models, evaluation, and serving. Every layer
below `serving` is a library; `bench.py`, `tune.py` and `cli.py` are the entry points.

## Data flow

```
SimConfig + seed
   |
   v
sim.simulate()  ->  Dataset (aligned numpy columns, sorted by time; reference tables; SHA-256)
   |                    cached under data/world_seed<seed>_<config hash>.npz
   v
features.replay.replay()
   |   for each transaction in time order:
   |     store.advance_to(ts)    apply fraud reports with report_ts < ts, build snapshots due by ts
   |     store.features(...)     read 53 features from state strictly before the transaction
   |     store.ingest(...)       update rolling state and the edge log
   |   optional checkpoint: the store pickled at the build day (the serving state)
   v
FeatureTable (X, snapshot id and GNN node indices per transaction, snapshots)
   |
   +--> evaluation.split.TimeSplit   train / validation / maturity gap / test, labels as of build day
   +--> models: rules, GBDTModel, GNNModel  ->  pipeline.ModelBundle (thresholds from validation)
   +--> bench.py: metrics, ablation, leakage proof, latency  ->  benchmarks/results.json
   |
   v
serving.ScoringService(bundle, store at build day)  ->  FastAPI app (serving.app)
```

## Modules

### `sim/`: the world

| Module | Responsibility |
|---|---|
| `config.py` | `SimConfig` (every knob, frozen), category profiles, pattern and channel codes, time constants |
| `world.py` | Static world: customers (signup, rate, spend, favourite merchants, hour profile), devices (personal, second, replacement after a phone change, household and workplace devices shared by several accounts), social contacts with locality, merchants with Zipf popularity, mule recruitment |
| `legit.py` | Bulk genuine purchases and transfers, vectorised with numpy from per-customer parameters |
| `confounders.py` | Genuine behaviours that look like fraud: trips, shopping sprees, in-app bursts, forwarding, landlords, large one-off transfers, rent collection, group payments |
| `patterns.py` | `FraudInjector`: rings, mule chains, account takeover (two styles) and card testing (burst or device farm), each with its reporting delay |
| `rows.py` | Column layout and a row buffer for event generators |
| `simulate.py` | Orchestration with one generator in a fixed order; loss of some reports; time ordering; cache |
| `dataset.py` | `Dataset` with labels-as-of, summary, save and load with checksum verification |

### `features/`: point-in-time features

| Module | Responsibility |
|---|---|
| `schema.py` | The 53 feature names in 13 groups, which groups are graph and which are batch, reason-code text |
| `store.py` | `OnlineFeatureStore`: per-account, per-device and per-merchant rolling state (deques with window pruning), scheduled fraud reports in a heap, the edge log, snapshot refresh at day boundaries, pickling |
| `graph.py` | `build_snapshot`: typed edges of the trailing 30 days, PageRank (merchant edges down-weighted), personalised PageRank seeded with reported accounts and devices over strong links, deterministic block label propagation for communities, community statistics, GNN node features |
| `replay.py` | Offline replay producing the `FeatureTable`, label schedules (`reported`, or the leaky `instant`), build-day checkpoint |
| `leakage.py` | The point-in-time checker: `known_at` (the log as known at a cut), `scrambled_future`, `check_point_in_time` |

The store's contract: `features()` is a pure read of state built from earlier events, `ingest()` mutates
afterwards, a report is applied only once the clock is strictly past its report time, and graph features come
from the last snapshot, which only contains events before its boundary. Because training features are
produced by `replay` through the same object, there is no second implementation that could disagree with the
API.

### `models/`

| Module | Responsibility |
|---|---|
| `rules.py` | Eleven vectorised rules with weights and reason codes; score is the weighted sum plus an amount tie-breaker |
| `gbdt.py` | `GBDTModel`: scikit-learn `HistGradientBoostingClassifier` on a column subset, negative down-sampling with compensating weights, tree count chosen by validation average precision via staged predictions, deterministic refit for serving |
| `gnn.py` | `snapshot_tensors` (row-normalised CSR adjacency per relation plus three unknown-entity nodes), `HeteroSAGELayer`, `FraudSAGE` (node encoder plus transaction head), `TabularEncoder`, `GNNModel` (per-day full-batch training, validation early stopping, online `embed_snapshot` and `score_rows`) |
| `explain.py` | `OcclusionExplainer`: replace one feature group at a time with the genuine-training median and rank the score drops; texts from the transaction's own values |

### `evaluation/`

| Module | Responsibility |
|---|---|
| `split.py` | `TimeSplit`: burn-in, train, validation, label-maturity gap, test; labels as of the build day |
| `metrics.py` | PR-AUC, ROC-AUC, recall at a false positive rate (strict threshold), precision at top-k per day, the cost model and its optimal threshold, fixed-budget alerts, day-block and paired bootstrap intervals |
| `report.py` | Markdown rendering, README update between markers, threshold gate |

### `serving/`

| Module | Responsibility |
|---|---|
| `service.py` | `ScoringService`: under one lock, advance the clock, read features, ingest; then score, explain, decide and alert outside the lock. Recomputes GNN embeddings on every snapshot. Reports from analysts and chargebacks become scheduled label events. Simulator replay feed, ring and network views |
| `alerts.py` | SQLite alert queue (`:memory:` by default, a file path for durability) |
| `app.py` | FastAPI routes, error handlers, static console |
| `schemas.py` | Pydantic request and response models with cross-field validation |
| `middleware.py` | Request ids, JSON access logs, request metrics, constant-time API key check |
| `metrics.py` | Prometheus collectors, including a scrape-time collector over the service counters |
| `loader.py` | Builds the service from `artifacts/` |
| `static/` | The console: `index.html`, `app.js` (vanilla JS, force layout in SVG), `style.css` |

## Online path of one request

1. Pydantic validates the body (422 on failure, store untouched).
2. `ScoringService._features` takes the lock, rejects a timestamp older than the clock (409), advances the
   clock (applying due reports and, at a day boundary, rebuilding the snapshot and the GNN embeddings), reads
   the 53 features and the transaction's snapshot node indices, ingests the transaction and releases the lock.
3. The model scores the row and its 13 occlusion variants in one call (about 1.3 ms for boosting, 0.07 ms
   for the GNN head, see the latency table).
4. The decision compares the score with the validation thresholds; review and block create an alert.
5. Middleware adds `X-Request-ID` and `X-Process-Time-ms`, counts the request and logs one JSON line.

## Design decisions

* **Replay instead of a batch feature job.** One implementation, so no training and serving skew; the price
  is a Python loop (about 17 s for 441 thousand transactions), which is fine at this scale.
* **Daily snapshots for graph features.** Matches how graph jobs run in production and keeps PageRank and
  communities cheap (about a quarter of a second per refresh); intra-day signals come from the streaming
  device and counterparty features instead.
* **Label propagation over strong links only.** Merchants are excluded from the community graph because one
  popular merchant would join every customer into one community; label propagation with block updates and
  smallest-label tie breaking is deterministic and does not oscillate on the bipartite account-device graph.
* **Heterogeneous GraphSAGE in plain PyTorch.** Six relation types with their own weights and mean
  aggregation over CSR matrices; full-batch per daily snapshot because the graph (about 12 thousand nodes)
  fits easily; unknown-entity nodes so a device first seen today still gets an embedding.
* **A label-maturity gap.** The most recent week before the build day is excluded from fitting because most
  of its fraud is not yet reported.
* **Two threads for native pools, one for GNN training.** Measured reasons are in REPORT.md.
