# fraudnet

[![ci](https://github.com/raunitgrey7/fraudnet/actions/workflows/ci.yml/badge.svg)](https://github.com/raunitgrey7/fraudnet/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Graph machine learning for payment fraud, built around one rule: a feature may only use what the bank knew
when the payment happened. fraudnet contains a seeded agent-based simulator of a payments network with four
planted fraud patterns and delayed labels, a point-in-time online feature store that both training and serving
read through, three detectors (a rules baseline, gradient boosting on tabular plus graph features, and a
GraphSAGE network written in plain PyTorch), a benchmark that measures what fraud teams actually report, and a
FastAPI scoring service with reason codes, an alert queue and a ring view.

Everything runs on a laptop CPU with no API key and no network at runtime. Every number below was produced by
`make bench` (seed 42) and copied from `benchmarks/results.json` by `python -m fraudnet.bench --update-readme`;
`tests/test_docs.py` fails if the README and the results file disagree.

## Why

Fraud is a graph problem that is usually evaluated like a tabular one. Synthetic-identity rings look like
ordinary new customers one account at a time but share a handful of devices; money mules look like people
paying friends until you follow the money two hops; account takeovers reuse the same attacker device across
victims. A model that sees one transaction at a time misses that structure.

Fraud numbers are also easy to inflate. Labels arrive days to weeks after the payment (chargebacks, customer
reports, ring discovery when the bust-out goes unpaid) and some never arrive. A pipeline that joins the final
label table onto historical transactions lets every feature see the future, and offline results then say
nothing about production. fraudnet makes the time ordering explicit and tests it:

1. **One feature store for training and serving.** The training table is produced by replaying the log through
   the same `OnlineFeatureStore` the API calls, and the benchmark checks that online features for held-out
   transactions are bit-identical to the offline ones.
2. **A point-in-time proof that can fail.** `features/leakage.py` replays the log as it was known at a cut time
   and with a scrambled future, and requires every feature before the cut to be unchanged. The honest pipeline
   passes on the real dataset; a pipeline that uses labels before they were reported is caught, and the
   benchmark shows what that leak does to offline and online numbers.
3. **Metrics a fraud team reads.** PR-AUC with day-block bootstrap intervals, recall at 0.5 and 1 percent false
   positive rate, precision of the top alerts per day, money saved under a stated cost model, recall per
   planted pattern and per attack, a graph feature ablation with paired intervals, and scoring latency.

## Architecture

```
  sim/  seeded world: 5,000 customers, devices, 400 merchants, 90 days        reference data: signup
        purchases, transfers, look-alikes (trips, rent, shared devices)        dates, merchant codes
        planted rings, mule chains, account takeover, card testing                     |
        fraud reports with pattern-specific delays (some never arrive)                 |
              |                                                                        |
              v                                                                        v
  features/  OnlineFeatureStore  <---------- one code path for training and serving ---------->
        streaming: amount, velocity, new device, pass-through, shared-device counts,
                   counterparty activity, reported fraud on linked entities (after report time)
        daily snapshot (events before midnight): degree, PageRank, fraud-seeded personalised
                   PageRank, label-propagation communities, GNN node features
              |                                   |                                   |
     offline replay                         leakage checker                    ScoringService
     feature table + snapshots              prefix replay and scrambled        lock: features, then ingest
              |                             future must match bit for bit      reason codes by occlusion
              v                                                                decision from val thresholds
  models/  rules | HistGradientBoosting | heterogeneous GraphSAGE (torch.sparse)          |
              |                                                                        v
  bench.py  time split with a label-maturity gap, metrics, ablation,          FastAPI: /v1/score, /v1/alerts,
            leakage cost, latency -> benchmarks/results.json, README          /v1/rings, /v1/replay, /metrics
                                                                              console at / (vanilla JS)
```

Module by module: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Methodology and metric definitions:
[docs/EVALUATION.md](docs/EVALUATION.md). Every measured number with its command: [REPORT.md](REPORT.md).

## Quickstart

```
make setup     # uv sync, then simulate the world (cached under data/)
make test      # pytest
make bench     # the full benchmark, about 8 minutes on a laptop CPU
make serve     # trains the bundle if artifacts/ is empty, then serves http://127.0.0.1:8772
curl -s -X POST localhost:8772/v1/score -H 'content-type: application/json' \
  -d '{"account_id": 12, "device_id": 12, "merchant_id": 3, "amount": 49.9, "channel": "ecom"}'
```

Open `http://127.0.0.1:8772/` for the console: replay the held-out test stream through the scoring path, work
the alert queue (a fraud verdict is fed back to the feature store), and inspect device-sharing clusters and
account networks. `/docs` has the OpenAPI UI. Set `APP_API_KEY` to require an `X-API-Key` header on every route
except `/health`, `/metrics` and the console page.

## Measured results

<!-- RESULTS:BEGIN -->
Seed 42, full profile, generated 2026-11-27T06:49:31Z, commit `2e2700b`, Python 3.13.17, torch 2.14.1, scikit-learn 1.9.1 on arm64 with 2 threads. Total benchmark time 339.4 s.

441,378 transactions, 5,076 accounts, 8,039 devices, 406 merchants; 2,982 fraud (0.676 percent); 14.0 percent of fraud is never reported; median label delay 9.99 days.

| window | days | transactions | fraud | fraud rate | fraud not yet labelled at build |
|---|---|---|---|---|---|
| train | 7 to 49 | 200,586 | 1347 | 0.671% | 35.3% |
| val | 49 to 63 | 69,456 | 462 | 0.665% | 54.8% |
| test | 70 to 90 | 104,125 | 786 | 0.755% | n/a |

### Detection on the test window

| model | PR-AUC [95% CI] | ROC-AUC | recall at 0.5% FPR | recall at 1% FPR [95% CI] | precision top-10/day | precision top-50/day |
|---|---|---|---|---|---|---|
| rules | 0.4829 [0.381, 0.571] | 0.8311 | 0.4847 | 0.5038 [0.399, 0.588] | 0.815 | 0.370 |
| gbdt_tabular | 0.7175 [0.643, 0.775] | 0.9929 | 0.7850 | 0.8677 [0.809, 0.911] | 0.805 | 0.521 |
| gbdt_streaming | 0.8173 [0.757, 0.886] | 0.9939 | 0.8410 | 0.8893 [0.810, 0.962] | 0.945 | 0.572 |
| gbdt | 0.8772 [0.843, 0.911] | 0.9985 | 0.9249 | 0.9733 [0.952, 0.986] | 0.960 | 0.590 |
| graphsage | 0.7204 [0.635, 0.801] | 0.9865 | 0.7036 | 0.7506 [0.639, 0.860] | 0.950 | 0.513 |
| gbdt+graphsage | 0.8044 [0.749, 0.860] | 0.9966 | 0.7735 | 0.8740 [0.819, 0.934] | 0.960 | 0.548 |

### Money saved under the cost model

| model | alerts/day | alert precision | fraud recall | money saved | share of loss saved | precision at 50/day | recall at 50/day | money saved at 50/day |
|---|---|---|---|---|---|---|---|---|
| rules | 33.9 | 0.551 | 0.475 | 163,350 | 0.613 | 0.370 | 0.471 | 156,709 |
| gbdt_tabular | 38.5 | 0.681 | 0.668 | 216,174 | 0.811 | 0.521 | 0.663 | 209,311 |
| gbdt_streaming | 25.4 | 0.911 | 0.588 | 195,182 | 0.732 | 0.572 | 0.728 | 213,694 |
| gbdt | 27.0 | 0.913 | 0.627 | 199,148 | 0.747 | 0.590 | 0.751 | 214,609 |
| graphsage | 20.6 | 0.934 | 0.488 | 180,088 | 0.675 | 0.513 | 0.653 | 194,614 |
| gbdt+graphsage | 22.9 | 0.948 | 0.551 | 182,069 | 0.683 | 0.548 | 0.697 | 203,151 |

Cost model: 5 per alert reviewed, 10 more when the alert is a genuine payment, a missed fraud costs its amount plus a 15 chargeback fee. The first block uses the threshold that maximised money saved on validation; the second gives analysts exactly 50 alerts per day. Fraud loss with no model: 266,587; a perfect detector would save 262,657.

### Per-pattern recall at 1 percent FPR

| model | fraud_ring | money_mule | account_takeover | card_testing |
|---|---|---|---|---|
| rules | 0.372 / 0.80 | 0.729 / 1.00 | 0.989 / 1.00 | 0.271 / 1.00 |
| gbdt_tabular | 0.794 / 1.00 | 0.971 / 1.00 | 0.859 / 1.00 | 0.950 / 1.00 |
| gbdt_streaming | 0.784 / 1.00 | 0.982 / 1.00 | 1.000 / 1.00 | 0.993 / 1.00 |
| gbdt | 0.948 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 |
| graphsage | 0.505 / 0.80 | 0.977 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 |
| gbdt+graphsage | 0.750 / 1.00 | 0.988 / 1.00 | 1.000 / 1.00 | 0.993 / 1.00 |

Cells: transaction recall / share of attacks with at least one transaction caught. Test window: fraud_ring 384 transactions in 5 attacks, money_mule 170 transactions in 36 attacks, account_takeover 92 transactions in 26 attacks, card_testing 140 transactions in 6 attacks.

### Graph feature ablation (gradient boosting)

| variant | PR-AUC | change vs tabular [paired 95% CI] | recall at 1% FPR | fraud_ring | money_mule | account_takeover | card_testing |
|---|---|---|---|---|---|---|---|
| tabular | 0.7175 | +0.0000 [+0.000, +0.000] | 0.8677 | 0.794 | 0.971 | 0.859 | 0.950 |
| tabular+shared_device | 0.8146 | +0.0971 [+0.061, +0.138] | 0.9262 | 0.878 | 0.965 | 0.946 | 1.000 |
| tabular+counterparty | 0.7377 | +0.0202 [+0.009, +0.033] | 0.8779 | 0.802 | 0.982 | 0.870 | 0.964 |
| tabular+risk | 0.8239 | +0.1064 [+0.074, +0.148] | 0.9288 | 0.867 | 0.988 | 0.989 | 0.986 |
| tabular+degree | 0.8269 | +0.1094 [+0.084, +0.140] | 0.9288 | 0.885 | 0.977 | 0.924 | 0.993 |
| tabular+pagerank | 0.8334 | +0.1159 [+0.079, +0.170] | 0.9402 | 0.901 | 0.988 | 0.924 | 1.000 |
| tabular+community | 0.8069 | +0.0894 [+0.053, +0.131] | 0.9351 | 0.922 | 0.977 | 0.859 | 0.971 |
| all-shared_device | 0.8862 | +0.1687 [+0.126, +0.232] | 0.9746 | 0.953 | 0.988 | 1.000 | 1.000 |
| all-counterparty | 0.8794 | +0.1619 [+0.118, +0.221] | 0.9707 | 0.945 | 0.988 | 1.000 | 1.000 |
| all-risk | 0.7877 | +0.0702 [+0.039, +0.118] | 0.9173 | 0.865 | 0.977 | 0.902 | 1.000 |
| all-degree | 0.8702 | +0.1527 [+0.110, +0.209] | 0.9784 | 0.956 | 1.000 | 1.000 | 1.000 |
| all-pagerank | 0.8715 | +0.1540 [+0.115, +0.212] | 0.9656 | 0.935 | 0.988 | 1.000 | 1.000 |
| all-community | 0.8300 | +0.1125 [+0.060, +0.192] | 0.9211 | 0.841 | 0.994 | 1.000 | 1.000 |
| all | 0.8772 | +0.1597 [+0.117, +0.218] | 0.9733 | 0.948 | 0.994 | 1.000 | 1.000 |

Pattern columns are transaction recall at 1 percent FPR. Intervals resample test days and score both models on the same days.

### Label-maturity gap

| model | protocol | PR-AUC | recall at 1% FPR | ring recall | money saved |
|---|---|---|---|---|---|
| gbdt | with_gap | 0.8772 | 0.9733 | 0.948 | 199,148 |
| gbdt | no_gap | 0.7843 | 0.9631 | 0.938 | 161,076 |
| graphsage | with_gap | 0.7204 | 0.7506 | 0.505 | 180,088 |
| graphsage | no_gap | 0.7890 | 0.8435 | 0.685 | 193,207 |

Without the gap, validation runs from day 56 to the build day and 67.7 percent of its fraud is unlabelled at build time.

### Point-in-time correctness

First 30 days (139,394 transactions), cut at day 21.29.

| pipeline | check | rows | values compared | mismatched values | result |
|---|---|---|---|---|---|
| honest | prefix_replay | 97161 | 5,149,533 | 0 | identical |
| honest | scrambled_future | 97161 | 5,149,533 | 0 | identical |
| leaky_instant_labels | prefix_replay | 97161 | 5,149,533 | 55,160 | differs (device_known_fraud, acct_known_fraud, cp_known_fraud, acct_devices_known_fraud) |
| leaky_instant_labels | scrambled_future | 97161 | 5,149,533 | 115,207 | differs (device_known_fraud, acct_known_fraud, cp_known_fraud, acct_devices_known_fraud) |

### Cost of a label leak

| setting | test PR-AUC | recall at 1% FPR |
|---|---|---|
| leaky_model_on_leaky_features | 0.9226 | 0.9949 |
| leaky_model_on_online_features | 0.5472 | 0.5394 |
| honest_model_on_online_features | 0.8772 | 0.9733 |

### Scoring latency

| path | p50 ms | p95 ms | p99 ms | max ms | sequential txn/s |
|---|---|---|---|---|---|
| rules | 0.068 | 0.074 | 0.086 | 0.131 | 14673.0 |
| rules_no_reasons | 0.039 | 0.043 | 0.048 | 0.079 | 25654.0 |
| gbdt | 1.155 | 1.232 | 1.351 | 63.750 | 815.4 |
| gbdt_no_reasons | 1.113 | 1.264 | 1.423 | 13.041 | 878.4 |
| graphsage | 0.065 | 0.073 | 0.090 | 0.157 | 15072.2 |
| graphsage_no_reasons | 0.046 | 0.051 | 0.061 | 0.125 | 21362.7 |
| gbdt over HTTP, server side | 1.444 | 1.604 | 1.704 | 2.037 | 684.1 |
| gbdt over HTTP, client side | 1.949 | 2.102 | 2.212 | 2.514 | 509.4 |

2000 test transactions scored one at a time after 50 untimed warm-up transactions; each path covers feature lookup, scoring and reason codes unless marked no_reasons. Online features equal the offline replay features (max absolute difference 0.0). A daily graph refresh over 12,144 nodes takes 0.233 s plus 0.007 s of GNN embedding.

### Seeds

| seed | rules | gbdt_tabular | gbdt | graphsage |
|---|---|---|---|---|
| 42 | 0.4829 | 0.7175 | 0.8772 | 0.7204 |
| 7 | 0.5890 | 0.8178 | 0.9370 | 0.9586 |
| 2026 | 0.5237 | 0.6525 | 0.9071 | 0.8974 |
| mean | 0.5319 | 0.7293 | 0.9071 | 0.8588 |
| standard deviation | 0.0535 | 0.0833 | 0.0299 | 0.1237 |

Test PR-AUC per simulator seed; each seed is a different world with the same configuration.
<!-- RESULTS:END -->

## What the numbers say

* Graph features carry most of the lift. Adding them to the tabular features raises test PR-AUC from 0.7175 to
  0.8772, and every graph group helps on its own (each paired interval in the ablation excludes zero). In the
  leave-one-out rows, removing reported-fraud links (`risk`) or communities costs the most; shared-device,
  degree, PageRank and counterparty features overlap enough that dropping any one of them changes PR-AUC by
  less than 0.01.
* Rings are where structure matters most. At 1 percent FPR the tabular model catches 0.794 of ring
  transactions and the full model 0.948, with all five test rings detected.
* The plain PyTorch GraphSAGE is competitive on average but unstable. Across three simulator seeds its mean
  PR-AUC is 0.8588 against 0.9071 for boosting; it wins on seed 7 and loses clearly on seed 42, where rings
  make up almost half of the test fraud. Learned neighbourhood aggregation did not replace hand-made graph
  features here, and averaging the two models did not beat boosting alone on seed 42.
* A label leak is invisible offline and expensive online. Trained and evaluated with labels visible at
  transaction time, the boosted model reports a test PR-AUC of 0.9226; the same model on the features the API
  can actually compute scores 0.5472, below the tabular baseline.
* Label delay also distorts thresholds. With validation running up to the build day, 67.7 percent of its fraud
  is still unlabelled, and boosting picks a threshold that saves 161,076 on test; leaving a one-week maturity
  gap saves 199,148. The gap did not help the GNN on this seed (see the table), so it is a trade-off, not a free
  win.
* Money and PR-AUC disagree at the margin. Under the cost model the tabular model saves the most at its
  validation-chosen threshold, because it alerts more often and missed fraud is expensive; with a fixed budget
  of 50 alerts per day the full boosted model saves the most.

## Models

| Model | What it is | Notes |
|---|---|---|
| `rules` | Eleven weighted rules (velocity, new device with high amount, device fan-out, card-testing burst, pass-through funds, known-fraud device, ...) | Reason codes are the fired rules; weights set from domain reasoning before any model was trained |
| `gbdt_tabular` | Histogram gradient boosting on the 22 tabular features | The no-graph baseline of the ablation |
| `gbdt_streaming` | Boosting on tabular plus the real-time graph features | What a team gets without a batch graph job |
| `gbdt` | Boosting on all 53 features, tree count chosen on validation average precision | The default serving model |
| `graphsage` | Two-layer heterogeneous GraphSAGE (six relation types, mean aggregation) on the daily snapshot, plus streaming features in the transaction head | Plain PyTorch with `torch.sparse`; embeddings recomputed once per snapshot |
| `gbdt+graphsage` | Mean of the two probabilities | Evaluation only |

## Features

53 point-in-time features in 13 groups (`features/schema.py`); the ablation and the reason codes use the same
groups. Tabular: amount, time, account, velocity, context, device-account, flow. Graph, real time:
shared-device counts, counterparty activity, reported fraud on linked entities. Graph, daily snapshot: degree,
PageRank and fraud-seeded personalised PageRank, community statistics. Reports reach the counters only after
their report time; snapshots use events before the snapshot boundary.

## API

| Route | Purpose |
|---|---|
| `POST /v1/score` | Score one transaction: `score`, `decision` (approve, review, block), `reasons`, `alert_id`, `latency_ms`; `include_features` returns all 53 features by name |
| `POST /v1/score/batch` | Up to 500 transactions, scored in order |
| `GET /v1/alerts`, `POST /v1/alerts/{id}/resolve` | The analyst queue, highest score first; a fraud verdict becomes a report at the current clock |
| `POST /v1/labels` | A chargeback for a recently scored transaction |
| `POST /v1/replay` | Stream the next held-out simulator transactions through the scoring path (demo) |
| `GET /v1/rings`, `GET /v1/accounts/{id}/network` | Device-sharing communities and an account's live ego network |
| `GET /v1/models`, `GET /v1/stats` | Thresholds, provenance, store and queue counters |
| `GET /health`, `GET /metrics` | Liveness and Prometheus text format |

Requests are validated before they touch the store (exactly one of merchant or payee, transfers on p2p, finite
positive amounts, no unknown fields); a timestamp older than the store clock returns 409, because its features
could no longer be point-in-time. Every response carries `X-Request-ID` and `X-Process-Time-ms`; logs are one
JSON object per line.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `FRAUDNET_DATA_DIR` | `./data` | Simulated world cache |
| `FRAUDNET_ARTIFACT_DIR` | `./artifacts` | Trained bundle and the feature store as of build time |
| `FRAUDNET_DEFAULT_MODEL` | `gbdt` | `rules`, `gbdt` or `graphsage` |
| `FRAUDNET_ALERT_DB` | `:memory:` | SQLite path for a durable alert queue |
| `APP_API_KEY` | unset | When set, required as `X-API-Key` |
| `OMP_NUM_THREADS` | `2` when unset | Native thread pools; set before import |
| `LOG_LEVEL` | `INFO` | JSON log level |

## Limitations

* The data is simulated. The patterns follow how rings, mules, takeovers and card testing work in practice and
  each has genuine look-alikes, but real portfolios have drift, more patterns and messier labels; absolute
  numbers will not transfer, the comparisons and the leakage results are the point.
* The daily graph snapshot is rebuilt inside the request that crosses midnight (about a quarter of a second);
  production would run it off the request path and swap it in.
* The store is in-process and single-node; state is pickled at build time and the alert queue is SQLite.
* GraphSAGE results vary with the seed more than boosting does, and its training uses one thread to stay
  bit-reproducible.
* Thresholds are chosen on validation labels that are still incomplete, which biases them; the fixed-budget
  column shows the effect but the bias is not corrected.

## Roadmap

* Run the snapshot job asynchronously and serve embeddings from a versioned cache.
* Temporal GNN (time-aware neighbour sampling) instead of daily snapshots.
* Delay-aware threshold selection (reweighting validation labels by report-time survival).
* An optional LLM analyst note per alert behind a flag, built only from the reason codes.
