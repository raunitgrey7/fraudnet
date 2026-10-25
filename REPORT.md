# REPORT: fraudnet

What was built, how every published number was produced, what changed along the way and why, and what is
still missing. Companion documents: `README.md` (overview and results), `docs/ARCHITECTURE.md` (modules and
data flow), `docs/EVALUATION.md` (methodology and metric definitions).

## 1. What was built

| Area | Delivered |
|---|---|
| Data | Seeded agent-based simulator (`sim/`): 5,000 customers, devices (personal, second, replacement, household, workplace), 400 merchants, 90 days of purchases and transfers; four planted patterns (rings, mule chains, account takeover in two styles, card testing as burst or device farm) with attack ids; eight genuine look-alike behaviours; pattern-specific report delays and lost reports; cached with a SHA-256 check |
| Feature store | `OnlineFeatureStore`: 53 point-in-time features in 13 groups, scheduled fraud reports, daily graph snapshots (degree, PageRank, fraud-seeded personalised PageRank, label-propagation communities, GNN node features); offline replay through the same object; pickled build-day checkpoint for serving |
| Leakage proof | `features/leakage.py`: prefix replay and scrambled-future checks; passes for the honest pipeline on the real dataset, fails for a label leak and for a whole-log aggregate (tests and benchmark) |
| Models | Rules baseline (11 rules with reason codes); scikit-learn histogram boosting with validation-chosen tree count; heterogeneous GraphSAGE in plain PyTorch (`torch.sparse`), six relation types, unknown-entity nodes, per-snapshot embeddings |
| Evaluation | Time split with a label-maturity gap, PR-AUC, ROC-AUC, recall at 0.5 and 1 percent FPR, precision at top-10 and top-50 per day, money saved under a stated cost model (validation threshold and fixed budget), per-pattern and per-attack recall, day-block and paired bootstrap intervals, graph feature ablation, protocol ablation, cost of a label leak, latency, three seeds |
| Serving | FastAPI with Pydantic schemas, request ids, JSON logs, Prometheus metrics (including a scrape-time collector over service counters), optional API key, reason codes by group occlusion, decisions from validation thresholds, SQLite alert queue with analyst feedback into the store, chargeback endpoint, replay of the held-out stream, ring and network views, a vanilla JS console |
| Tooling | `python -m fraudnet.bench` (results.json, results.md, README and REPORT tables, threshold gate), `python -m fraudnet.tune` (validation-only sweep), `fraudnet` CLI (simulate, train, serve, leakage), Makefile, multi-stage Dockerfile with a non-root runtime, GitHub Actions workflow with the benchmark gate |
| Tests | Unit, integration, property-based (Hypothesis), adversarial and robustness, determinism, point-in-time, API contract and docs consistency tests; the count and the run are in section 6 |

## 2. Measured results

All numbers below come from `make bench` (seed 42, extra seeds 7 and 2026) on this machine (Apple Silicon,
CPU only, two threads). `benchmarks/results.json` is the source of truth; the section between the markers is
generated from it by `python -m fraudnet.bench --update-readme`, and `tests/test_docs.py` checks the README and
this file against it.

<!-- MEASURED:BEGIN -->
Seed 42, full profile, generated 2026-10-25T05:49:58Z, commit `ab23bad`, Python 3.13.17, torch 2.14.1, scikit-learn 1.9.1 on arm64 with 2 threads. Total benchmark time 483.8 s.

### Dataset

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
| rules | 0.070 | 0.078 | 0.090 | 0.199 | 14131.8 |
| rules_no_reasons | 0.049 | 0.151 | 0.241 | 0.455 | 15194.7 |
| gbdt | 1.282 | 1.777 | 3.185 | 76.702 | 695.0 |
| gbdt_no_reasons | 1.200 | 1.458 | 1.584 | 13.555 | 810.2 |
| graphsage | 0.066 | 0.072 | 0.083 | 0.165 | 14894.5 |
| graphsage_no_reasons | 0.044 | 0.049 | 0.061 | 0.150 | 22306.7 |
| gbdt over HTTP, server side | 1.600 | 1.927 | 2.117 | 2.553 | 611.5 |
| gbdt over HTTP, client side | 2.148 | 2.517 | 2.773 | 3.229 | 458.0 |

2000 test transactions scored one at a time after 50 untimed warm-up transactions; each path covers feature lookup, scoring and reason codes unless marked no_reasons. Online features equal the offline replay features (max absolute difference 0.0). A daily graph refresh over 12,144 nodes takes 0.236 s plus 0.008 s of GNN embedding.

### Seeds

| seed | rules | gbdt_tabular | gbdt | graphsage |
|---|---|---|---|---|
| 42 | 0.4829 | 0.7175 | 0.8772 | 0.7204 |
| 7 | 0.5890 | 0.8178 | 0.9370 | 0.9586 |
| 2026 | 0.5237 | 0.6525 | 0.9071 | 0.8974 |
| mean | 0.5319 | 0.7293 | 0.9071 | 0.8588 |
<!-- MEASURED:END -->

## 3. How the design got here (development measurements)

These measurements were taken during development with scratch scripts, some on earlier versions of the code
or of the simulator, so they are not reproduced by a command in the repository. They are recorded because
they explain decisions; none of them is quoted in the README.

* **The first simulator was too easy.** With fraud that had a clean tabular signature and no genuine
  look-alikes, tabular-only boosting reached a test PR-AUC of 0.920 and the rules 0.744, and every model
  caught 99 percent of fraud at 1 percent FPR, which says nothing about graph features. I added camouflage
  (low-and-slow takeovers, device-farm card testing, slower mule hops with reused payees and exit accounts,
  rings that warm up like new customers) and a genuine twin for every pattern (shared workplace devices,
  trips, sprees, in-app bursts, landlords, forwarding, large transfers). On the next version tabular boosting
  fell to 0.733 and the rules to 0.567, while boosting with graph features stayed at 0.864.
* **Validation labels decide what early stopping can see.** With validation running up to the build day, 65
  percent of its fraud was unlabelled and the validation average precision of the boosted model was about
  0.58 while its test PR-AUC was 0.86. Leaving a one-week maturity gap raised validation average precision to
  about 0.90 and made model selection meaningful. The benchmark keeps the no-gap protocol as a measured
  comparison (section 2).
* **Loss-based early stopping stopped too early.** scikit-learn's built-in early stopping on validation log
  loss ended after 56 to 68 trees, because confident fraud predictions on not-yet-labelled fraud look like
  errors under log loss. Tree count is now chosen by validation average precision from staged predictions,
  which costs one pass over the trees.
* **Thread pools.** Single-row prediction of the boosted model took 1.37 ms in the latency loop with the
  default OpenMP pool and 0.32 ms with two threads; one HTTP latency run measured 20.8 ms per request because
  FastAPI worker threads did not inherit the main thread's limit. Native pools now default to two threads at
  import (`OMP_NUM_THREADS`, overridable) and the service pins them; the HTTP latency in section 2 was measured
  after that change.
* **GNN determinism.** With two torch threads, two training runs of eight epochs on the full world differed by
  up to 3.9e-7 in their predictions (enough to move early stopping over 40 epochs); with one thread they were
  bit-identical and took 8.7 s against 8.2 s. GNN training now uses one thread, and
  `tests/test_models.py::test_gnn_is_bit_identical_where_torch_runs_in_parallel` checks it on the quick world.
* **Bugs found by the checks.** (1) The scrambled-future check first planted fake late reports on random past
  rows; when it hit a real fraud already reported before the cut it moved that report into the future, so the
  honest pipeline appeared to fail. Fake reports now go on genuine rows only. (2) A long mule chain could make
  a hop's report earlier than the hop itself; reports now count from the later of the victim's report and the
  hop (caught by `test_reports_arrive_after_the_fraud_and_never_for_genuine`). (3) The dataset cache was keyed
  on the config only, so a generator change silently reused the stale world; the key now includes a
  simulator version.
* **Cost model.** The first cost model charged only the review (5 per alert). With an average missed fraud
  costing about 270, the break-even alert precision was 2 percent and the rules looked best on money by
  alerting 204 times a day. A false-alert friction cost of 10 was added and stated, and the fixed-budget view
  (top 50 alerts per day) was added to separate ranking quality from threshold calibration.

## 4. Hyperparameters

Chosen by `python -m fraudnet.tune` (`make tune`), which fits each configuration on the train window and
scores it on the validation window with build-day labels; test labels are never read. Output on the final
world (seed 42), 3 min 13 s:

```
gbdt {'learning_rate': 0.05, 'max_leaf_nodes': 15, 'trees': 160, 'val_ap': 0.8585, 'seconds': 6.4}
gbdt {'learning_rate': 0.05, 'max_leaf_nodes': 31, 'trees': 300, 'val_ap': 0.8643, 'seconds': 8.8}
gbdt {'learning_rate': 0.05, 'max_leaf_nodes': 63, 'trees': 300, 'val_ap': 0.8669, 'seconds': 10.2}
gbdt {'learning_rate': 0.1, 'max_leaf_nodes': 15, 'trees': 400, 'val_ap': 0.8502, 'seconds': 6.2}
gbdt {'learning_rate': 0.1, 'max_leaf_nodes': 31, 'trees': 340, 'val_ap': 0.8472, 'seconds': 7.7}
gbdt {'learning_rate': 0.1, 'max_leaf_nodes': 63, 'trees': 260, 'val_ap': 0.8475, 'seconds': 9.3}
graphsage {'hidden': 48, 'lr': 0.003, 'best_epoch': 14, 'val_ap': 0.8908, 'seconds': 27.0}
graphsage {'hidden': 64, 'lr': 0.002, 'best_epoch': 14, 'val_ap': 0.8915, 'seconds': 28.6}
graphsage {'hidden': 32, 'lr': 0.005, 'best_epoch': 22, 'val_ap': 0.9007, 'seconds': 30.8}
graphsage {'hidden': 64, 'lr': 0.002, 'negative_rate': 0.6, 'best_epoch': 14, 'val_ap': 0.8994, 'seconds': 30.5}
```

The defaults are the winners: boosting with learning rate 0.05 and 63 leaves, GraphSAGE with width 32 and
learning rate 0.005 (40 epochs, patience 6). An earlier sweep on a previous version of the world had picked 31
leaves; the defaults were updated when the sweep was rerun on the final world. During development I also
looked at test metrics for four boosting settings once, before the sweep existed; the final choice was made by
the sweep on validation.

## 5. Quality gate

`benchmarks/thresholds.json` (checked by `make bench`, `make bench-quick`, `make check` and CI). The bounds
were set after the first full and quick runs on the final code and sit below the measured values; no bound was
relaxed after a failure.

| Profile | Check | Measured | Bound |
|---|---|---|---|
| full | gbdt PR-AUC | 0.8772 | at least 0.80 |
| full | gbdt recall at 1 percent FPR | 0.9733 | at least 0.90 |
| full | gbdt share of fraud loss saved | 0.7470 | at least 0.60 |
| full | gbdt ring recall at 1 percent FPR | 0.9479 | at least 0.80 |
| full | gbdt_tabular, graphsage, rules PR-AUC | 0.7175, 0.7204, 0.4829 | at least 0.60, 0.60, 0.35 |
| full | graph features over tabular (PR-AUC) | +0.1597 | at least +0.05 |
| full | honest pipeline passes the point-in-time check; leaky pipeline is caught | passed; caught | exact |
| full | online versus offline features | 0.0 | exactly 0.0 |
| full | gbdt p95 latency in process; server-side over HTTP | 1.777 ms; 1.927 ms | at most 10 ms; 25 ms |
| quick | gbdt PR-AUC; recall at 1 percent FPR | 0.5918; 0.6194 | at least 0.45; 0.50 |
| quick | graphsage, rules PR-AUC | 0.6527, 0.5227 | at least 0.45, 0.35 |
| quick | graph features over tabular | +0.0764 | at least +0.02 |
| quick | point-in-time, parity, latency | passed, 0.0, 3.603 ms, 5.218 ms | exact, 0.0, 25 ms, 50 ms |

The quick values are from `python -m fraudnet.bench --quick` (102.3 s). The quick world has only six rings over
60 days and most of its training rings are still unreported at build time, so its ring recall is low (0.1203 for
gbdt); it is a smoke gate for CI, not a result. Latency values in this table are from the first full run and
change slightly from run to run; section 2 has the latest.

## 6. Commands and outputs

<!-- CLEAN-RUN -->

## 7. Known gaps

* **Simulated data only.** The patterns are modelled on how each fraud type works and each has a genuine
  twin, but no real transaction data was available. Absolute numbers will not transfer; the comparisons
  between models, the ablation and the leakage results are the point.
* **GraphSAGE is unstable across worlds** (PR-AUC 0.7204, 0.9586 and 0.8974 on seeds 42, 7 and 2026) and
  trails boosting on average. I did not find a principled change that closed the gap on validation; the
  sweep covered width, learning rate and negative sampling only. Neighbour sampling, attention aggregation and
  a temporal model are untested.
* **The maturity gap is a trade-off.** It helped boosting on seed 42 and hurt the GNN; it was not evaluated on
  the other seeds.
* **Thresholds use incomplete validation labels.** The money-optimal threshold is chosen on labels that are
  still partly missing at build time; the bias is shown (fixed-budget column, protocol table) but not
  corrected.
* **Snapshot refresh runs in the request path** of the first transaction after midnight (about a quarter of a
  second); a production deployment would build it off the request path.
* **Single node.** The store lives in one process and is restored from a pickle; the alert queue is SQLite.
  There is no horizontal scaling story.
* **Latency is measured in process** (and through the in-process ASGI client), not over a network with
  concurrent clients.
* **No LLM component.** The charter allows an optional one; reason codes are deterministic and none was added.
